# model.py
# TrOCR wrapper with optional fine-tuning on the user's own handwriting

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

from recognizer.preprocess import preprocess_for_model

MODEL_NAME = "microsoft/trocr-base-handwritten"
FINE_TUNED_DIR = "fine_tuned_trocr"

# beam search gives meaningfully better results than greedy for handwriting
GENERATE_KWARGS = {
    "max_new_tokens": 128,
    "num_beams": 4,
    "early_stopping": True,
}


class HandwritingRecognizer:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor: Optional[TrOCRProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None
        self._load()

    def transcribe(
        self,
        image: np.ndarray | Image.Image,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Transcribe handwriting from an image. Splits into lines internally.
        progress_cb(current_line, total_lines) is called if provided.
        """
        pil = self._to_pil(image)
        lines = self._split_lines(pil)
        total = len(lines)
        out = []

        for i, line in enumerate(lines):
            if progress_cb:
                progress_cb(i, total)

            px = self.processor(images=line, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                ids = self.model.generate(px, **GENERATE_KWARGS)
            out.append(self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip())

        if progress_cb:
            progress_cb(total, total)
        return "\n".join(out)

    def fine_tune(
        self,
        image_paths: list[str],
        ground_truth: str,
        epochs: int = 3,
        learning_rate: float = 5e-5,
    ):
        """
        Fine-tune TrOCR on the user's samples.
        Uses the known prompt paragraph as ground-truth labels (self-supervised).
        Saves the fine-tuned model locally so it only needs to run once.
        """
        print("Building fine-tune dataset...")
        pairs: list[tuple[Image.Image, str]] = []

        for path in image_paths:
            pil = preprocess_for_model(path)
            lines = self._split_lines(pil)
            labels = _split_text(ground_truth, len(lines))
            for img_line, label in zip(lines, labels):
                if label.strip():
                    pairs.append((img_line, label))

        if not pairs:
            raise ValueError("No usable image/label pairs found.")

        # freeze encoder — only update decoder layers
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        # VisionEncoderDecoderConfig does not auto-populate pad/start/eos
        # tokens on the top-level config, but Seq2SeqTrainer reads them from
        # there. Copy them up from the tokenizer before training.
        tok = self.processor.tokenizer
        self.model.config.pad_token_id = tok.pad_token_id
        self.model.config.decoder_start_token_id = tok.cls_token_id
        self.model.config.eos_token_id = tok.sep_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.pad_token_id = tok.pad_token_id
            self.model.generation_config.decoder_start_token_id = tok.cls_token_id
            self.model.generation_config.eos_token_id = tok.sep_token_id

        out_dir = str(self.model_dir / FINE_TUNED_DIR)
        args = Seq2SeqTrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            predict_with_generate=True,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            report_to="none",
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=_HTRDataset(pairs, self.processor),
            data_collator=default_data_collator,
        )

        print(f"Fine-tuning on {len(pairs)} line pairs for {epochs} epoch(s)...")
        trainer.train()
        self.model.save_pretrained(out_dir)
        self.processor.save_pretrained(out_dir)
        print(f"Saved fine-tuned model → {out_dir}")

        for p in self.model.parameters():
            p.requires_grad = True

    # internal

    def _load(self):
        fine_tuned = self.model_dir / FINE_TUNED_DIR
        src = str(fine_tuned) if fine_tuned.exists() else MODEL_NAME
        print(f"Loading TrOCR from: {src}")
        self.processor = TrOCRProcessor.from_pretrained(src)
        self.model = VisionEncoderDecoderModel.from_pretrained(src)
        self.model.to(self.device).eval()
        print("Model ready.")

    def _to_pil(self, image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if len(image.shape) == 2 else cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr)

    def _split_lines(self, pil: Image.Image) -> list[Image.Image]:
        """Split a full-page image into line strips via horizontal projection."""
        gray = np.array(pil.convert("L"))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj = np.sum(binary, axis=1)
        thresh = proj.max() * 0.05

        starts, ends, in_line = [], [], False
        for i, v in enumerate(proj):
            if not in_line and v > thresh:
                in_line = True
                starts.append(i)
            elif in_line and v <= thresh:
                in_line = False
                ends.append(i)
        if in_line:
            ends.append(len(proj))

        if not starts:
            return [pil]

        pad, w = 5, pil.width
        lines = []
        for top, bot in zip(starts, ends):
            lines.append(pil.crop((0, max(0, top - pad), w, min(pil.height, bot + pad))))
        return lines


# dataset for fine-tuning

class _HTRDataset(Dataset):
    def __init__(self, pairs: list[tuple[Image.Image, str]], processor: TrOCRProcessor):
        self.pairs = pairs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        image, label = self.pairs[idx]
        px = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        tok = self.processor.tokenizer
        ids = tok(label, padding="max_length", max_length=128,
                  truncation=True, return_tensors="pt").input_ids.squeeze(0)
        ids[ids == tok.pad_token_id] = -100  # ignore padding in loss

        return {"pixel_values": px, "labels": ids}


# helpers

def _split_text(text: str, n: int) -> list[str]:
    """Split ground-truth text into ~n roughly equal chunks by word count."""
    words = text.split()
    if n <= 1:
        return [text]
    size = max(1, len(words) // n)
    chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    while len(chunks) > n:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()
    return chunks
