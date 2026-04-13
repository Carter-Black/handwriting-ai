"""
Handwriting Recognition model — wraps Microsoft TrOCR.

Two modes:
  1. Out-of-the-box transcription using the pretrained model.
  2. Fine-tuning the last few transformer layers on the user's own samples,
     using the known prompt paragraph as ground-truth labels (self-supervised).

The fine-tuned model weights are saved locally so training only happens once.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from torch.utils.data import Dataset

from recognizer.preprocess import preprocess_for_model, preprocess_image, crop_to_text
import cv2

MODEL_NAME = "microsoft/trocr-base-handwritten"
FINE_TUNED_DIR = "fine_tuned_trocr"


class HandwritingRecognizer:
    """
    Wraps TrOCR for local handwriting recognition.

    Usage:
        r = HandwritingRecognizer(model_dir="/path/to/storage/models")
        text = r.transcribe(pil_image_or_numpy_array)
        r.fine_tune(["/path/to/sample1.jpg", ...], ground_truth="The quick...")
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor: Optional[TrOCRProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None
        self._load_model()

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(self, image: np.ndarray | Image.Image) -> str:
        """
        Transcribe handwriting from an image.
        Accepts either a numpy array (grayscale or RGB) or a PIL Image.
        """
        pil_img = self._ensure_pil_rgb(image)
        lines = self._split_into_lines(pil_img)

        transcribed_lines = []
        for line_img in lines:
            pixel_values = self.processor(
                images=line_img, return_tensors="pt"
            ).pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            transcribed_lines.append(text.strip())

        return "\n".join(transcribed_lines)

    def fine_tune(
        self,
        image_paths: list[str],
        ground_truth: str,
        epochs: int = 3,
        learning_rate: float = 5e-5,
    ):
        """
        Fine-tune TrOCR on the user's handwriting samples.

        We use the known prompt paragraph as the label for the full image.
        For multi-line images, we line-segment and assign label portions
        heuristically (best-effort for a local, no-annotation approach).

        Saves the fine-tuned model to model_dir/fine_tuned_trocr/.
        """
        print("Preparing fine-tuning dataset...")
        pairs: list[tuple[Image.Image, str]] = []

        for path in image_paths:
            pil = preprocess_for_model(path)
            lines = self._split_into_lines(pil)
            gt_lines = self._split_ground_truth(ground_truth, len(lines))
            for img_line, gt_line in zip(lines, gt_lines):
                if gt_line.strip():
                    pairs.append((img_line, gt_line))

        if not pairs:
            raise ValueError("No valid image/label pairs could be prepared.")

        dataset = _HandwritingDataset(pairs, self.processor)

        # Only fine-tune the decoder — freeze the encoder vision backbone
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        output_dir = str(self.model_dir / FINE_TUNED_DIR)
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
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
            args=training_args,
            train_dataset=dataset,
            data_collator=default_data_collator,
        )

        print(f"Fine-tuning on {len(pairs)} line pairs for {epochs} epoch(s)...")
        trainer.train()

        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Fine-tuned model saved to {output_dir}")

        # Re-enable all params after training
        for param in self.model.parameters():
            param.requires_grad = True

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_model(self):
        """Load fine-tuned model if available, otherwise load pretrained."""
        fine_tuned = self.model_dir / FINE_TUNED_DIR
        source = str(fine_tuned) if fine_tuned.exists() else MODEL_NAME

        print(f"Loading TrOCR from: {source}")
        self.processor = TrOCRProcessor.from_pretrained(source)
        self.model = VisionEncoderDecoderModel.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()
        print("Model ready.")

    def _ensure_pil_rgb(self, image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if len(image.shape) == 2:
            # Grayscale → RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _split_into_lines(self, pil_img: Image.Image) -> list[Image.Image]:
        """
        Split a full-page handwriting image into individual line images.
        Uses horizontal projection profile on the binarized image.
        """
        arr = np.array(pil_img.convert("L"))
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal projection: sum of white pixels per row
        h_proj = np.sum(binary, axis=1)
        threshold = h_proj.max() * 0.05  # rows with < 5% of max are "gaps"

        in_line = False
        line_starts = []
        line_ends = []

        for i, val in enumerate(h_proj):
            if not in_line and val > threshold:
                in_line = True
                line_starts.append(i)
            elif in_line and val <= threshold:
                in_line = False
                line_ends.append(i)

        if in_line:
            line_ends.append(len(h_proj))

        if not line_starts:
            return [pil_img]  # Couldn't split — return whole image

        lines = []
        padding = 5
        w = pil_img.width
        for top, bottom in zip(line_starts, line_ends):
            top = max(0, top - padding)
            bottom = min(pil_img.height, bottom + padding)
            line_crop = pil_img.crop((0, top, w, bottom))
            lines.append(line_crop)

        return lines

    @staticmethod
    def _split_ground_truth(text: str, n_lines: int) -> list[str]:
        """
        Split the ground-truth paragraph into approximately n_lines chunks.
        Simple word-based split — good enough for few-shot fine-tuning.
        """
        words = text.split()
        if n_lines <= 1:
            return [text]
        chunk_size = max(1, len(words) // n_lines)
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i : i + chunk_size]))
        # If we have more chunks than lines, merge the last ones
        while len(chunks) > n_lines:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()
        return chunks


# ── Dataset ────────────────────────────────────────────────────────────────────

class _HandwritingDataset(Dataset):
    """Simple dataset of (image, label) pairs for Seq2SeqTrainer."""

    def __init__(self, pairs: list[tuple[Image.Image, str]], processor: TrOCRProcessor):
        self.pairs = pairs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        image, label = self.pairs[idx]
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        with self.processor.tokenizer as tok:
            labels = tok(
                label,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)

        # Replace padding token id with -100 so loss ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}
