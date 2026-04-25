# model.py
# TrOCR wrapper with optional fine-tuning on the user's own handwriting

from __future__ import annotations

import random
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

MODEL_NAME_BASE = "microsoft/trocr-base-handwritten"
MODEL_NAME_LARGE = "microsoft/trocr-large-handwritten"
FINE_TUNED_DIR = "fine_tuned_trocr"  # holds the BASE model's fine-tuned variant only

# Canonical TrOCR generation config from the official Microsoft/HuggingFace
# tutorial (NielsRogge/Transformers-Tutorials, microsoft/unilm/trocr README).
# This is the same recipe shipped with microsoft/trocr-base-handwritten's
# generation_config.json. Per arXiv research: max_new_tokens MUST be 128 (not
# 80) — short caps combined with no_repeat_ngram_size=3 cause beam search
# deadlock on long lines. length_penalty=2.0 explicitly rewards longer
# sequences, which prevents premature <eos> on full-width handwriting lines.
# We DON'T use repetition_penalty: it distorts the RoBERTa-decoder LM prior
# and degrades natural sentences. _trim_repeats post-processor handles loops.
GENERATE_KWARGS = {
    "max_new_tokens": 128,
    "num_beams": 4,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "length_penalty": 2.0,
}


class HandwritingRecognizer:
    def __init__(
        self,
        model_dir: str,
        use_fine_tuned: bool = True,
        use_large_model: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fine_tuned = use_fine_tuned
        self.use_large_model = use_large_model
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
        print(f"[transcribe] image: {pil.width}x{pil.height} px")
        lines = self._split_lines(pil)
        total = len(lines)
        if total == 0:
            print(f"[transcribe] WARNING: no lines detected — image may be blank or threshold too high")
            return ""
        heights = [l.height for l in lines]
        print(f"[transcribe] detected {total} line(s); heights min={min(heights)} max={max(heights)} px")
        out = []

        for i, line in enumerate(lines):
            if progress_cb:
                progress_cb(i, total)

            px = self.processor(images=line, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                ids = self.model.generate(px, **GENERATE_KWARGS)
            raw = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            text = _trim_trailing_noise(_trim_repeats(raw))
            out.append(text)
            if text != raw:
                print(f"[transcribe]   line {i+1}/{total} (h={line.height}): {text!r} (cleaned)")
            else:
                print(f"[transcribe]   line {i+1}/{total} (h={line.height}): {text!r}")

        if progress_cb:
            progress_cb(total, total)
        return "\n".join(out)

    def fine_tune(
        self,
        image_paths: list[str],
        ground_truth: str,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        augment_multiplier: int = 10,
    ):
        """
        Fine-tune TrOCR on the user's samples.
        Uses the known prompt paragraph as ground-truth labels (self-supervised).
        Saves the fine-tuned model locally so it only needs to run once.
        """
        # Split ground truth on newlines — one expected label per visual line.
        # This is how the prompt is structured (4 lines on the page); each
        # detected handwritten line gets paired with its corresponding label.
        # No more word-count splitting → no more misaligned labels.
        expected_lines = [l.strip() for l in ground_truth.split("\n") if l.strip()]
        n_expected = len(expected_lines)

        print(f"[fine-tune] building dataset from {len(image_paths)} image(s)")
        print(f"[fine-tune] ground truth: {n_expected} expected line(s), {sum(len(l.split()) for l in expected_lines)} words total")
        pairs: list[tuple[Image.Image, str]] = []
        skipped = 0

        for idx, path in enumerate(image_paths):
            pil = preprocess_for_model(path)
            lines = self._split_lines(pil)

            if len(lines) != n_expected:
                print(f"[fine-tune] image {idx+1}/{len(image_paths)}: detected {len(lines)} lines, expected {n_expected} — SKIPPING")
                print(f"[fine-tune]   each image must contain the prompt written on exactly {n_expected} lines.")
                print(f"[fine-tune]   if your line splitter is detecting the wrong number, retake the photo")
                print(f"[fine-tune]   with cleaner spacing between rows.")
                skipped += 1
                continue

            print(f"[fine-tune] image {idx+1}/{len(image_paths)}: {len(lines)} lines aligned to {n_expected} ground-truth lines")
            for j, (img_line, label) in enumerate(zip(lines, expected_lines)):
                preview = label if len(label) <= 50 else label[:50] + "..."
                print(f"[fine-tune]   line {j+1} ↔ {preview!r}")
                pairs.append((img_line, label))

        if not pairs:
            raise ValueError(
                f"No usable image/label pairs. Each image must contain the prompt "
                f"written on exactly {n_expected} lines (matching the prompt's line "
                f"breaks). {skipped} image(s) skipped — see warnings above."
            )
        if skipped:
            print(f"[fine-tune] {skipped} image(s) skipped due to line-count mismatch")
        print(f"[fine-tune] real training pairs: {len(pairs)}")
        effective = len(pairs) * augment_multiplier
        print(f"[fine-tune] augmenting {augment_multiplier}× → effective dataset size {effective} per epoch")
        print(f"[fine-tune] {epochs} epoch(s) → {effective * epochs} total forward passes")

        # Train all parameters (encoder + decoder). DLoRA-TrOCR ablation
        # (arXiv:2404.12734v3) found that freezing the encoder and updating
        # only the decoder is the WORST PEFT strategy for TrOCR — the encoder
        # is where visual style adaptation happens (slant, stroke thickness,
        # baseline curvature), and freezing it eliminates the only place
        # those features can be learned. So we unfreeze everything.
        for p in self.model.parameters():
            p.requires_grad = True

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
            # "no" = don't write per-epoch checkpoint folders during training.
            # Each checkpoint includes the Adam optimizer state (~2x the model
            # size), so 3 epochs = ~12 GB of disk we never reuse. The final
            # model is saved explicitly via save_pretrained() after train().
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            report_to="none",
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=_HTRDataset(
                pairs, self.processor,
                multiplier=augment_multiplier,
                augment=True,
            ),
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
        # Selection logic:
        #   use_large_model=True  → trocr-large-handwritten (no fine-tune support;
        #                            fine-tuning the large model on CPU is impractical,
        #                            and the saved fine-tune in storage is BASE-only).
        #   use_large_model=False → trocr-base-handwritten, with the fine-tuned
        #                            variant if one exists and use_fine_tuned=True.
        if self.use_large_model:
            src = MODEL_NAME_LARGE
            label = "LARGE"
        else:
            fine_tuned = self.model_dir / FINE_TUNED_DIR
            if fine_tuned.exists() and self.use_fine_tuned:
                src = str(fine_tuned)
                label = "BASE+FINE-TUNED"
            else:
                src = MODEL_NAME_BASE
                label = "BASE"

        print(f"[model] device: {self.device}")
        print(f"[model] loading {label}: {src}")
        if label == "BASE+FINE-TUNED":
            print(f"[model] note: previously fine-tuned model. If transcription")
            print(f"[model]   quality is poor (e.g. repetitive phrases or hallucinations),")
            print(f"[model]   delete this folder to fall back to the base model:")
            print(f"[model]   {self.model_dir / FINE_TUNED_DIR}")
        elif label == "LARGE":
            print(f"[model] note: large model is ~558 MB; first load downloads it.")
            print(f"[model]   Saved fine-tuned models are BASE-only and not used here.")
        self.processor = TrOCRProcessor.from_pretrained(src)
        self.model = VisionEncoderDecoderModel.from_pretrained(src)
        self.model.to(self.device).eval()
        print(f"[model] ready")

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
        if proj.max() == 0:
            return [pil]
        thresh = proj.max() * 0.05

        runs = _projection_runs(proj, thresh)
        if not runs:
            return [pil]

        merged = _merge_text_runs(runs)
        pad, w = 5, pil.width
        return [pil.crop((0, max(0, t - pad), w, min(pil.height, b + pad))) for t, b in merged]


# dataset for fine-tuning

class _HTRDataset(Dataset):
    """
    Wraps (image, label) pairs with on-the-fly augmentation.

    The `multiplier` parameter virtually grows the dataset N× — each "real"
    pair is yielded `multiplier` times per epoch, with a different random
    DA2-style augmentation each time. This is how you get TrOCR fine-tuning
    to actually learn from a small handwriting sample without making the
    user write the prompt 100 times.

    Per the TrOCR paper §3.2, augmentations are: random rotation ±10°,
    Gaussian blur, dilation, erosion, downscaling, additive noise,
    brightness shift, and elastic deformation (Simard et al. 2003).
    Elastic deformation is documented as the highest-gain single
    augmentation for handwriting.
    """

    def __init__(
        self,
        pairs: list[tuple[Image.Image, str]],
        processor: TrOCRProcessor,
        multiplier: int = 10,
        augment: bool = True,
    ):
        self.pairs = pairs
        self.processor = processor
        self.multiplier = max(1, multiplier)
        self.augment = augment

    def __len__(self):
        return len(self.pairs) * self.multiplier

    def __getitem__(self, idx: int) -> dict:
        image, label = self.pairs[idx % len(self.pairs)]
        if self.augment:
            image = _augment(image)

        px = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        tok = self.processor.tokenizer
        ids = tok(label, padding="max_length", max_length=128,
                  truncation=True, return_tensors="pt").input_ids.squeeze(0)
        ids[ids == tok.pad_token_id] = -100  # ignore padding in loss

        return {"pixel_values": px, "labels": ids}


# helpers

_AUG_CHOICES = (
    "identity", "identity",  # double weight on no-op
    "rotate", "blur", "dilate", "erode",
    "downscale", "noise", "brightness", "elastic",
)


def _augment(image: Image.Image) -> Image.Image:
    """
    Apply one random DA2-style augmentation to an image.
    Each call returns a slightly perturbed version that's still recognizably
    the user's handwriting — the model sees these as "more samples" without
    requiring more handwritten input.
    """
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    h, w = arr.shape[:2]
    aug = random.choice(_AUG_CHOICES)
    bg = (255, 255, 255) if arr.ndim == 3 else 255

    if aug == "identity":
        return image

    if aug == "rotate":
        angle = random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=bg)

    elif aug == "blur":
        k = random.choice([3, 5])
        arr = cv2.GaussianBlur(arr, (k, k), 0)

    elif aug == "dilate":
        kernel = np.ones((2, 2), np.uint8)
        arr = cv2.dilate(arr, kernel, iterations=1)

    elif aug == "erode":
        kernel = np.ones((2, 2), np.uint8)
        arr = cv2.erode(arr, kernel, iterations=1)

    elif aug == "downscale":
        scale = random.uniform(0.6, 0.9)
        small = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))))
        arr = cv2.resize(small, (w, h))

    elif aug == "noise":
        noise = np.random.normal(0, 8, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif aug == "brightness":
        delta = random.randint(-25, 25)
        arr = np.clip(arr.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    elif aug == "elastic":
        arr = _elastic_transform(arr, alpha=15.0, sigma=3.0)

    return Image.fromarray(arr)


def _elastic_transform(img: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """
    Elastic deformation per Simard, Steinkraus & Platt (ICDAR 2003).
    Generates a smooth random displacement field and remaps the image
    through it — gives a wobbly, "natural" variation in stroke shape that
    research identifies as the highest-gain single augmentation for HTR
    (arXiv:2508.11499).
    """
    h, w = img.shape[:2]
    rng = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(rng, (0, 0), sigma) * alpha
    rng = np.random.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = cv2.GaussianBlur(rng, (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    bg = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )


def _trim_trailing_noise(text: str) -> str:
    """
    Strip trailing artifacts that TrOCR with length_penalty=2.0 sometimes
    appends to fill out beam scores past the visual content. Common
    patterns: trailing ellipsis ('...'), single isolated punctuation
    floating after the last meaningful word, runs of dots/commas.
    """
    import re as _re
    text = text.rstrip()
    # Repeatedly strip trailing ellipses and orphan punctuation tokens
    while True:
        prev = text
        text = _re.sub(r'\s*\.{2,}\s*$', '', text)            # trailing "..." / "...."
        text = _re.sub(r'\s+[.,;:!?\'"\-/()]+\s*$', '', text) # orphan punct after a space
        text = text.rstrip()
        if text == prev:
            break
    return text


def _trim_repeats(text: str, max_consecutive: int = 3) -> str:
    """
    Cap any run of identical consecutive whitespace-separated tokens at
    `max_consecutive` occurrences. Cleans up beam search loops like
    '9 9 9 9 9 9 9' or "' ' ' ' ' ' '" that TrOCR sometimes emits on
    out-of-distribution input (rows of standalone characters), without
    touching natural sentences where each word is unique.
    """
    tokens = text.split()
    out: list[str] = []
    run = 0
    for t in tokens:
        if out and out[-1] == t:
            run += 1
        else:
            run = 1
        if run > max_consecutive:
            continue
        out.append(t)
    return " ".join(out)


def _projection_runs(proj: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    """Return [(start, end)] for each contiguous run in proj that exceeds thresh."""
    runs, in_run, start = [], False, 0
    for i, v in enumerate(proj):
        if not in_run and v > thresh:
            in_run = True
            start = i
        elif in_run and v <= thresh:
            in_run = False
            runs.append((start, i))
    if in_run:
        runs.append((start, len(proj)))
    return runs


def _merge_text_runs(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Filter and merge raw projection runs into actual text-line bounds.

    Horizontal projection often picks up descender ('g', 'p', 'y') and dot
    ('i', 'j') fragments as their own short "lines" between real text rows.
    Without cleanup, these fragments get fed to TrOCR as if they were text
    (which produces hallucinated tokens like '8 8 8...') and to the fine-tuner
    as if they were valid label/image pairs (which poisons the model).

    Strategy:
      1. Treat any run shorter than 30% of the tallest run as a fragment.
      2. Absorb each fragment into the closest real line if it's within
         half a median-line-height — this preserves descenders/dots as part
         of their parent letter rather than dropping them.
    """
    if not runs:
        return []
    heights = [b - t for t, b in runs]
    max_h = max(heights)
    real = [(t, b) for t, b in runs if (b - t) >= max_h * 0.3]
    if not real:
        return [(min(t for t, _ in runs), max(b for _, b in runs))]

    fragments = [(t, b) for t, b in runs if (b - t) < max_h * 0.3]
    bounds = [list(r) for r in real]
    median_h = int(np.median([b - t for t, b in real])) if real else 0
    absorb_dist = median_h * 0.5

    for ft, fb in fragments:
        best, best_d = None, float("inf")
        for lb in bounds:
            d = min(abs(ft - lb[1]), abs(fb - lb[0]))
            if d < best_d:
                best_d, best = d, lb
        if best is not None and best_d <= absorb_dist:
            best[0] = min(best[0], ft)
            best[1] = max(best[1], fb)

    bounds.sort()
    return [(t, b) for t, b in bounds]
