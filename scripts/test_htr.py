#!/usr/bin/env python3
"""
test_htr.py — Module 1 test script

Tests the full HTR pipeline:
  1. Image quality check
  2. Preprocessing (load → grayscale → deskew → binarize)
  3. TrOCR transcription

Usage:
    cd handwriting-ai
    source .venv/bin/activate          # (or .venv\\Scripts\\activate on Windows)
    python scripts/test_htr.py path/to/your/photo.jpg

    # Test with multiple images:
    python scripts/test_htr.py photo1.jpg photo2.jpg

    # Skip quality check (e.g. to test a deliberately bad image):
    python scripts/test_htr.py photo.jpg --no-quality-check

    # Save the preprocessed image so you can visually inspect it:
    python scripts/test_htr.py photo.jpg --save-preprocessed

Output:
    - Quality report (warnings / errors / pass)
    - Preprocessing stats (dimensions, mean brightness)
    - Detected line count
    - Full transcription text
    - Timing info for each stage
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from the repo root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import cv2
import numpy as np

from recognizer.preprocess import (
    check_quality,
    preprocess_image,
    preprocess_for_model,
)
from recognizer.model import HandwritingRecognizer

MODELS_DIR = Path(__file__).parent.parent / "backend" / "storage" / "models"


def hr(char: str = "─", width: int = 60) -> str:
    return char * width


def fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def test_image(
    image_path: str,
    recognizer: HandwritingRecognizer,
    run_quality: bool = True,
    save_preprocessed: bool = False,
) -> bool:
    """
    Run the full pipeline on one image. Returns True if transcription succeeded.
    """
    path = Path(image_path)
    print(f"\n{hr()}")
    print(f"Image: {path.name}")
    print(hr())

    # ── 1. Quality check ───────────────────────────────────────────────────────
    if run_quality:
        print("\n[1/3] Quality check...")
        t0 = time.time()
        report = check_quality(image_path)
        elapsed = time.time() - t0

        print(f"  {report.summary()}")
        print(f"  ({fmt_time(elapsed)})")

        if not report.ok:
            print("\n  ❌ Stopping — image failed quality check.")
            print("  Fix the issues above and try again.")
            return False
    else:
        print("\n[1/3] Quality check: SKIPPED")

    # ── 2. Preprocessing ───────────────────────────────────────────────────────
    print("\n[2/3] Preprocessing...")
    t0 = time.time()
    processed = preprocess_image(image_path)
    elapsed = time.time() - t0

    h, w = processed.shape
    ink_frac = float(np.sum(processed == 0)) / processed.size  # black pixels = ink
    print(f"  Output size:      {w} × {h} px")
    print(f"  Ink coverage:     {ink_frac * 100:.1f}% of image")
    print(f"  ({fmt_time(elapsed)})")

    if save_preprocessed:
        out_path = path.parent / f"{path.stem}_preprocessed.png"
        cv2.imwrite(str(out_path), processed)
        print(f"  Saved preprocessed image → {out_path}")

    # ── 3. Transcription ───────────────────────────────────────────────────────
    print("\n[3/3] Transcription...")
    t0 = time.time()

    pil_img = preprocess_for_model(image_path)
    lines_detected = [None]  # mutable container for callback

    def progress_cb(current: int, total: int):
        if lines_detected[0] is None:
            lines_detected[0] = total
            print(f"  Lines detected:   {total}")
        if current > 0:
            print(f"  Transcribing line {current}/{total}...", end="\r", flush=True)

    text = recognizer.transcribe(pil_img, progress_cb=progress_cb)
    elapsed = time.time() - t0

    print(f"\n  Done ({fmt_time(elapsed)})")
    print(f"\n{'─' * 40}")
    print("TRANSCRIPTION OUTPUT:")
    print('─' * 40)
    print(text)
    print('─' * 40)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test the HandwritingAI HTR pipeline on one or more images."
    )
    parser.add_argument("images", nargs="+", help="Path(s) to handwriting image(s)")
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Skip the image quality check step",
    )
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save the preprocessed binary image alongside the input",
    )
    args = parser.parse_args()

    # Validate files exist
    valid_paths = []
    for p in args.images:
        if not Path(p).exists():
            print(f"⚠️  File not found: {p}", file=sys.stderr)
        else:
            valid_paths.append(p)

    if not valid_paths:
        print("❌ No valid image files provided.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print("  HandwritingAI — HTR Pipeline Test")
    print(f"{'═' * 60}")
    print(f"\nLoading TrOCR model from: {MODELS_DIR}")

    t_model_start = time.time()
    try:
        recognizer = HandwritingRecognizer(model_dir=str(MODELS_DIR))
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\nMake sure you've run the installer first:")
        print("  ./install.sh   (macOS/Linux)")
        print("  install.bat    (Windows)")
        sys.exit(1)

    t_model = time.time() - t_model_start
    print(f"Model loaded in {fmt_time(t_model)}")

    # Run tests
    successes = 0
    for img_path in valid_paths:
        ok = test_image(
            img_path,
            recognizer,
            run_quality=not args.no_quality_check,
            save_preprocessed=args.save_preprocessed,
        )
        if ok:
            successes += 1

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  Results: {successes}/{len(valid_paths)} images transcribed successfully")
    print(f"{'═' * 60}\n")

    if successes < len(valid_paths):
        sys.exit(1)


if __name__ == "__main__":
    main()
