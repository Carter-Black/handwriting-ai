# preprocess.py
# image quality validation and preprocessing pipeline for handwriting photos

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# quality thresholds
MIN_WIDTH = 400
MIN_HEIGHT = 200
BLUR_THRESHOLD = 80.0
DARK_THRESHOLD = 50.0
BRIGHT_THRESHOLD = 230.0


@dataclass
class QualityReport:
    ok: bool
    warnings: list[str]
    errors: list[str]

    def summary(self) -> str:
        lines = [f"❌ {e}" for e in self.errors] + [f"⚠️  {w}" for w in self.warnings]
        return "\n".join(lines) if lines else "✅ Image quality looks good."


def check_quality(image_path: str) -> QualityReport:
    """
    Validate a raw photo before processing. Returns a QualityReport.
    Call this before preprocess_image() so the user gets useful feedback
    if their photo is blurry, too dark, or too small.
    """
    img = _load(image_path)
    gray = _gray(img)
    h, w = gray.shape
    errors, warnings = [], []

    # resolution
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        errors.append(f"Image too small ({w}×{h}px, need {MIN_WIDTH}×{MIN_HEIGHT}px minimum).")

    # blur — Laplacian variance drops sharply on blurry images
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < BLUR_THRESHOLD:
        level = "very" if blur_score < BLUR_THRESHOLD / 2 else "slightly"
        errors.append(f"Image is {level} blurry (score {blur_score:.0f}, need ≥{BLUR_THRESHOLD:.0f}). Hold camera still.")

    # exposure
    brightness = float(gray.mean())
    if brightness < DARK_THRESHOLD:
        errors.append(f"Image too dark (brightness {brightness:.0f}/255). Use better lighting.")
    elif brightness > BRIGHT_THRESHOLD:
        warnings.append(f"Image may be overexposed (brightness {brightness:.0f}/255).")

    # ink coverage check
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = float(np.sum(binary > 0)) / binary.size
    if ink < 0.005:
        errors.append("Almost no ink detected. Make sure handwriting fills the frame.")
    elif ink > 0.6:
        warnings.append("Image is very dark overall — is the paper visible?")

    return QualityReport(ok=len(errors) == 0, warnings=warnings, errors=errors)


def preprocess_image(image_path: str) -> np.ndarray:
    """Full pipeline: load → grayscale → scale → denoise → deskew → binarize."""
    img = _load(image_path)
    img = _gray(img)
    img = _upscale(img)
    img = _denoise(img)
    img = _deskew(img)
    img = _binarize(img)
    return img


def preprocess_for_model(image_path: str) -> Image.Image:
    """Preprocess and return an RGB PIL image ready for TrOCR."""
    arr = preprocess_image(image_path)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))


# pipeline steps

def _load(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load: {path}")
    return img


def _gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _upscale(img: np.ndarray, target_h: int = 1000) -> np.ndarray:
    """Scale up small images — TrOCR was pretrained on higher-res crops."""
    h, w = img.shape
    if h >= target_h:
        return img
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)


def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(img, 3)


def _deskew(img: np.ndarray) -> np.ndarray:
    """Detect and correct page skew using Hough lines. Clamped to ±15°."""
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
    if lines is None:
        return img

    angles = []
    for line in lines[:30]:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img

    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return img

    angle = max(-15.0, min(15.0, angle))
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _binarize(img: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold — handles uneven phone-photo lighting."""
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )


def crop_to_text(img: np.ndarray, padding: int = 10) -> np.ndarray:
    """Crop whitespace borders, keeping only the text region."""
    inv = cv2.bitwise_not(img)
    coords = cv2.findNonZero(inv)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    return img[y:y + h, x:x + w]
