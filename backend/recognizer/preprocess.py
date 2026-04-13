"""
Image preprocessing utilities.

Handles:
  - Loading images in any common format
  - Deskewing (correcting tilted photos)
  - Binarization / adaptive thresholding
  - Noise removal
  - Normalization for model input
"""

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline.
    Returns a cleaned, deskewed, binarized uint8 numpy array (grayscale).
    """
    img = _load_image(image_path)
    img = _to_grayscale(img)
    img = _remove_noise(img)
    img = _deskew(img)
    img = _binarize(img)
    return img


def preprocess_for_model(image_path: str) -> Image.Image:
    """
    Preprocess and return a PIL Image ready for TrOCR input.
    TrOCR expects an RGB PIL image — we convert the cleaned grayscale to RGB.
    """
    arr = preprocess_image(image_path)
    # Convert grayscale binary back to RGB for model compatibility
    rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image at path: {path}")
    return img


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _remove_noise(img: np.ndarray) -> np.ndarray:
    """Median blur to remove salt-and-pepper noise from photo."""
    return cv2.medianBlur(img, 3)


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Estimate and correct skew angle using Hough line transform.
    Works best when there are clear horizontal lines of text.
    """
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

    if lines is None:
        return img  # Can't detect lines — return as-is

    angles = []
    for line in lines[:20]:  # Use top 20 strongest lines
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:  # Ignore near-vertical lines
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img  # Close enough — skip rotation to avoid artifacts

    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, scale=1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _binarize(img: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding — handles uneven lighting from phone photos much
    better than a global threshold.
    """
    return cv2.adaptiveThreshold(
        img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )


def crop_to_text(img: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Crop away empty whitespace borders, leaving only the text region.
    Useful before passing line crops into the model.
    """
    # Invert so text is white on black, find bounding box of non-zero pixels
    inv = cv2.bitwise_not(img)
    coords = cv2.findNonZero(inv)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    return img[y : y + h, x : x + w]
