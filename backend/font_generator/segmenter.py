"""
Character Segmentation

Takes a preprocessed (binarized) handwriting image and extracts individual
character images, labelled by their character value.

Strategy:
  1. Detect text lines via horizontal projection.
  2. Within each line, detect characters via vertical projection (works well
     for printed handwriting — a v2 could tackle connected cursive).
  3. Return a dict mapping character → best-quality binary image crop.

NOTE: This requires the user to have written the PROMPT_PARAGRAPH in print
      (not cursive), which is clearly communicated in the UI.
"""

from __future__ import annotations

import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# The characters we expect to be present in the prompt paragraph, in order.
# We use this to label the segmented characters automatically.
PROMPT_CHARS = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "0123456789!?,;:'\"()-/"
)

# Characters we actually care about building glyphs for
TARGET_CHARS = (
    string.ascii_lowercase
    + string.ascii_uppercase
    + string.digits
    + r"""!?.,;:'"()-/"""
)


def segment_characters(
    image: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Segment characters from a binarized handwriting image.

    Returns:
        dict mapping character string → binary numpy image (uint8, 0/255)
        Only characters found in TARGET_CHARS are included.
        If a character appears multiple times, we keep the clearest crop
        (largest bounding box area as a proxy for quality).
    """
    # Invert: text = white on black for connected component analysis
    if _is_mostly_white(image):
        inv = cv2.bitwise_not(image)
    else:
        inv = image.copy()

    lines = _extract_lines(inv)
    char_images: dict[str, list[np.ndarray]] = {}

    prompt_index = 0
    prompt_text = [c for c in PROMPT_CHARS]

    for line_img in lines:
        chars_in_line = _extract_chars_from_line(line_img)
        for char_img in chars_in_line:
            # Advance through prompt to find next non-space character
            while prompt_index < len(prompt_text) and prompt_text[prompt_index] == " ":
                prompt_index += 1
            if prompt_index >= len(prompt_text):
                break
            label = prompt_text[prompt_index]
            prompt_index += 1

            if label in TARGET_CHARS:
                char_images.setdefault(label, []).append(char_img)

    # For each character, pick the best crop (largest area = most ink detail)
    best: dict[str, np.ndarray] = {}
    for char, crops in char_images.items():
        best[char] = max(crops, key=lambda c: c.shape[0] * c.shape[1])

    return best


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_mostly_white(img: np.ndarray) -> bool:
    """True if image is light background (standard binarized output)."""
    return np.mean(img) > 127


def _extract_lines(inv: np.ndarray) -> list[np.ndarray]:
    """Split inverted image into line strips via horizontal projection."""
    h_proj = np.sum(inv, axis=1)
    gap_threshold = h_proj.max() * 0.03

    in_line = False
    starts, ends = [], []
    for i, val in enumerate(h_proj):
        if not in_line and val > gap_threshold:
            in_line = True
            starts.append(i)
        elif in_line and val <= gap_threshold:
            in_line = False
            ends.append(i)
    if in_line:
        ends.append(inv.shape[0])

    lines = []
    pad = 4
    for top, bot in zip(starts, ends):
        strip = inv[max(0, top - pad) : min(inv.shape[0], bot + pad), :]
        if strip.shape[0] > 5:
            lines.append(strip)

    return lines


def _extract_chars_from_line(line_inv: np.ndarray) -> list[np.ndarray]:
    """
    Split a line strip into individual character crops via vertical projection
    and connected component analysis.
    """
    # Connected components (more robust than pure projection for printed text)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_inv, connectivity=8
    )

    line_h = line_inv.shape[0]
    char_crops = []

    # Collect bounding boxes, filter noise
    boxes = []
    for i in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[i]
        # Heuristics to filter punctuation dots from 'i', 'j' etc.
        # and outright noise
        if area < 20:
            continue
        if h < line_h * 0.1:
            continue
        boxes.append((x, y, w, h))

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])

    # Merge overlapping/adjacent boxes (handles letters like 'i' with a dot)
    merged = _merge_close_boxes(boxes, gap_threshold=line_inv.shape[1] * 0.01)

    pad = 3
    for (x, y, w, h) in merged:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(line_inv.shape[1], x + w + pad)
        y2 = min(line_inv.shape[0], y + h + pad)
        crop = line_inv[y1:y2, x1:x2]
        # Normalize to a square with padding
        crop = _pad_to_square(crop)
        char_crops.append(crop)

    return char_crops


def _merge_close_boxes(
    boxes: list[tuple[int, int, int, int]], gap_threshold: float
) -> list[tuple[int, int, int, int]]:
    """Merge horizontally close bounding boxes (e.g. 'i' dot + stem)."""
    if not boxes:
        return []
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        prev = merged[-1]
        prev_right = prev[0] + prev[2]
        if x - prev_right <= gap_threshold:
            # Merge
            new_x = prev[0]
            new_y = min(prev[1], y)
            new_right = max(prev_right, x + w)
            new_bottom = max(prev[1] + prev[3], y + h)
            merged[-1] = [new_x, new_y, new_right - new_x, new_bottom - new_y]
        else:
            merged.append([x, y, w, h])
    return [tuple(b) for b in merged]


def _pad_to_square(img: np.ndarray, size: int = 64) -> np.ndarray:
    """Resize and pad a character crop to a fixed square."""
    h, w = img.shape
    scale = (size - 8) / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas
