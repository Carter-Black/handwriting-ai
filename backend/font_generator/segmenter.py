# segmenter.py
# extracts individual character images from a preprocessed handwriting page
# works best with printed (not cursive) handwriting

from __future__ import annotations

import string

import cv2
import numpy as np

# the paragraph we ask users to write — defines the expected character sequence
PROMPT_CHARS = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "0123456789!?,;:'\"()-/"
)

TARGET_CHARS = (
    string.ascii_lowercase
    + string.ascii_uppercase
    + string.digits
    + r"""!?.,;:'"()-/"""
)


def segment_characters(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Extract character crops from a binarized handwriting image.
    Returns {char: binary_image} using the known prompt sequence as labels.
    When a character appears multiple times, we keep the largest (clearest) crop.
    """
    print(f"[segment] image: {image.shape[1]}x{image.shape[0]} px")
    inv = cv2.bitwise_not(image) if _mostly_white(image) else image.copy()

    lines = _extract_lines(inv)
    print(f"[segment] detected {len(lines)} line(s)")
    collected: dict[str, list[np.ndarray]] = {}
    prompt_chars = [c for c in PROMPT_CHARS]
    prompt_idx = 0
    total_regions = 0

    for line_idx, line in enumerate(lines):
        chars = _chars_from_line(line)
        total_regions += len(chars)
        for crop in chars:
            # skip spaces in the prompt sequence
            while prompt_idx < len(prompt_chars) and prompt_chars[prompt_idx] == " ":
                prompt_idx += 1
            if prompt_idx >= len(prompt_chars):
                break
            label = prompt_chars[prompt_idx]
            prompt_idx += 1
            if label in TARGET_CHARS:
                collected.setdefault(label, []).append(crop)

    print(f"[segment] segmented {total_regions} character regions; assigned positionally against prompt")
    print(f"[segment] consumed prompt index: {prompt_idx}/{len(prompt_chars)}")
    print(f"[segment] kept {len(collected)} unique chars: {' '.join(sorted(collected))}")
    missing = [c for c in TARGET_CHARS if c not in collected]
    if missing:
        print(f"[segment] {len(missing)} target chars not captured: {' '.join(missing)}")

    # keep the largest crop per character — bigger usually means more detail
    return {ch: max(crops, key=lambda c: c.shape[0] * c.shape[1]) for ch, crops in collected.items()}


# line detection

def _mostly_white(img: np.ndarray) -> bool:
    return np.mean(img) > 127


def _extract_lines(inv: np.ndarray) -> list[np.ndarray]:
    """Split inverted image into horizontal line strips."""
    proj = np.sum(inv, axis=1)
    gap_thresh = proj.max() * 0.03

    in_line = False
    starts, ends = [], []
    for i, v in enumerate(proj):
        if not in_line and v > gap_thresh:
            in_line = True
            starts.append(i)
        elif in_line and v <= gap_thresh:
            in_line = False
            ends.append(i)
    if in_line:
        ends.append(inv.shape[0])

    pad = 4
    return [
        inv[max(0, t - pad):min(inv.shape[0], b + pad), :]
        for t, b in zip(starts, ends)
        if b - t > 5
    ]


# character detection within a line

def _chars_from_line(line: np.ndarray) -> list[np.ndarray]:
    """Segment individual character crops from a line strip."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    h = line.shape[0]

    boxes = []
    for i in range(1, num):  # skip background (label 0)
        x, y, w, bh, area = stats[i]
        if area < 20 or bh < h * 0.1:  # filter noise and tiny specks
            continue
        boxes.append((x, y, w, bh))

    boxes.sort(key=lambda b: b[0])  # left to right
    merged = _merge_boxes(boxes, gap=line.shape[1] * 0.01)

    pad = 3
    crops = []
    for x, y, w, bh in merged:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(line.shape[1], x + w + pad)
        y2 = min(line.shape[0], y + bh + pad)
        crops.append(_to_square(line[y1:y2, x1:x2]))
    return crops


def _merge_boxes(boxes: list[tuple], gap: float) -> list[tuple]:
    """Merge horizontally adjacent boxes (e.g. dot above 'i' + stem)."""
    if not boxes:
        return []
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        prev = merged[-1]
        if x - (prev[0] + prev[2]) <= gap:
            new_x = prev[0]
            new_y = min(prev[1], y)
            new_r = max(prev[0] + prev[2], x + w)
            new_b = max(prev[1] + prev[3], y + h)
            merged[-1] = [new_x, new_y, new_r - new_x, new_b - new_y]
        else:
            merged.append([x, y, w, h])
    return [tuple(b) for b in merged]


def _to_square(img: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize and center a character crop onto a fixed square canvas."""
    h, w = img.shape
    scale = (size - 8) / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    canvas = np.zeros((size, size), dtype=np.uint8)
    yo, xo = (size - nh) // 2, (size - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas
