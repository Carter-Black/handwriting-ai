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
    if proj.max() == 0:
        return []
    gap_thresh = proj.max() * 0.03

    runs = []
    in_line, start = False, 0
    for i, v in enumerate(proj):
        if not in_line and v > gap_thresh:
            in_line, start = True, i
        elif in_line and v <= gap_thresh:
            in_line = False
            runs.append((start, i))
    if in_line:
        runs.append((start, inv.shape[0]))
    if not runs:
        return []

    # Drop tiny fragments (descenders/dots that detached from their main row)
    # and absorb close-by fragments into their nearest real line. Without this,
    # a 'g' descender between rows becomes its own "line" and breaks the
    # positional matching against PROMPT_CHARS.
    max_h = max(b - t for t, b in runs)
    real = [(t, b) for t, b in runs if (b - t) >= max_h * 0.3]
    if not real:
        return []
    fragments = [(t, b) for t, b in runs if (b - t) < max_h * 0.3]
    bounds = [list(r) for r in real]
    median_h = int(np.median([b - t for t, b in real]))
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

    pad = 4
    return [
        inv[max(0, t - pad):min(inv.shape[0], b + pad), :]
        for t, b in bounds
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
    merged = _merge_boxes(boxes)

    pad = 3
    crops = []
    for x, y, w, bh in merged:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(line.shape[1], x + w + pad)
        y2 = min(line.shape[0], y + bh + pad)
        crops.append(_to_square(line[y1:y2, x1:x2]))
    return crops


def _merge_boxes(boxes: list[tuple]) -> list[tuple]:
    """
    Merge components that are vertically stacked (e.g. dot above 'i' stem,
    dot above 'j', dot above '!' or '?'). Does NOT merge horizontally
    adjacent components — those are usually separate characters within a
    word, and merging them collapses entire words into single boxes.

    The previous version merged anything within 1% of line width (~33px on
    a 3300px-wide image), which over-merged adjacent letters and caused the
    segmenter to detect ~1/3 of the actual characters.
    """
    if not boxes:
        return []
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        prev_x, prev_y, prev_w, prev_h = merged[-1]
        prev_right = prev_x + prev_w
        # Vertical-stacking test: x-ranges overlap by at least 50% of the
        # narrower box. The dot of an 'i' shares the stem's x-range.
        x_overlap = max(0, min(x + w, prev_right) - max(x, prev_x))
        smaller_w = min(w, prev_w)
        overlap_ratio = x_overlap / smaller_w if smaller_w > 0 else 0

        if overlap_ratio > 0.5:
            new_x = min(prev_x, x)
            new_y = min(prev_y, y)
            new_r = max(prev_right, x + w)
            new_b = max(prev_y + prev_h, y + h)
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
