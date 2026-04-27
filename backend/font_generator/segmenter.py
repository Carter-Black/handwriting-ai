# segmenter.py
# forced-aligns the known handwriting prompt into labeled character crops

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from .prompt import FONT_PROMPT_LINES, FONT_TARGET_CHARS, prompt_words_by_line
except ImportError:  # when backend/main.py imports font_generator as a top-level package
    from font_generator.prompt import FONT_PROMPT_LINES, FONT_TARGET_CHARS, prompt_words_by_line


Box = tuple[int, int, int, int]


@dataclass
class GlyphSample:
    char: str
    crop: np.ndarray
    score: float
    source: str


@dataclass
class SegmentResult:
    glyphs: dict[str, np.ndarray]
    diagnostics: dict
    samples: list[GlyphSample] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.diagnostics.get("errors")


class SegmentAlignmentError(ValueError):
    def __init__(self, message: str, result: SegmentResult):
        super().__init__(message)
        self.result = result


def segment_characters(image: np.ndarray) -> SegmentResult:
    """
    Extract character crops from a binarized handwriting page by forced-aligning
    detected ink to the known prompt. If alignment is ambiguous, raise
    SegmentAlignmentError instead of shifting later labels onto the wrong glyphs.
    """
    print(f"[segment] image: {image.shape[1]}x{image.shape[0]} px")
    inv = cv2.bitwise_not(image) if _mostly_white(image) else image.copy()

    expected_lines = prompt_words_by_line()
    lines = _extract_lines(inv, expected_count=len(expected_lines))
    diagnostics = {
        "expected_lines": len(expected_lines),
        "detected_lines": len(lines),
        "lines": [],
        "errors": [],
        "warnings": [],
        "selected_chars": [],
        "missing_chars": [],
    }

    all_samples: list[GlyphSample] = []
    if len(lines) != len(expected_lines):
        diagnostics["errors"].append(
            f"Expected {len(expected_lines)} prompt lines, detected {len(lines)}. "
            "Write the prompt on exactly the shown line breaks."
        )
        result = SegmentResult({}, diagnostics, all_samples)
        raise SegmentAlignmentError(diagnostics["errors"][0], result)

    for line_index, (line_img, expected_words) in enumerate(zip(lines, expected_lines), start=1):
        line_diag = _align_line(line_img, expected_words, line_index)
        diagnostics["lines"].append(line_diag)
        diagnostics["errors"].extend(line_diag["errors"])
        diagnostics["warnings"].extend(line_diag["warnings"])
        all_samples.extend(line_diag.pop("_samples", []))

    selected: dict[str, GlyphSample] = {}
    for sample in all_samples:
        if sample.char not in FONT_TARGET_CHARS:
            continue
        previous = selected.get(sample.char)
        if previous is None or sample.score > previous.score:
            selected[sample.char] = sample

    missing = [ch for ch in FONT_TARGET_CHARS if ch not in selected]
    diagnostics["missing_chars"] = missing
    if missing:
        shown = " ".join(_display_char(ch) for ch in missing)
        diagnostics["errors"].append(f"Missing required font characters after alignment: {shown}")

    diagnostics["selected_chars"] = sorted(selected)
    result = SegmentResult({ch: sample.crop for ch, sample in selected.items()}, diagnostics, all_samples)

    print(f"[segment] aligned {len(all_samples)} samples; selected {len(result.glyphs)} glyphs")
    if diagnostics["errors"]:
        for err in diagnostics["errors"]:
            print(f"[segment] ERROR: {err}")
        raise SegmentAlignmentError(diagnostics["errors"][0], result)
    return result


def save_diagnostic_preview(result: SegmentResult, output_path: str | Path) -> str:
    """
    Save a labeled contact sheet of the chosen glyph crop for every target char.
    The sheet is useful even when generation is blocked because it makes wrong
    labels visible instead of hiding them inside the .ttf.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cell_w, cell_h = 92, 116
    cols = 10
    chars = list(FONT_TARGET_CHARS)
    rows = int(np.ceil(len(chars) / cols))
    canvas = np.full((rows * cell_h + 36, cols * cell_w, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Font glyph alignment preview", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

    chosen: dict[str, GlyphSample] = {}
    for sample in result.samples:
        previous = chosen.get(sample.char)
        if previous is None or sample.score > previous.score:
            chosen[sample.char] = sample
    for idx, ch in enumerate(chars):
        row, col = divmod(idx, cols)
        x, y = col * cell_w, 36 + row * cell_h
        cv2.rectangle(canvas, (x + 3, y + 3), (x + cell_w - 4, y + cell_h - 4), (210, 210, 210), 1)
        cv2.putText(canvas, _display_char(ch), (x + 7, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        sample = chosen.get(ch)
        if sample is None:
            cv2.putText(canvas, "missing", (x + 12, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
            continue
        glyph = cv2.bitwise_not(sample.crop)
        glyph = cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR)
        glyph = cv2.resize(glyph, (64, 64), interpolation=cv2.INTER_AREA)
        canvas[y + 30:y + 94, x + 14:x + 78] = glyph
        cv2.putText(canvas, f"{sample.score:.2f}", (x + 7, y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (90, 90, 90), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
    return str(output_path)


# line and word alignment

def _align_line(line: np.ndarray, expected_words: list[str], line_index: int) -> dict:
    boxes = _component_boxes(line)
    words = _split_into_word_groups(boxes, expected_words)
    diag = {
        "line": line_index,
        "expected_words": expected_words,
        "detected_components": len(boxes),
        "words": [],
        "errors": [],
        "warnings": [],
        "_samples": [],
    }
    if words is None:
        diag["errors"].append(
            f"Could not confidently split line {line_index}: expected {len(expected_words)} words, "
            f"found {len(boxes)} character-like components."
        )
        return diag

    for word_index, (word, word_boxes) in enumerate(zip(expected_words, words), start=1):
        word_diag, samples = _align_word(line, word, word_boxes, line_index, word_index)
        diag["words"].append(word_diag)
        diag["errors"].extend(word_diag["errors"])
        diag["warnings"].extend(word_diag["warnings"])
        diag["_samples"].extend(samples)
    return diag


def _align_word(line: np.ndarray, word: str, boxes: list[Box], line_index: int, word_index: int) -> tuple[dict, list[GlyphSample]]:
    expected = list(word)
    fitted, actions = _fit_boxes_to_expected(line, boxes, len(expected))
    diag = {
        "word_index": word_index,
        "expected": word,
        "expected_chars": len(expected),
        "detected_components": len(boxes),
        "final_components": len(fitted) if fitted is not None else 0,
        "actions": actions,
        "errors": [],
        "warnings": [],
    }
    if fitted is None:
        diag["errors"].append(
            f"Could not confidently align line {line_index}, word '{word}': "
            f"expected {len(expected)} letters, found {len(boxes)} after merging/splitting."
        )
        return diag, []

    samples: list[GlyphSample] = []
    for char_index, (ch, box) in enumerate(zip(expected, fitted), start=1):
        crop = _crop_box(line, box)
        sample = GlyphSample(
            char=ch,
            crop=crop,
            score=_quality_score(crop, box),
            source=f"line {line_index}, word {word_index}, char {char_index}",
        )
        samples.append(sample)
    return diag, samples


def _split_into_word_groups(boxes: list[Box], expected_words: int | list[str]) -> list[list[Box]] | None:
    expected_tokens = expected_words if isinstance(expected_words, list) else None
    expected_words = len(expected_tokens) if expected_tokens is not None else expected_words
    if expected_words <= 0:
        return []
    if not boxes or len(boxes) < expected_words:
        return None
    if expected_words == 1:
        return [boxes]

    if expected_tokens is not None and all(len(token) == 1 for token in expected_tokens):
        standalone = _split_standalone_char_groups(boxes, expected_tokens)
        if standalone is not None:
            return standalone

    gaps = []
    for i in range(len(boxes) - 1):
        left = boxes[i]
        right = boxes[i + 1]
        gap = right[0] - (left[0] + left[2])
        gaps.append((max(0, gap), i))
    if len(gaps) < expected_words - 1:
        return None

    split_after = sorted(i for _, i in sorted(gaps, reverse=True)[:expected_words - 1])
    groups: list[list[Box]] = []
    start = 0
    for idx in split_after:
        groups.append(boxes[start:idx + 1])
        start = idx + 1
    groups.append(boxes[start:])
    return groups if len(groups) == expected_words and all(groups) else None


def _split_standalone_char_groups(boxes: list[Box], expected_tokens: list[str]) -> list[list[Box]] | None:
    """
    Split rows like "a b c ..." or "0 1 ... ! ?" without letting a multi-part
    glyph shift every following token. Some expected tokens can consume two
    nearby components before the row resumes one-token-per-component.
    """
    needed = len(expected_tokens)
    if len(boxes) < needed:
        return None

    extra = len(boxes) - needed
    groups: list[list[Box]] = []
    pos = 0
    gaps = [max(0, boxes[i + 1][0] - (boxes[i][0] + boxes[i][2])) for i in range(len(boxes) - 1)]
    positive_gaps = [g for g in gaps if g > 0]
    median_gap = float(np.median(positive_gaps)) if positive_gaps else 0.0

    for idx, token in enumerate(expected_tokens):
        remaining_tokens = needed - idx
        remaining_boxes = len(boxes) - pos
        if remaining_boxes < remaining_tokens:
            return None

        count = 1
        if extra > 0 and _can_consume_extra_component(token, boxes, pos, median_gap):
            count = 2
            extra -= 1

        # If there are still surplus boxes near the end, absorb them into
        # token types that commonly draw as separated pieces before falling
        # back to the next token and shifting labels.
        if extra > 0 and token in {'"', ":", ";", "!", "?", "i", "j"} and pos + count < len(boxes):
            gap = max(0, boxes[pos + count][0] - (boxes[pos + count - 1][0] + boxes[pos + count - 1][2]))
            if median_gap == 0 or gap <= median_gap * 0.65:
                count += 1
                extra -= 1

        groups.append(boxes[pos:pos + count])
        pos += count

    if pos != len(boxes):
        return None
    return groups if len(groups) == needed and all(groups) else None


def _can_consume_extra_component(token: str, boxes: list[Box], pos: int, median_gap: float) -> bool:
    if pos + 1 >= len(boxes):
        return False
    a, b = boxes[pos], boxes[pos + 1]
    gap = max(0, b[0] - (a[0] + a[2]))
    close_gap = median_gap == 0 or gap <= median_gap * 0.75
    if token == '"':
        return close_gap or _both_quote_like(a, b)
    if token in {":", ";", "!", "?", "i", "j"}:
        return _vertically_related(a, b) or close_gap and (_is_small_mark(a) or _is_small_mark(b))
    return False


def _both_quote_like(a: Box, b: Box) -> bool:
    return _is_small_mark(a) and _is_small_mark(b) and abs(_box_center_y(a) - _box_center_y(b)) <= max(a[3], b[3]) * 1.5


def _vertically_related(a: Box, b: Box) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    close_centers = abs(_box_center_x(a) - _box_center_x(b)) <= max(aw, bw) * 0.8
    return x_overlap > 0 or close_centers


def _fit_boxes_to_expected(line: np.ndarray, boxes: list[Box], expected_count: int) -> tuple[list[Box] | None, list[str]]:
    current = [tuple(b) for b in boxes]
    actions: list[str] = []
    if expected_count == 0:
        return [], actions

    while len(current) > expected_count:
        idx = _best_merge_pair(current)
        current[idx:idx + 2] = [_union_boxes(current[idx], current[idx + 1])]
        actions.append(f"merged components {idx + 1}-{idx + 2}")

    while len(current) < expected_count:
        remaining = expected_count - len(current)
        idx = _best_split_box(line, current, remaining)
        if idx is None:
            return None, actions
        left, right = _split_box_at_valley(line, current[idx])
        if left is None or right is None:
            return None, actions
        current[idx:idx + 1] = [left, right]
        actions.append(f"split wide component {idx + 1}")

    current.sort(key=lambda b: b[0])
    return current, actions


def _best_merge_pair(boxes: list[Box]) -> int:
    scores = []
    for i in range(len(boxes) - 1):
        a, b = boxes[i], boxes[i + 1]
        gap = max(0, b[0] - (a[0] + a[2]))
        overlap = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        scores.append((gap - overlap * 2, i))
    return min(scores)[1]


def _best_split_box(line: np.ndarray, boxes: list[Box], remaining_splits: int = 1) -> int | None:
    widths = [b[2] for b in boxes]
    median_w = float(np.median(widths)) if widths else 0.0
    candidates = sorted(enumerate(boxes), key=lambda item: item[1][2], reverse=True)
    for idx, box in candidates:
        # When a word is under-segmented, the widest blob is usually two
        # touching letters. Be permissive here; bad splits are still visible in
        # the preview, while refusing to split shifts every later label.
        threshold = max(10, median_w * (1.2 if remaining_splits == 1 else 1.05))
        if box[2] >= threshold:
            left, right = _split_box_at_valley(line, box)
            if left is not None and right is not None:
                return idx
    return None


def _split_box_at_valley(line: np.ndarray, box: Box) -> tuple[Box | None, Box | None]:
    x, y, w, h = box
    if w < 12:
        return None, None
    roi = line[max(0, y):min(line.shape[0], y + h), max(0, x):min(line.shape[1], x + w)]
    if roi.size == 0:
        return None, None
    proj = np.sum(roi > 0, axis=0).astype(np.float32)
    if len(proj) >= 5:
        proj = np.convolve(proj, np.ones(5, dtype=np.float32) / 5, mode="same")
    lo, hi = max(2, int(w * 0.18)), min(w - 2, int(w * 0.82))
    if hi <= lo:
        return None, None
    rel = int(np.argmin(proj[lo:hi]) + lo)
    # Touching letters rarely have a fully empty column, especially with pencil
    # or rounded handwriting. Accept shallow valleys for wide blobs so "fox",
    # "over", and similar words can recover from fused pairs.
    if proj[rel] > max(1, proj.max() * 0.78):
        return None, None
    return (x, y, rel, h), (x + rel, y, w - rel, h)


# image primitives

def _mostly_white(img: np.ndarray) -> bool:
    return np.mean(img) > 127


def _extract_lines(inv: np.ndarray, expected_count: int | None = None) -> list[np.ndarray]:
    """Split inverted image into horizontal line strips."""
    proj = np.sum(inv > 0, axis=1)
    if proj.max() == 0:
        return []
    gap_thresh = max(1, proj.max() * 0.03)

    runs: list[tuple[int, int]] = []
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
    projection_lines = [inv[max(0, t - pad):min(inv.shape[0], b + pad), :] for t, b in bounds]
    if expected_count is None or len(projection_lines) == expected_count:
        return projection_lines

    clustered = _extract_lines_by_component_clusters(inv, expected_count)
    return clustered or projection_lines


def _extract_lines_by_component_clusters(inv: np.ndarray, expected_count: int) -> list[np.ndarray]:
    """
    Fallback for very wide/short prompt photos where horizontal projection can
    merge adjacent visual rows. Cluster connected-component y-centers into the
    known number of prompt rows, then crop full-width strips from those groups.
    """
    boxes = _raw_component_boxes(inv)
    if len(boxes) < expected_count:
        return []

    centers = sorted((y + h / 2, (x, y, w, h)) for x, y, w, h in boxes)
    gaps = [
        (centers[i + 1][0] - centers[i][0], i)
        for i in range(len(centers) - 1)
    ]
    if len(gaps) < expected_count - 1:
        return []

    split_after = sorted(i for _, i in sorted(gaps, reverse=True)[:expected_count - 1])
    groups = []
    start = 0
    for idx in split_after:
        groups.append([box for _, box in centers[start:idx + 1]])
        start = idx + 1
    groups.append([box for _, box in centers[start:]])
    if len(groups) != expected_count or any(not g for g in groups):
        return []

    pad = 8
    strips = []
    for group in groups:
        top = min(y for _, y, _, _ in group)
        bottom = max(y + h for _, y, _, h in group)
        strips.append(inv[max(0, top - pad):min(inv.shape[0], bottom + pad), :])
    print(f"[segment] projection found {len(_projection_line_bounds(inv))} line(s); component clustering recovered {len(strips)}")
    return strips


def _projection_line_bounds(inv: np.ndarray) -> list[tuple[int, int]]:
    proj = np.sum(inv > 0, axis=1)
    if proj.max() == 0:
        return []
    gap_thresh = max(1, proj.max() * 0.03)
    runs: list[tuple[int, int]] = []
    in_line, start = False, 0
    for i, v in enumerate(proj):
        if not in_line and v > gap_thresh:
            in_line, start = True, i
        elif in_line and v <= gap_thresh:
            in_line = False
            runs.append((start, i))
    if in_line:
        runs.append((start, inv.shape[0]))
    return runs


def _raw_component_boxes(img: np.ndarray) -> list[Box]:
    num, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    h_img, w_img = img.shape[:2]
    boxes: list[Box] = []
    for i in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < 12 or h < max(3, h_img * 0.015) or w > w_img * 0.4:
            continue
        boxes.append((x, y, w, h))
    return boxes


def _component_boxes(line: np.ndarray) -> list[Box]:
    num, _, stats, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    h = line.shape[0]
    boxes: list[Box] = []
    for i in range(1, num):
        x, y, w, bh, area = [int(v) for v in stats[i]]
        if area < 5 or bh < max(2, h * 0.025):
            continue
        boxes.append((x, y, w, bh))
    boxes.sort(key=lambda b: b[0])
    return _merge_vertical_marks(boxes)


def _merge_vertical_marks(boxes: list[Box]) -> list[Box]:
    if not boxes:
        return []
    stems: list[Box] = []
    marks: list[Box] = []
    median_h = float(np.median([b[3] for b in boxes])) if boxes else 0.0
    for box in boxes:
        (marks if _is_small_mark(box, median_h) else stems).append(box)

    for mark in marks:
        best_i, best_score = None, float("inf")
        for i, stem in enumerate(stems):
            if not _mark_belongs_to_stem(mark, stem):
                continue
            score = abs(_box_center_x(mark) - _box_center_x(stem)) + max(0, stem[1] - (mark[1] + mark[3]))
            if score < best_score:
                best_i, best_score = i, score
        if best_i is None:
            stems.append(mark)
        else:
            stems[best_i] = _union_boxes(stems[best_i], mark)

    stems.sort(key=lambda b: b[0])
    return stems


def _is_small_mark(box: Box, median_h: float | None = None) -> bool:
    _, _, w, h = box
    if median_h is None or median_h <= 0:
        return h <= 10 or w <= 6
    return h <= median_h * 0.45 or (w <= median_h * 0.35 and h <= median_h * 0.75)


def _mark_belongs_to_stem(mark: Box, stem: Box) -> bool:
    mx, my, mw, mh = mark
    sx, sy, sw, sh = stem
    center_close = abs(_box_center_x(mark) - _box_center_x(stem)) <= max(6, max(mw, sw) * 0.7)
    x_overlap = max(0, min(mx + mw, sx + sw) - max(mx, sx))
    vertically_close = my + mh <= sy + sh + max(8, sh * 0.45)
    return (center_close or x_overlap > 0) and vertically_close


def _box_center_x(box: Box) -> float:
    return box[0] + box[2] / 2


def _box_center_y(box: Box) -> float:
    return box[1] + box[3] / 2


def _crop_box(line: np.ndarray, box: Box, pad: int = 3) -> np.ndarray:
    x, y, w, h = box
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(line.shape[1], x + w + pad)
    y2 = min(line.shape[0], y + h + pad)
    return _to_square(line[y1:y2, x1:x2])


def _to_square(img: np.ndarray, size: int = 128) -> np.ndarray:
    h, w = img.shape
    if h <= 0 or w <= 0:
        return np.zeros((size, size), dtype=np.uint8)
    scale = (size - 8) / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    canvas = np.zeros((size, size), dtype=np.uint8)
    yo, xo = (size - nh) // 2, (size - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def _quality_score(crop: np.ndarray, box: Box) -> float:
    ink = crop > 0
    ink_ratio = float(np.mean(ink))
    coords = cv2.findNonZero(crop)
    if coords is None:
        return 0.0
    x, y, w, h = cv2.boundingRect(coords)
    aspect = w / max(1, h)
    fill_score = max(0.0, 1.0 - abs(ink_ratio - 0.12) / 0.18)
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.55) / 1.2)
    area_score = min(1.0, (w * h) / (128 * 128 * 0.45))
    center_x = x + w / 2
    center_y = y + h / 2
    center_score = 1.0 - min(1.0, (abs(center_x - 64) + abs(center_y - 64)) / 90)
    return round(0.35 * fill_score + 0.25 * aspect_score + 0.25 * area_score + 0.15 * center_score, 4)


def _union_boxes(a: Box, b: Box) -> Box:
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return (x1, y1, x2 - x1, y2 - y1)


def _display_char(ch: str) -> str:
    if ch == " ":
        return "space"
    if ch == '"':
        return '\\"'
    if ch == "'":
        return "'"
    return ch
