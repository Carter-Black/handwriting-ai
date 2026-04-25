# vectorizer.py
# converts binary glyph images → normalized SVG path data ready for font assembly
# requires potrace to be installed (handled by install scripts)

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# must match the constants in builder.py
EM_SIZE = 1000
ASCENDER = 800


def vectorize_glyphs(char_images: dict[str, np.ndarray]) -> dict[str, str]:
    """Convert {char: binary_image} → {char: em-normalized svg path string}."""
    print(f"[vectorize] tracing {len(char_images)} glyphs through potrace...")
    results: dict[str, str] = {}
    failures: list[tuple[str, str]] = []
    for char, img in char_images.items():
        try:
            h, w = img.shape[:2]
            raw = _run_potrace(img)
            if raw:
                results[char] = _normalize_to_em(raw, w, h)
            else:
                failures.append((char, "potrace returned no path data"))
        except Exception as e:
            failures.append((char, str(e)))

    print(f"[vectorize] vectorized {len(results)}/{len(char_images)} glyphs")
    for ch, reason in failures:
        print(f"[vectorize]   '{ch}' skipped: {reason}")
    return results


# potrace

def _run_potrace(img: np.ndarray) -> str | None:
    """Trace a binary glyph image with potrace, return raw SVG path data."""
    cleaned = _clean_glyph(img)

    with tempfile.TemporaryDirectory() as tmp:
        bmp = os.path.join(tmp, "g.bmp")
        svg = os.path.join(tmp, "g.svg")

        # potrace needs black ink on white — our images are inverted; threshold
        # to pure binary first so grayscale fringe from resize doesn't confuse it
        _, binary = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(bmp, cv2.bitwise_not(binary))

        r = subprocess.run(
            ["potrace", bmp, "--svg", "--output", svg,
             "--turdsize", "2", "--alphamax", "1.0", "--opttolerance", "0.2"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())

        content = Path(svg).read_text()
        paths = re.findall(r'<path[^>]+\bd="([^"]+)"', content)
        return " ".join(paths) if paths else None


def _clean_glyph(img: np.ndarray) -> np.ndarray:
    """Close small stroke gaps and remove noise specks."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, k)


# coordinate transform

def _normalize_to_em(path_data: str, src_w: int, src_h: int) -> str:
    """
    Map SVG path coordinates → font em coordinates.

    IMPORTANT: we measure the path's actual bounding box rather than trusting
    the source-image dimensions. potrace's default --unit=10 quantization
    means its path coords are 10× the source-pixel size; previous versions of
    this function assumed coords were in [0, src_w] and ended up scaling
    glyphs to ~6× the em box, producing the "shattered" Windows rendering
    pattern. Measuring the actual bbox sidesteps potrace's quantization
    factor entirely.

    src_w/src_h are kept in the signature for callers' compatibility but
    are no longer used to compute the scale.
    """
    # Collect ONLY on-curve points for bbox calculation. potrace's cubic
    # off-curve handles can sit well outside the visible curve and would
    # otherwise stretch the bbox, making the visible letter shrink to a tiny
    # corner of the em (the "slash + starburst" rendering pattern).
    tokens = re.findall(r'[MLCZz]|[-+]?(?:\d+\.?\d*|\.\d+)', path_data)
    xs: list[float] = []
    ys: list[float] = []
    i = 0
    cmd = None
    while i < len(tokens):
        tok = tokens[i]
        if tok in ('M', 'L', 'C', 'Z', 'z'):
            cmd = tok
            i += 1
            continue
        if cmd == 'M':
            xs.append(float(tokens[i])); ys.append(float(tokens[i + 1]))
            i += 2
            cmd = 'L'
        elif cmd == 'L':
            xs.append(float(tokens[i])); ys.append(float(tokens[i + 1]))
            i += 2
        elif cmd == 'C':
            # Only the third (on-curve) point counts for bbox; first two are
            # off-curve handles that can fly far from the actual curve.
            xs.append(float(tokens[i + 4])); ys.append(float(tokens[i + 5]))
            i += 6
        else:
            i += 1

    if not xs or not ys:
        return path_data
    path_w = max(xs) - min(xs)
    path_h = max(ys) - min(ys)
    if path_w <= 0 or path_h <= 0:
        return path_data

    # Fit the actual path bbox into the em with 15% breathing room.
    scale = min(EM_SIZE / path_w, ASCENDER / path_h) * 0.85
    new_w = path_w * scale
    # Center horizontally, drop baseline to y=0 so descenders sit naturally below.
    x_offset = (EM_SIZE - new_w) / 2 - min(xs) * scale
    y_offset = -min(ys) * scale

    def xform(x: float, y: float) -> tuple[float, float]:
        return (x * scale + x_offset, y * scale + y_offset)

    return _walk_path(path_data, xform)


def _walk_path(path_data: str, fn) -> str:
    """
    Walk every coordinate pair in an SVG path and apply fn(x, y).
    Handles absolute M, L, C, Z commands (the only ones potrace emits).
    """
    tokens = re.findall(r'[MLCZz]|[-+]?(?:\d+\.?\d*|\.\d+)', path_data)
    out = []
    i = 0
    cmd = None

    while i < len(tokens):
        tok = tokens[i]

        if tok in ('M', 'L', 'C', 'Z', 'z'):
            cmd = tok
            out.append(tok)
            i += 1
            continue

        if cmd == 'M':
            x, y = fn(float(tokens[i]), float(tokens[i + 1]))
            out.append(f"{x:.2f} {y:.2f}")
            i += 2
        elif cmd == 'L':
            x, y = fn(float(tokens[i]), float(tokens[i + 1]))
            out.append(f"{x:.2f} {y:.2f}")
            i += 2
        elif cmd == 'C':
            # three coord pairs per cubic segment (two off-curve, one on-curve)
            pts = []
            for _ in range(3):
                x, y = fn(float(tokens[i]), float(tokens[i + 1]))
                pts.append(f"{x:.2f} {y:.2f}")
                i += 2
            out.append(" ".join(pts))
        else:
            out.append(tok)
            i += 1

    return " ".join(out)
