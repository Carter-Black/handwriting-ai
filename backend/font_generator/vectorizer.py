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
    results: dict[str, str] = {}
    for char, img in char_images.items():
        try:
            h, w = img.shape[:2]
            raw = _run_potrace(img)
            if raw:
                results[char] = _normalize_to_em(raw, w, h)
        except Exception as e:
            print(f"  skipping '{char}': {e}")
    return results


# potrace

def _run_potrace(img: np.ndarray) -> str | None:
    """Trace a binary glyph image with potrace, return raw SVG path data."""
    cleaned = _clean_glyph(img)

    with tempfile.TemporaryDirectory() as tmp:
        bmp = os.path.join(tmp, "g.bmp")
        svg = os.path.join(tmp, "g.svg")

        # potrace needs black ink on white — our images are inverted
        cv2.imwrite(bmp, cv2.bitwise_not(cleaned))

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
    Map SVG pixel coordinates → font em coordinates.

    potrace path data is in image pixel space (y-down, origin top-left).
    Font coordinates are y-up, origin at baseline. We scale, center, and
    flip here so builder.py gets ready-to-use font coordinates.
    """
    if src_w == 0 or src_h == 0:
        return path_data

    scale = min(EM_SIZE / src_w, ASCENDER / src_h) * 0.85  # 15% breathing room
    x_offset = (EM_SIZE - src_w * scale) / 2               # center horizontally

    def xform(x: float, y: float) -> tuple[float, float]:
        return (x * scale + x_offset, ASCENDER - y * scale)

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
