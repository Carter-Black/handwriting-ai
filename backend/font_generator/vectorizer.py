"""
Glyph Vectorizer

Converts binary character images (numpy uint8 arrays) into SVG path data
suitable for embedding into a font.

Pipeline per glyph:
  1. Clean up the binary image (morphological ops).
  2. Write to a temporary BMP file.
  3. Run `potrace` to trace the bitmap → SVG.
  4. Parse the <path d="..."> from the SVG output.
  5. Return the path string for fonttools.

Requires `potrace` to be installed (included in install scripts).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def vectorize_glyphs(
    char_images: dict[str, np.ndarray],
) -> dict[str, str]:
    """
    Convert a dict of {char: binary_image} → {char: svg_path_data_string}.
    Characters that fail vectorization are silently skipped.
    """
    results: dict[str, str] = {}
    for char, img in char_images.items():
        try:
            svg_path = _vectorize_single(img)
            if svg_path:
                results[char] = svg_path
        except Exception as e:
            print(f"  Warning: could not vectorize '{char}': {e}")
    return results


def _vectorize_single(img: np.ndarray) -> str | None:
    """
    Vectorize a single binary glyph image.
    Returns the SVG path data string, or None on failure.
    """
    cleaned = _clean_glyph(img)

    with tempfile.TemporaryDirectory() as tmpdir:
        bmp_path = os.path.join(tmpdir, "glyph.bmp")
        svg_path = os.path.join(tmpdir, "glyph.svg")

        # potrace expects black ink on white background
        # Our images are white ink on black → invert
        inverted = cv2.bitwise_not(cleaned)
        cv2.imwrite(bmp_path, inverted)

        result = subprocess.run(
            [
                "potrace",
                bmp_path,
                "--svg",
                "--output", svg_path,
                "--turdsize", "2",    # remove speckles smaller than 2px
                "--alphamax", "1.0",  # smooth curves
                "--opttolerance", "0.2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"potrace failed: {result.stderr.strip()}")

        svg_content = Path(svg_path).read_text()
        return _extract_path_data(svg_content)


def _clean_glyph(img: np.ndarray) -> np.ndarray:
    """
    Morphological cleanup: close small gaps, remove isolated noise pixels.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Close small gaps in strokes
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remove tiny noise blobs
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def _extract_path_data(svg_content: str) -> str | None:
    """
    Parse all <path d="..."> elements from potrace SVG output and
    return them concatenated. potrace may output multiple path elements
    for compound glyphs (letters with holes, like 'o', 'e', 'a').
    """
    matches = re.findall(r'<path[^>]+\bd="([^"]+)"', svg_content)
    if not matches:
        return None
    # Concatenate multiple paths (compound shapes)
    return " ".join(matches)


def normalize_path_to_em(
    path_data: str,
    src_width: int,
    src_height: int,
    em_size: int = 1000,
    ascender: int = 800,
) -> str:
    """
    Scale and translate SVG path coordinates from pixel-space to font em-space.

    Font coordinate system:
      - Origin is at the baseline
      - Y increases upward (SVG Y increases downward — we flip)
      - Em square is typically 1000 units (UPM)

    This is called by builder.py after vectorization.
    """
    if src_width == 0 or src_height == 0:
        return path_data

    scale = min(em_size / src_width, ascender / src_height) * 0.85  # 15% margin
    x_offset = (em_size - src_width * scale) / 2
    y_offset = ascender  # baseline offset

    # Replace all coordinate pairs in path data
    # SVG path commands: M, L, C, Q, S, T, A — we scale all numeric pairs
    def scale_coords(match: re.Match) -> str:
        x = float(match.group(1))
        y = float(match.group(2))
        new_x = x * scale + x_offset
        new_y = y_offset - y * scale  # flip Y axis
        return f"{new_x:.2f},{new_y:.2f}"

    scaled = re.sub(
        r"(-?\d+\.?\d*),(-?\d+\.?\d*)",
        scale_coords,
        path_data,
    )
    return scaled
