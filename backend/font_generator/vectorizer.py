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

# Set to a directory path to save pre-potrace BMP crops + raw SVG for every
# glyph.  Useful for diagnosing "slash" rendering: open the debug folder and
# confirm the BMPs look like actual letters before potrace sees them.
# Set to None (or leave as-is) for production; set to a path string to enable.
DEBUG_DIR: str | None = None


def vectorize_glyphs(char_images: dict[str, np.ndarray]) -> dict[str, str]:
    """Convert {char: binary_image} → {char: em-normalized svg path string}."""
    print(f"[vectorize] tracing {len(char_images)} glyphs through potrace...")
    results: dict[str, str] = {}
    failures: list[tuple[str, str]] = []

    # resolve debug dir relative to this file's storage sibling
    debug_dir: Path | None = None
    if DEBUG_DIR:
        debug_dir = Path(DEBUG_DIR)
    else:
        candidate = Path(__file__).parent.parent / "storage" / "debug" / "glyphs"
        # Only auto-enable when the marker file exists so we don't litter disk
        if (candidate.parent / "ENABLE_GLYPH_DEBUG").exists():
            debug_dir = candidate
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[vectorize] debug images → {debug_dir}")

    for char, img in char_images.items():
        try:
            h, w = img.shape[:2]
            safe = f"uni{ord(char):04X}" if not char.isalnum() else char
            raw = _run_potrace(img, debug_dir=debug_dir, debug_name=safe)
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

def _run_potrace(
    img: np.ndarray,
    debug_dir: "Path | None" = None,
    debug_name: str = "glyph",
) -> str | None:
    """Trace a binary glyph image with potrace, return raw SVG path data."""
    cleaned = _clean_glyph(img)

    with tempfile.TemporaryDirectory() as tmp:
        bmp = os.path.join(tmp, "g.bmp")
        svg = os.path.join(tmp, "g.svg")

        # potrace needs black ink on white — our images are inverted; threshold
        # to pure binary first so grayscale fringe from resize doesn't confuse it
        _, binary = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        bmp_for_potrace = cv2.bitwise_not(binary)
        cv2.imwrite(bmp, bmp_for_potrace)

        # Debug: save pre-potrace BMP so we can inspect what potrace receives
        if debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"{debug_name}_input.bmp"), bmp_for_potrace)

        r = subprocess.run(
            ["potrace", bmp, "--svg", "--output", svg,
             "--turdsize", "2", "--alphamax", "1.0", "--opttolerance", "0.2"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())

        content = Path(svg).read_text()

        # Debug: save the raw SVG for the first few glyphs
        if debug_dir is not None:
            (debug_dir / f"{debug_name}_potrace.svg").write_text(content)

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

    Two coordinate-system issues matter here:

    1. Scale: potrace's default --unit=10 means path coords are 10× the
       source-pixel size.  We measure the actual path bbox so we don't need
       to know the exact scale factor.  We use ONLY on-curve points for the
       bbox (off-curve cubic handles can sit far outside the visible curve).

    2. Y orientation: potrace's SVG output stores the visible flip in the
       surrounding <g transform="... scale(...,-...)">. The raw path numbers
       we extract are already in a y-up drawing space. Flipping them again
       mirrors every glyph vertically.

    src_w/src_h are kept in the signature for callers' compatibility but
    are no longer used to compute the scale.
    """
    # Collect ONLY on-curve points for bbox calculation.
    xs: list[float] = []
    ys: list[float] = []
    for cmd, pts in _iter_svg_path(path_data):
        if cmd in ("M", "L"):
            x, y = pts[0]
            xs.append(x); ys.append(y)
        elif cmd == "C":
            # Only the third (on-curve) point counts for bbox; first two are
            # off-curve handles that can fly far from the actual curve.
            x, y = pts[2]
            xs.append(x); ys.append(y)

    if not xs or not ys:
        return path_data
    path_w = max(xs) - min(xs)
    path_h = max(ys) - min(ys)
    if path_w <= 0 or path_h <= 0:
        return path_data

    # Fit the actual path bbox into the em with 15% breathing room.
    scale = min(EM_SIZE / path_w, ASCENDER / path_h) * 0.85
    new_w = path_w * scale
    x_min = min(xs)
    y_min = min(ys)
    # Center horizontally.
    x_offset = (EM_SIZE - new_w) / 2 - x_min * scale

    def xform(x: float, y: float) -> tuple[float, float]:
        return (x * scale + x_offset, (y - y_min) * scale)

    return _walk_path(path_data, xform)


def _walk_path(path_data: str, fn) -> str:
    """
    Walk every coordinate pair in an SVG path and apply fn(x, y).
    Handles absolute/relative M, L, H, V, C, Z commands. Potrace commonly
    emits lowercase relative cubic commands; treating their numbers as
    absolute lines is exactly how a valid glyph turns into slash/starburst
    geometry.
    """
    out = []
    for cmd, pts in _iter_svg_path(path_data):
        out.append(cmd)
        if cmd in ("M", "L"):
            x, y = fn(*pts[0])
            out.append(f"{x:.2f} {y:.2f}")
        elif cmd == "C":
            transformed = []
            for pt in pts:
                x, y = fn(*pt)
                transformed.append(f"{x:.2f} {y:.2f}")
            out.append(" ".join(transformed))

    return " ".join(out)


def _iter_svg_path(path_data: str):
    """
    Yield absolute uppercase SVG operations as (cmd, points).

    The previous parser only recognized uppercase M/L/C/Z. Potrace SVG output
    frequently uses relative lowercase commands (especially "c"), and ignoring
    those command tokens makes the numeric cubic deltas look like a long chain
    of absolute line segments.
    """
    tokens = re.findall(
        r'[MmLlHhVvCcZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',
        path_data,
    )
    i = 0
    cmd = None
    cur = (0.0, 0.0)
    start = (0.0, 0.0)

    def is_cmd(value: str) -> bool:
        return bool(re.fullmatch(r'[MmLlHhVvCcZz]', value))

    def number() -> float:
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    while i < len(tokens):
        if is_cmd(tokens[i]):
            cmd = tokens[i]
            i += 1

        if cmd is None:
            i += 1
            continue

        if cmd in ("Z", "z"):
            cur = start
            yield "Z", ()
            cmd = None
            continue

        if cmd in ("M", "m"):
            first = True
            rel = cmd == "m"
            while i < len(tokens) and not is_cmd(tokens[i]):
                x, y = number(), number()
                if rel:
                    x += cur[0]; y += cur[1]
                cur = (x, y)
                if first:
                    start = cur
                    yield "M", (cur,)
                    first = False
                else:
                    yield "L", (cur,)
            cmd = "l" if rel else "L"
            continue

        if cmd in ("L", "l"):
            rel = cmd == "l"
            while i < len(tokens) and not is_cmd(tokens[i]):
                x, y = number(), number()
                if rel:
                    x += cur[0]; y += cur[1]
                cur = (x, y)
                yield "L", (cur,)
            continue

        if cmd in ("H", "h"):
            rel = cmd == "h"
            while i < len(tokens) and not is_cmd(tokens[i]):
                x = number()
                if rel:
                    x += cur[0]
                cur = (x, cur[1])
                yield "L", (cur,)
            continue

        if cmd in ("V", "v"):
            rel = cmd == "v"
            while i < len(tokens) and not is_cmd(tokens[i]):
                y = number()
                if rel:
                    y += cur[1]
                cur = (cur[0], y)
                yield "L", (cur,)
            continue

        if cmd in ("C", "c"):
            rel = cmd == "c"
            while i < len(tokens) and not is_cmd(tokens[i]):
                pts = []
                for _ in range(3):
                    x, y = number(), number()
                    if rel:
                        x += cur[0]; y += cur[1]
                    pts.append((x, y))
                cur = pts[2]
                yield "C", tuple(pts)
            continue
