"""
Font Builder

Takes a dict of {character: svg_path_data} and assembles a .ttf font file
using fonttools.

Font metrics used:
  - Units Per Em (UPM): 1000
  - Ascender: 800
  - Descender: -200
  - Line gap: 0
  - Cap height: 700
  - x-height: 500

These are standard proportions; they don't need to be exact for a handwriting
font to look good.
"""

from __future__ import annotations

import time
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2Pen import T2Pen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.svgLib.path import SVGPath
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools import svgLib
import re


# ── Font constants ─────────────────────────────────────────────────────────────
UPM = 1000
ASCENDER = 800
DESCENDER = -200
CAP_HEIGHT = 700
X_HEIGHT = 500
LINE_GAP = 0

# Default advance widths (em units) — approximate for handwriting
DEFAULT_ADVANCE = 600
SPACE_ADVANCE = 250
NARROW_CHARS = set("il1!.,;:'\"|/")
WIDE_CHARS = set("mwMW@")


def build_font(
    glyph_paths: dict[str, str],
    output_path: str,
    family_name: str = "MyHandwriting",
    style_name: str = "Regular",
):
    """
    Build a .ttf font from SVG path data strings.

    Args:
        glyph_paths: {character: svg_path_data_string}
        output_path: Where to write the .ttf file
        family_name: Font family name (shown in font picker)
        style_name: Style name (Regular, Bold, etc.)
    """
    fb = FontBuilder(UPM, isTTF=True)

    # ── Names ──────────────────────────────────────────────────────────────────
    fb.setupNameTable({
        "familyName": family_name,
        "styleName": style_name,
    })

    # ── Glyph order ───────────────────────────────────────────────────────────
    # Always include .notdef and space
    all_glyphs = [".notdef", "space"] + [
        _char_to_glyph_name(c) for c in sorted(glyph_paths.keys())
    ]
    # Deduplicate while preserving order
    seen = set()
    glyph_order = []
    for g in all_glyphs:
        if g not in seen:
            seen.add(g)
            glyph_order.append(g)

    fb.setupGlyphOrder(glyph_order)

    # ── Character map (cmap) ──────────────────────────────────────────────────
    cmap: dict[int, str] = {0x0020: "space"}  # space
    for char, _ in glyph_paths.items():
        cmap[ord(char)] = _char_to_glyph_name(char)

    fb.setupCharacterMap(cmap)

    # ── Metrics ───────────────────────────────────────────────────────────────
    fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER)
    fb.setupHorizontalMetrics(_build_metrics(glyph_paths))
    fb.setupOs2(
        sTypoAscender=ASCENDER,
        sTypoDescender=DESCENDER,
        sTypoLineGap=LINE_GAP,
        usWinAscent=ASCENDER,
        usWinDescent=abs(DESCENDER),
        sxHeight=X_HEIGHT,
        sCapHeight=CAP_HEIGHT,
        fsType=0,
    )
    fb.setupPost()
    fb.setupHead(unitsPerEm=UPM, created=int(time.time()), modified=int(time.time()))

    # ── Glyphs ────────────────────────────────────────────────────────────────
    glyphs = {}

    # .notdef — simple rectangle box
    pen = TTGlyphPen(None)
    pen.beginPath()
    _draw_notdef(pen)
    glyphs[".notdef"] = pen.endPath()

    # space — empty glyph
    pen = TTGlyphPen(None)
    glyphs["space"] = pen.endPath()

    # User's characters
    for char, path_data in glyph_paths.items():
        glyph_name = _char_to_glyph_name(char)
        pen = TTGlyphPen(None)
        try:
            _draw_svg_path(pen, path_data)
            glyphs[glyph_name] = pen.endPath()
        except Exception as e:
            print(f"  Warning: could not draw glyph for '{char}': {e}")
            # Fall back to .notdef-style box
            pen2 = TTGlyphPen(None)
            _draw_notdef(pen2)
            glyphs[glyph_name] = pen2.endPath()

    fb.setupGlyf(glyphs)

    # ── Write file ────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fb.font.save(output_path)
    print(f"Font saved: {output_path}  ({len(glyph_paths)} glyphs)")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _char_to_glyph_name(char: str) -> str:
    """Map a character to a PostScript glyph name."""
    _SPECIAL: dict[str, str] = {
        " ": "space", "!": "exclam", '"': "quotedbl", "#": "numbersign",
        "$": "dollar", "%": "percent", "&": "ampersand", "'": "quotesingle",
        "(": "parenleft", ")": "parenright", "*": "asterisk", "+": "plus",
        ",": "comma", "-": "hyphen", ".": "period", "/": "slash",
        ":": "colon", ";": "semicolon", "<": "less", "=": "equal",
        ">": "greater", "?": "question", "@": "at", "[": "bracketleft",
        "\\": "backslash", "]": "bracketright", "^": "asciicircum",
        "_": "underscore", "`": "grave", "{": "braceleft", "|": "bar",
        "}": "braceright", "~": "asciitilde",
    }
    if char in _SPECIAL:
        return _SPECIAL[char]
    if char.isalpha():
        if char.isupper():
            return char  # A-Z as-is
        return char  # a-z as-is
    if char.isdigit():
        _DIGIT_NAMES = {
            "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        }
        return _DIGIT_NAMES.get(char, char)
    return f"uni{ord(char):04X}"


def _build_metrics(glyph_paths: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Build {glyph_name: (advance_width, lsb)} for all glyphs."""
    metrics: dict[str, tuple[int, int]] = {
        ".notdef": (DEFAULT_ADVANCE, 50),
        "space": (SPACE_ADVANCE, 0),
    }
    for char in glyph_paths:
        name = _char_to_glyph_name(char)
        if char in NARROW_CHARS:
            advance = 300
        elif char in WIDE_CHARS:
            advance = 750
        else:
            advance = DEFAULT_ADVANCE
        metrics[name] = (advance, 50)
    return metrics


def _draw_notdef(pen: TTGlyphPen):
    """Draw a simple rectangular box for the .notdef glyph."""
    pen.moveTo((50, 0))
    pen.lineTo((550, 0))
    pen.lineTo((550, 700))
    pen.lineTo((50, 700))
    pen.closePath()
    pen.moveTo((100, 50))
    pen.lineTo((100, 650))
    pen.lineTo((500, 650))
    pen.lineTo((500, 50))
    pen.closePath()


def _draw_svg_path(pen: TTGlyphPen, path_data: str):
    """
    Parse and replay a simplified SVG path onto a TTGlyphPen.
    Handles: M, L, C, Z commands (output of potrace).
    """
    commands = re.findall(r"([MLCQZz])([^MLCQZz]*)", path_data)
    for cmd, args_str in commands:
        nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", args_str)]
        if cmd == "M" and len(nums) >= 2:
            pen.moveTo((nums[0], nums[1]))
        elif cmd == "L" and len(nums) >= 2:
            pen.lineTo((nums[0], nums[1]))
        elif cmd == "C" and len(nums) >= 6:
            # Cubic bezier — iterate over groups of 6
            for i in range(0, len(nums) - 5, 6):
                pen.qCurveConvert(
                    (nums[i], nums[i+1]),
                    (nums[i+2], nums[i+3]),
                    (nums[i+4], nums[i+5]),
                )
        elif cmd in ("Z", "z"):
            pen.closePath()
