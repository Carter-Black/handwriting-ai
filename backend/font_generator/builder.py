# builder.py
# assembles a .ttf font from normalized SVG path data using fonttools
# coordinates arriving here are already in em space (from vectorizer.py)

from __future__ import annotations

import re
import time
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen

# em metrics — keep in sync with vectorizer.py
UPM = 1000
ASCENDER = 800
DESCENDER = -200
CAP_HEIGHT = 700
X_HEIGHT = 500

DEFAULT_ADVANCE = 600
SPACE_ADVANCE = 250
NARROW = set("il1!.,;:'\"|/")
WIDE = set("mwMW@")


def build_font(
    glyph_paths: dict[str, str],
    output_path: str,
    family_name: str = "MyHandwriting",
    style_name: str = "Regular",
):
    """Build and save a .ttf from {char: normalized_svg_path_data}."""
    fb = FontBuilder(UPM, isTTF=True)

    ps_name = "".join(c for c in family_name if c.isalnum() or c == "-")[:63] or "MyHandwriting"
    fb.setupNameTable({
        "familyName": family_name,
        "styleName": style_name,
        "fullName": f"{family_name} {style_name}",
        "version": "Version 1.0",
        "psName": ps_name,
    })

    glyph_order = _dedup([".notdef", "space"] + [_name(c) for c in sorted(glyph_paths)])
    fb.setupGlyphOrder(glyph_order)

    cmap = {0x0020: "space"}
    for ch in glyph_paths:
        cmap[ord(ch)] = _name(ch)
    fb.setupCharacterMap(cmap)

    fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER)
    fb.setupHorizontalMetrics(_metrics(glyph_paths))
    fb.setupOS2(
        sTypoAscender=ASCENDER, sTypoDescender=DESCENDER, sTypoLineGap=0,
        usWinAscent=ASCENDER, usWinDescent=abs(DESCENDER),
        sxHeight=X_HEIGHT, sCapHeight=CAP_HEIGHT, fsType=0,
        usWeightClass=400, usWidthClass=5, fsSelection=0x40,
    )
    fb.setupPost()
    fb.setupHead(unitsPerEm=UPM, created=int(time.time()), modified=int(time.time()))

    # build glyph outlines
    glyphs = {
        ".notdef": _notdef(),
        "space": TTGlyphPen(None).glyph(),
    }
    for ch, path_data in glyph_paths.items():
        gname = _name(ch)
        try:
            glyphs[gname] = _make_glyph(path_data)
        except Exception as e:
            print(f"  '{ch}' failed, using placeholder: {e}")
            glyphs[gname] = _notdef()

    fb.setupGlyf(glyphs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fb.font.save(output_path)
    print(f"saved {output_path} ({len(glyph_paths)} glyphs)")


# glyph construction

def _make_glyph(path_data: str):
    """
    Draw path data through Cu2QuPen → TTGlyphPen.
    Cu2QuPen converts the cubic beziers from potrace into the quadratic
    curves that TrueType (.ttf) requires.
    """
    tt = TTGlyphPen(None)
    _replay(Cu2QuPen(tt, max_err=1.0, reverse_direction=True), path_data)
    return tt.glyph()


def _notdef():
    """Hollow rectangle placeholder for unmapped characters."""
    pen = TTGlyphPen(None)
    # outer box — clockwise in y-up font space (TTF filled contour)
    pen.moveTo((50, 0))
    pen.lineTo((50, 700))
    pen.lineTo((550, 700))
    pen.lineTo((550, 0))
    pen.closePath()
    # inner box — counter-clockwise in y-up (punches a hole)
    pen.moveTo((100, 50))
    pen.lineTo((500, 50))
    pen.lineTo((500, 650))
    pen.lineTo((100, 650))
    pen.closePath()
    return pen.glyph()


def _replay(pen, path_data: str):
    """
    Parse and replay SVG path data onto any fonttools pen.
    Handles absolute M, L, C, Z — the commands potrace emits.
    After an M, implicit repeated coords are treated as L (per SVG spec).
    """
    tokens = re.findall(r'[MLCZz]|[-+]?(?:\d+\.?\d*|\.\d+)', path_data)
    i = 0
    cmd = None

    while i < len(tokens):
        tok = tokens[i]

        if tok in ('M', 'L', 'C', 'Z', 'z'):
            cmd = tok
            if cmd in ('Z', 'z'):
                pen.closePath()
            i += 1
            continue

        if cmd == 'M':
            pen.moveTo((_n(tokens, i), _n(tokens, i + 1)))
            i += 2
            cmd = 'L'  # subsequent pairs in an M block are implicit lineTo
        elif cmd == 'L':
            pen.lineTo((_n(tokens, i), _n(tokens, i + 1)))
            i += 2
        elif cmd == 'C':
            # cubic bezier: off1, off2, on-curve
            pen.curveTo(
                (_n(tokens, i),     _n(tokens, i + 1)),
                (_n(tokens, i + 2), _n(tokens, i + 3)),
                (_n(tokens, i + 4), _n(tokens, i + 5)),
            )
            i += 6
        else:
            i += 1


def _n(tokens: list[str], i: int) -> float:
    return float(tokens[i])


# name and metrics helpers

def _name(char: str) -> str:
    """Map a character to its PostScript glyph name."""
    specials = {
        " ": "space", "!": "exclam", '"': "quotedbl", "#": "numbersign",
        "$": "dollar", "%": "percent", "&": "ampersand", "'": "quotesingle",
        "(": "parenleft", ")": "parenright", "*": "asterisk", "+": "plus",
        ",": "comma", "-": "hyphen", ".": "period", "/": "slash",
        ":": "colon", ";": "semicolon", "?": "question", "@": "at",
    }
    digits = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }
    if char in specials:
        return specials[char]
    if char.isdigit():
        return digits[char]
    if char.isalpha():
        return char
    return f"uni{ord(char):04X}"


def _metrics(glyph_paths: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Return {glyph_name: (advance_width, lsb)} for all glyphs."""
    m = {".notdef": (DEFAULT_ADVANCE, 50), "space": (SPACE_ADVANCE, 0)}
    for ch in glyph_paths:
        if ch in NARROW:
            adv = 300
        elif ch in WIDE:
            adv = 750
        else:
            adv = DEFAULT_ADVANCE
        m[_name(ch)] = (adv, 50)
    return m


def _dedup(lst: list[str]) -> list[str]:
    seen, out = set(), []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
