# builder.py
# assembles a .ttf font from normalized SVG path data using fonttools
# coordinates arriving here are already in em space (from vectorizer.py)
#
# Approach: line-flatten the cubic beziers from potrace into polylines, then
# feed through ReverseContourPen → TTGlyphPen.  We do NOT use Cu2QuPen as
# primary because it places off-curve quadratic control points far outside
# near-linear cubics, stretching the glyph bbox into a tiny corner.
#
# Winding:  potrace path data reaches us in SVG drawing coordinates; after
# normalization, ReverseContourPen keeps the final TrueType contour direction
# consistent for Windows rasterizers.
#
# We DO NOT call TTGlyphPen.curveTo() directly with cubics: that triggers
# `flagCubic = 0x80` (the cubic-glyf extension), which Windows GDI and
# DirectWrite can't render and which causes the "shattered glyph" pattern.
# See FontResearch.txt §8 for the full diagnosis.

from __future__ import annotations

import re
import time
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import flagCubic, flagOverlapSimple

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

# cu2qu max conversion error in font units. 1.0 = 1/1000 em is the canonical
# tolerance. If a glyph fails at this tightness we progressively loosen until
# it succeeds, then ultimately fall back to manual line flattening.
CU2QU_TOLERANCES = (1.0, 5.0, 20.0)


def build_font(
    glyph_paths: dict[str, str],
    output_path: str,
    family_name: str = "MyHandwriting",
    style_name: str = "Regular",
):
    """Build and save a .ttf from {char: normalized_svg_path_data}."""
    print(f"[build] assembling TTF: {len(glyph_paths)} source glyphs (+ .notdef + space)")
    fb = FontBuilder(UPM, isTTF=True)

    ps_name = "".join(c for c in family_name if c.isalnum() or c == "-")[:63] or "MyHandwriting"
    # mac=True writes Mac platform records too — harmless on Windows, lets the
    # font work in PDF/printer paths that still expect Mac (1,0) name records.
    fb.setupNameTable({
        "familyName":           family_name,
        "styleName":            style_name,
        "uniqueFontIdentifier": f"{family_name};1.000;{style_name};{ps_name}-{style_name}",
        "fullName":             f"{family_name} {style_name}",
        "version":              "Version 1.000",
        "psName":               f"{ps_name}-{style_name}",
    }, mac=True)

    glyph_order = _dedup([".notdef", "space"] + [_name(c) for c in sorted(glyph_paths)])
    fb.setupGlyphOrder(glyph_order)

    cmap = {0x0020: "space"}
    for ch in glyph_paths:
        cmap[ord(ch)] = _name(ch)
    fb.setupCharacterMap(cmap)

    # Build glyphs first so hmtx can read xMin from the actual glyf data.
    glyphs = {
        ".notdef": _make_notdef(),
        "space":   _make_space(),
    }
    failures: list[tuple[str, str]] = []
    for ch, path_data in glyph_paths.items():
        gname = _name(ch)
        try:
            glyphs[gname] = _make_glyph(path_data)
        except Exception as e:
            failures.append((ch, str(e)))
            glyphs[gname] = _make_notdef()

    if failures:
        print(f"[build] {len(failures)} glyph(s) used placeholder:")
        for ch, reason in failures:
            print(f"[build]   '{ch}': {reason}")

    fb.setupGlyf(glyphs)

    # Compute hmtx after setupGlyf so lsb can match each glyph's actual xMin.
    glyf_table = fb.font["glyf"]
    metrics = _metrics(glyph_paths, glyf_table)
    fb.setupHorizontalMetrics(metrics)

    # Per-glyph bbox diagnostic. If glyphs are wildly oversized or
    # microscopic the path-normalization bbox-fit didn't work and the glyph
    # will render as a slash + starburst.
    print("[build] per-glyph bbox check (first 8 user glyphs):")
    sample = [_name(c) for c in sorted(glyph_paths)[:8]]
    for gname in sample:
        g = glyf_table[gname]
        if g.numberOfContours > 0:
            w = g.xMax - g.xMin
            h = g.yMax - g.yMin
            warn = "  <-- OUT OF RANGE" if (w > 1100 or h > 1100 or w < 30 or h < 30) else ""
            print(f"[build]   {gname:>10}: contours={g.numberOfContours:>2}  pts={len(g.coordinates):>4}  bbox=({g.xMin:>4},{g.yMin:>4})-({g.xMax:>4},{g.yMax:>4})  size={w}x{h}{warn}")
        else:
            print(f"[build]   {gname:>10}: empty (placeholder used)")

    fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER, lineGap=0)

    # OS/2 — Windows checks every one of these. usWinAscent/usWinDescent are
    # the line-clipping rectangle (zero values would clip glyphs to nothing).
    # ulCodePageRange1=1 declares Latin 1 coverage. fsSelection=0x40 (REGULAR)
    # MUST agree with head.macStyle (set to 0 below); Windows uses fsSelection
    # over macStyle but enforces consistency.
    codepoints = sorted(cmap.keys())
    fb.setupOS2(
        sTypoAscender=ASCENDER, sTypoDescender=DESCENDER, sTypoLineGap=0,
        usWinAscent=ASCENDER, usWinDescent=abs(DESCENDER),
        sxHeight=X_HEIGHT, sCapHeight=CAP_HEIGHT, fsType=0,
        usWeightClass=400, usWidthClass=5,
        fsSelection=0x40,
        achVendID="NONE",
        ulUnicodeRange1=1,    # bit 0 = Basic Latin
        ulCodePageRange1=1,   # bit 0 = Latin 1
        usFirstCharIndex=codepoints[0],
        usLastCharIndex=codepoints[-1],
    )

    fb.setupPost(
        isFixedPitch=0,
        italicAngle=0.0,
        underlinePosition=-100,
        underlineThickness=50,
    )
    fb.setupHead(unitsPerEm=UPM, created=int(time.time()), modified=int(time.time()))

    # Final head settings per FontResearch §8 known-good script.
    # glyphDataFormat=0 is mandatory for Windows compat (1 = experimental
    # cubic-glyf extension Windows can't render).
    head = fb.font["head"]
    head.glyphDataFormat = 0
    head.macStyle = 0           # matches OS/2.fsSelection=0x40
    head.flags = 0x0009         # bit 0 baseline at y=0, bit 3 force ppem to int
    head.fontRevision = 1.0
    head.lowestRecPPEM = 8

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fb.font.save(output_path)

    _validate_font(output_path)

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"[build] saved: {output_path} ({size_kb:.1f} KB, {len(glyphs)} total glyphs)")


# glyph construction

def _make_glyph(path_data: str):
    """
    Build a TTF glyph from a cubic-bezier SVG path.

    PRIMARY: line-flatten the cubics into many lineTo's. This is lossy
    (curves become piecewise-linear) but bulletproof: all coordinates are
    on-curve points sampled along the actual curve, so no off-curve
    outliers can stretch the glyph bbox or create the "slash + starburst"
    rendering signature.

    Why we don't use Cu2QuPen as primary anymore: Cu2QuPen converts cubic
    beziers to quadratic, but for near-linear input cubics the optimal
    quadratic approximation often places off-curve control points far
    outside the visible curve. Those off-curve outliers stretched the
    visible glyph into a tiny corner of the em with a long line connecting
    to the outlier. Line flattening sidesteps the problem entirely.

    Wrapping order:  pen ← ReverseContourPen ← _DegenerateFilterPen ← TTGlyphPen
    - ReverseContourPen flips potrace's CCW outer to TTF's required CW.
    - _DegenerateFilterPen drops contours <3 unique integer points so
      rounding artifacts can't produce stray-line glyph errors.

    Fallback: try Cu2QuPen if for any reason flattening fails.
    """
    try:
        tt_pen = TTGlyphPen(None)
        filt = _DegenerateFilterPen(tt_pen)
        rev = ReverseContourPen(filt)
        _replay_flattened(rev, path_data)
        glyph = tt_pen.glyph()
        _mark_overlap(glyph)
        return glyph
    except Exception as flatten_err:
        # Last-resort fallback: try Cu2QuPen at progressively looser tolerances.
        last = flatten_err
        for max_err in CU2QU_TOLERANCES:
            try:
                tt_pen = TTGlyphPen(None)
                filt = _DegenerateFilterPen(tt_pen)
                cu2qu = Cu2QuPen(filt, max_err=max_err, reverse_direction=True)
                _replay_cubics(cu2qu, path_data)
                glyph = tt_pen.glyph()
                _mark_overlap(glyph)
                return glyph
            except Exception as e:
                last = e
        raise RuntimeError(
            f"both line-flatten and Cu2QuPen fallback failed; "
            f"flatten: {flatten_err}; cu2qu: {last}"
        ) from last


def _mark_overlap(glyph) -> None:
    """
    Set the OVERLAP_SIMPLE flag (0x40) on the first point of the first
    contour. Tells the OpenType rasterizer that contours in this glyph
    may overlap and should be unioned via non-zero winding rather than
    XOR'd via even-odd. Without this, overlapping contours from the
    potrace+cu2qu pipeline render as the chaotic black/white triangle
    pattern characteristic of even-odd-on-overlap.
    """
    if glyph.numberOfContours > 0 and len(glyph.flags) > 0:
        glyph.flags[0] |= flagOverlapSimple


class _DegenerateFilterPen:
    """
    Wraps a TTGlyphPen-like target. Buffers each contour and only
    forwards it on closePath if it has at least 3 unique integer
    points after rounding. Drops degenerate 1-2-point contours which
    are silently invalid in TrueType and produce stray-triangle
    rendering on Windows (per FontResearch §8 culprit B.4).
    """

    def __init__(self, target):
        self.target = target
        self._buf: list[tuple[str, tuple]] = []
        self._unique_pts: list[tuple[int, int]] = []

    def _track(self, pt) -> None:
        if pt is None:
            return
        ipt = (round(pt[0]), round(pt[1]))
        if not self._unique_pts or self._unique_pts[-1] != ipt:
            self._unique_pts.append(ipt)

    def moveTo(self, pt) -> None:
        self._buf = [("moveTo", (pt,))]
        self._unique_pts = []
        self._track(pt)

    def lineTo(self, pt) -> None:
        self._buf.append(("lineTo", (pt,)))
        self._track(pt)

    def curveTo(self, *points) -> None:
        self._buf.append(("curveTo", points))
        for p in points:
            self._track(p)

    def qCurveTo(self, *points) -> None:
        self._buf.append(("qCurveTo", points))
        for p in points:
            self._track(p)

    def _flush_if_valid(self, closer: str) -> None:
        # Drop the trailing duplicate of moveTo if the contour wraps back to start
        pts = self._unique_pts
        if len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) >= 3:
            for op, args in self._buf:
                getattr(self.target, op)(*args)
            getattr(self.target, closer)()
        self._buf = []
        self._unique_pts = []

    def closePath(self) -> None:
        self._flush_if_valid("closePath")

    def endPath(self) -> None:
        self._flush_if_valid("endPath")

    def addComponent(self, glyphName, transformation) -> None:
        # Pass through composite references unchanged
        self.target.addComponent(glyphName, transformation)


def _make_notdef():
    """Hollow rectangle, outer CW (TTF convention) inner CCW."""
    pen = TTGlyphPen(None)
    # outer (CW in y-up): bottom-left → top-left → top-right → bottom-right
    pen.moveTo((50, 0))
    pen.lineTo((50, 700))
    pen.lineTo((550, 700))
    pen.lineTo((550, 0))
    pen.closePath()
    # inner counter (CCW): cuts a hole through the fill
    pen.moveTo((100, 50))
    pen.lineTo((500, 50))
    pen.lineTo((500, 650))
    pen.lineTo((100, 650))
    pen.closePath()
    glyph = pen.glyph()
    _mark_overlap(glyph)
    return glyph


def _make_space():
    """Empty contour set; advance comes from hmtx."""
    return TTGlyphPen(None).glyph()


def _replay_cubics(pen, path_data: str):
    """
    Replay SVG path data, passing cubic beziers through verbatim.

    Defensively auto-closes any open contour before a new moveTo, and at
    end of path. This handles potrace output where Z is missing between
    subpaths — without this the second moveTo crashes TTGlyphPen with
    "invalid contour", which is exactly the failure pattern we hit on
    glyphs 'e', 'w', 'y', 'd', '8'.
    """
    in_contour = False

    for cmd, pts in _iter_svg_path(path_data):
        if cmd == "M":
            if in_contour:
                pen.closePath()  # auto-close before new contour
            pen.moveTo(pts[0])
            in_contour = True
        elif cmd == "L":
            pen.lineTo(pts[0])
        elif cmd == "C":
            pen.curveTo(*pts)
        elif cmd == "Z" and in_contour:
            pen.closePath()
            in_contour = False

    if in_contour:
        pen.closePath()


def _replay_flattened(pen, path_data: str, segments: int = 12):
    """
    Fallback: replay SVG path with cubics flattened to N line segments each.
    Same defensive auto-close behavior as _replay_cubics.
    """
    cur = (0.0, 0.0)
    in_contour = False

    for cmd, pts in _iter_svg_path(path_data):
        if cmd == "M":
            if in_contour:
                pen.closePath()  # auto-close before new contour
            cur = pts[0]
            pen.moveTo(cur)
            in_contour = True
        elif cmd == "L":
            cur = pts[0]
            pen.lineTo(cur)
        elif cmd == "C":
            p0 = cur
            p1, p2, p3 = pts
            for j in range(1, segments + 1):
                t = j / segments
                u = 1 - t
                x = u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0]
                y = u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]
                pen.lineTo((x, y))
            cur = p3
        elif cmd == "Z" and in_contour:
            pen.closePath()
            in_contour = False

    if in_contour:
        pen.closePath()


def _iter_svg_path(path_data: str):
    """
    Yield absolute uppercase SVG operations as (cmd, points).

    Potrace often emits relative lowercase cubics. If the parser ignores the
    "c" command token, the six cubic deltas are consumed as three absolute
    line segments, producing the classic diagonal slash/starburst glyph while
    still leaving a structurally valid font.
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


def _validate_font(path: str) -> None:
    """
    Re-open the saved font and run the diagnostics from FontResearch §8:
      - all required tables present?
      - any cubic-glyf flags? (Windows-fatal if present)
      - head.glyphDataFormat == 0?
      - every glyph drawable?
    """
    required = {"cmap", "head", "hhea", "hmtx", "maxp", "name", "OS/2", "post", "glyf", "loca"}
    try:
        test = TTFont(path)
        present = set(test.keys())
        missing = required - present
        if missing:
            print(f"[build] self-check WARNING: missing required tables: {sorted(missing)}")
        else:
            print(f"[build] self-check: all required tables present")

        # The #1 cause of "shattered glyph" rendering on Windows is the cubic-glyf
        # extension flag bit being set in a font advertising classic format.
        # Confirm we're clean.
        glyf = test["glyf"]
        head_format = test["head"].glyphDataFormat
        cubic_glyphs = []
        for name in test.getGlyphOrder():
            g = glyf[name]
            if g.numberOfContours > 0 and any(fl & flagCubic for fl in g.flags):
                cubic_glyphs.append(name)

        if cubic_glyphs or head_format != 0:
            print(f"[build] self-check WARNING: cubic-glyf extension detected!")
            print(f"[build]   head.glyphDataFormat = {head_format} (must be 0 for Windows)")
            print(f"[build]   {len(cubic_glyphs)} glyph(s) carry cubic flags: {cubic_glyphs[:10]}")
            print(f"[build]   This is the documented #1 cause of garbled glyphs on Windows.")
        else:
            print(f"[build] self-check: no cubic-glyf flags (Windows-safe), glyphDataFormat=0")

        # Verify every glyph draws
        errors = 0
        for name in test.getGlyphOrder():
            pen = BoundsPen(None)
            try:
                glyf[name].draw(pen, glyf)
            except Exception as e:
                print(f"[build]   ! glyph '{name}' draw failed: {type(e).__name__}: {e}")
                errors += 1
        if errors == 0:
            print(f"[build] self-check: all {len(test.getGlyphOrder())} glyphs draw OK")

        # Hint: ots-sanitize is the validator Chrome/Firefox use. If it accepts
        # the font, the major browsers and Windows will too.
        print(f"[build] tip: validate against Chromium's checker with:")
        print(f"[build]   pip install opentype-sanitizer && ots-sanitize \"{path}\" check.ttf")
    except Exception as e:
        print(f"[build] self-check FAILED to re-open: {type(e).__name__}: {e}")


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


def _metrics(glyph_paths: dict[str, str], glyf_table=None) -> dict[str, tuple[int, int]]:
    """Return {glyph_name: (advance_width, lsb)} using real xMin if available."""
    def lsb_for(gname: str, default: int) -> int:
        if glyf_table is None or gname not in glyf_table.glyphs:
            return default
        g = glyf_table[gname]
        return getattr(g, "xMin", default)

    m = {
        ".notdef": (DEFAULT_ADVANCE, lsb_for(".notdef", 50)),
        "space":   (SPACE_ADVANCE,   lsb_for("space", 0)),
    }
    for ch in glyph_paths:
        gname = _name(ch)
        if ch in NARROW:
            adv = 300
        elif ch in WIDE:
            adv = 750
        else:
            adv = DEFAULT_ADVANCE
        m[gname] = (adv, lsb_for(gname, 50))
    return m


def _dedup(lst: list[str]) -> list[str]:
    seen, out = set(), []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
