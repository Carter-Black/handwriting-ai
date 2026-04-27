import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

try:
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401
    from font_generator.vectorizer import _iter_svg_path, _normalize_to_em
except Exception as exc:  # pragma: no cover - depends on local OpenCV install
    _iter_svg_path = None
    _normalize_to_em = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(_normalize_to_em is None, f"Vectorizer dependencies unavailable: {IMPORT_ERROR}")
class VectorizerTests(unittest.TestCase):
    def test_normalize_does_not_vertically_flip_path(self):
        path = "M 10 10 L 10 90 L 30 90 L 30 10 Z"
        normalized = _normalize_to_em(path, 100, 100)
        ops = list(_iter_svg_path(normalized))
        points = [pt for _, pts in ops for pt in pts]
        bottom_left = points[0]
        top_left = points[1]
        self.assertLess(bottom_left[1], top_left[1])


if __name__ == "__main__":
    unittest.main()
