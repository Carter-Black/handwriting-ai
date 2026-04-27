import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

try:
    import cv2
    import numpy as np
    from font_generator import segmenter
except Exception as exc:  # pragma: no cover - depends on local OpenCV install
    cv2 = None
    np = None
    segmenter = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(segmenter is None, f"OpenCV segmenter dependencies unavailable: {IMPORT_ERROR}")
class SegmenterAlignmentTests(unittest.TestCase):
    def test_word_split_uses_largest_gaps(self):
        boxes = [(0, 0, 5, 10), (8, 0, 5, 10), (40, 0, 5, 10), (48, 0, 5, 10)]
        groups = segmenter._split_into_word_groups(boxes, 2)
        self.assertEqual(groups, [boxes[:2], boxes[2:]])

    def test_double_quote_can_consume_two_components_without_shifting_row(self):
        boxes = [
            (0, 0, 5, 20),
            (20, 0, 3, 8),
            (26, 0, 3, 8),
            (46, 0, 8, 20),
        ]
        groups = segmenter._split_into_word_groups(boxes, ["1", '"', "("])
        self.assertEqual(groups, [[boxes[0]], [boxes[1], boxes[2]], [boxes[3]]])

    def test_too_few_components_for_words_rejects(self):
        boxes = [(0, 0, 5, 10)]
        self.assertIsNone(segmenter._split_into_word_groups(boxes, 2))

    def test_vertical_mark_merges_with_stem(self):
        boxes = [(10, 20, 6, 30), (11, 5, 4, 5)]
        merged = segmenter._merge_vertical_marks(sorted(boxes, key=lambda b: b[0]))
        self.assertEqual(len(merged), 1)

    def test_i_j_dots_merge_even_when_not_adjacent_to_stem(self):
        boxes = [(0, 15, 8, 25), (25, 15, 9, 26), (26, 2, 4, 4), (1, 3, 4, 4)]
        merged = segmenter._merge_vertical_marks(sorted(boxes, key=lambda b: b[0]))
        self.assertEqual(len(merged), 2)
        self.assertTrue(any(b[1] <= 3 and b[3] >= 36 for b in merged))

    def test_wide_component_can_split_at_empty_valley(self):
        line = np.zeros((30, 40), dtype=np.uint8)
        line[5:25, 2:10] = 255
        line[5:25, 26:34] = 255
        left, right = segmenter._split_box_at_valley(line, (2, 5, 32, 20))
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertGreater(right[0], left[0])

    def test_touching_letters_can_split_at_shallow_valley(self):
        line = np.zeros((30, 42), dtype=np.uint8)
        line[5:25, 2:15] = 255
        line[12:18, 15:20] = 255
        line[5:25, 20:35] = 255
        left, right = segmenter._split_box_at_valley(line, (2, 5, 33, 20))
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)

    def test_component_cluster_line_fallback_recovers_expected_rows(self):
        img = np.zeros((90, 260), dtype=np.uint8)
        for row_y in (8, 34, 60):
            for x in range(5, 240, 28):
                img[row_y:row_y + 18, x:x + 10] = 255
        # Add a faint bridge that would make pure projection less trustworthy.
        img[26:34, 128:132] = 255
        lines = segmenter._extract_lines_by_component_clusters(img, 3)
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
