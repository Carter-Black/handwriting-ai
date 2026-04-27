import sys
import unittest
import re
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from font_generator.prompt import (
    FINETUNE_PROMPT,
    FINETUNE_PROMPT_LINES,
    FONT_PROMPT,
    FONT_PROMPT_LINES,
    FONT_TARGET_CHARS,
)


class FontPromptTests(unittest.TestCase):
    def test_prompt_has_exactly_three_visual_lines(self):
        self.assertEqual(len(FONT_PROMPT_LINES), 3)
        self.assertEqual(FONT_PROMPT, "\n".join(FONT_PROMPT_LINES))

    def test_finetune_prompt_uses_sentence_lines(self):
        self.assertEqual(len(FINETUNE_PROMPT_LINES), 3)
        self.assertEqual(FINETUNE_PROMPT, "\n".join(FINETUNE_PROMPT_LINES))
        self.assertTrue(all(" " in line for line in FINETUNE_PROMPT_LINES))
        self.assertIn("quick brown fox", FINETUNE_PROMPT)

    def test_prompt_contains_all_required_font_characters(self):
        expected = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?.,;:'\"()-/")
        self.assertEqual(set(FONT_TARGET_CHARS), expected)

    def test_prompt_splits_lowercase_uppercase_and_numbers_rows(self):
        lowercase = FONT_PROMPT_LINES[0]
        uppercase = FONT_PROMPT_LINES[1]
        numbers = FONT_PROMPT_LINES[2]
        for ch in "abcdefghijklmnopqrstuvwxyz":
            self.assertIn(ch, lowercase)
            self.assertNotIn(ch, uppercase)
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertIn(ch, uppercase)
            self.assertNotIn(ch, numbers)
        for ch in "0123456789!?.,;:'\"()-/":
            self.assertIn(ch, numbers)
        self.assertIn("A B C D E F", uppercase)
        self.assertNotIn("ABCDEFGHIJKLMNOPQRSTUVWXYZ ", uppercase)
        self.assertIn("!", FONT_PROMPT)

    def test_frontend_fallback_prompt_matches_backend_prompt(self):
        html = unescape((ROOT / "frontend" / "index.html").read_text(encoding="utf-8")).replace("\xa0", " ")
        html = re.sub(r"\s+", " ", html)
        for line in FONT_PROMPT_LINES + FINETUNE_PROMPT_LINES:
            self.assertIn(re.sub(r"\s+", " ", line), html)


if __name__ == "__main__":
    unittest.main()
