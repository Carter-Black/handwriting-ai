import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

try:
    from recognizer.model import _clean_transcription
except Exception as exc:  # pragma: no cover - depends on ML stack availability
    _clean_transcription = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(_clean_transcription is None, f"Recognizer dependencies unavailable: {IMPORT_ERROR}")
class RecognizerCleanupTests(unittest.TestCase):
    def test_trims_alphabet_suffix_after_natural_words(self):
        text = "be more NEAT ! ? a b c d e f g h i j k l m n"
        self.assertEqual(_clean_transcription(text), "be more NEAT ! ?")

    def test_keeps_actual_prompt_row(self):
        text = "a b c d e f g h i j k l m n o p"
        self.assertEqual(_clean_transcription(text), text)


if __name__ == "__main__":
    unittest.main()
