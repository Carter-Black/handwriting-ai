"""Canonical prompt text for transcription fine-tuning and font generation."""

from __future__ import annotations

import string

FONT_PROMPT_LINES = [
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "0 1 2 3 4 5 6 7 8 9 ! ? . , ; : ' \" ( ) - /",
]

FONT_PROMPT = "\n".join(FONT_PROMPT_LINES)

FINETUNE_PROMPT_LINES = [
    "the quick brown fox jumps over the lazy dog.",
    "pack my box with five dozen liquor jugs.",
    "how vexingly quick daft zebras jump!",
]

FINETUNE_PROMPT = "\n".join(FINETUNE_PROMPT_LINES)

FONT_TARGET_CHARS = "".join(
    ch for ch in (
        string.ascii_lowercase
        + string.ascii_uppercase
        + string.digits
        + r"""!?.,;:'"()-/"""
    )
    if ch in FONT_PROMPT
)


def prompt_words_by_line() -> list[list[str]]:
    """Return expected non-space tokens per visual line."""
    return [line.split() for line in FONT_PROMPT_LINES]
