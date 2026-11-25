"""Pre-tokenization: split text into pieces that BPE never merges across.

Byte-level BPE (GPT-2 onwards) first splits text with a regex, so that merges stay
inside words, numbers or runs of punctuation. A merge never joins "dog" and "." or a
space and the word before it. The two patterns below are GPT-2's and cl100k's (GPT-4's).
cl100k makes three changes:
- contractions match case-insensitively;
- numbers are split into groups of at most 3 digits, so the vocabulary cannot learn
  arbitrary long numbers;
- newlines are kept apart from other whitespace.
"""

from __future__ import annotations

from collections import Counter

import regex

GPT2 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CL100K = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)
PATTERNS = {"gpt2": GPT2, "cl100k": CL100K}


def compile_pattern(pattern: str) -> regex.Pattern:
    """A pattern name ("gpt2", "cl100k") or a raw regex."""
    return regex.compile(PATTERNS.get(pattern, pattern))


def word_counts(text: str, pattern: str = "gpt2") -> Counter[bytes]:
    """How often each pre-token occurs, as UTF-8 bytes: the only input BPE training needs."""
    pat = compile_pattern(pattern)
    return Counter(m.encode("utf-8") for m in pat.findall(text))
