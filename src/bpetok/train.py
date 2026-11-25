"""Byte-level BPE training, two ways, with identical output.

The input is the pre-token counts (``word -> frequency``). Every word starts as its bytes
(ids 0–255). At each step the most frequent adjacent pair, weighted by word frequency,
becomes a new token, ``256 + step``. Ties go to the smallest pair ``(a, b)``, so the
result is deterministic and both trainers agree exactly.

``train_naive`` recounts every pair in every word after each merge:
``O(merges × total symbols)``.

``train_fast`` keeps the pair counts and, for each pair, the set of words containing it.
A merge touches only the words that contain the merged pair. In each one it subtracts the
word's old pairs and adds its new ones. The next pair comes from a lazy max-heap: entries
whose count has since changed are skipped when popped, and every change pushes a fresh
entry. The cost per merge is proportional to the occurrences of that pair, which falls
fast as merges proceed. This is the same idea as the Hugging Face ``tokenizers`` trainer.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from collections.abc import Mapping

Pair = tuple[int, int]


def merge_word(w: list[int], pair: Pair, new: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``w``, left to right."""
    a, b = pair
    out, i, n = [], 0, len(w)
    while i < n:
        if i + 1 < n and w[i] == a and w[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(w[i])
            i += 1
    return out


def train_naive(counts: Mapping[bytes, int], vocab_size: int, min_freq: int = 2) -> list[Pair]:
    words = [list(w) for w in counts]
    freqs = list(counts.values())
    merges: list[Pair] = []
    while 256 + len(merges) < vocab_size:
        pc: Counter[Pair] = Counter()
        for w, f in zip(words, freqs, strict=True):
            for p in zip(w, w[1:], strict=False):
                pc[p] += f
        if not pc:
            break
        best = min(pc, key=lambda p: (-pc[p], p))
        if pc[best] < min_freq:
            break
        new = 256 + len(merges)
        merges.append(best)
        words = [merge_word(w, best, new) for w in words]
    return merges


def train_fast(counts: Mapping[bytes, int], vocab_size: int, min_freq: int = 2) -> list[Pair]:
    words = [list(w) for w in counts]
    freqs = list(counts.values())
    pc: dict[Pair, int] = defaultdict(int)
    where: dict[Pair, set[int]] = defaultdict(set)
    for i, (w, f) in enumerate(zip(words, freqs, strict=True)):
        for p in zip(w, w[1:], strict=False):
            pc[p] += f
            where[p].add(i)
    heap = [(-c, p) for p, c in pc.items()]
    heapq.heapify(heap)
    merges: list[Pair] = []
    while heap and 256 + len(merges) < vocab_size:
        negc, p = heapq.heappop(heap)
        c = pc.get(p, 0)
        if c != -negc:
            continue  # stale entry: the count changed after it was pushed
        if c < min_freq:
            break
        new = 256 + len(merges)
        merges.append(p)
        changed: set[Pair] = set()
        for i in where.pop(p, ()):
            w, f = words[i], freqs[i]
            for q in zip(w, w[1:], strict=False):
                pc[q] -= f
                changed.add(q)
            w = merge_word(w, p, new)
            words[i] = w
            for q in zip(w, w[1:], strict=False):
                pc[q] += f
                changed.add(q)
                where[q].add(i)
        for q in changed:
            if pc[q] > 0:
                heapq.heappush(heap, (-pc[q], q))
            else:
                del pc[q]
    return merges
