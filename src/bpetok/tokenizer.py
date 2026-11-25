"""Encoding and decoding with a trained merge table, including GPT-2's own.

A merge table maps a pair of token ids to the id of their merge. Ids are assigned in
merge order, so the merged id is also the merge's priority (its *rank*). To encode a
pre-token, start from its bytes and repeatedly apply the lowest-ranked merge available
among adjacent pairs, until none applies. Two implementations give identical output:

- ``simple`` scans all adjacent pairs at every step: ``O(n²)`` for an ``n``-byte piece.
  This is how GPT-2's original ``encoder.py`` works (it merges the pair everywhere at
  once, which is equivalent).
- ``heap`` keeps the pieces in a doubly linked list and the candidate pairs in a min-heap
  keyed by ``(rank, position)``, which is ``O(n log n)``. After a merge, only the two new
  neighbouring pairs are pushed. Entries invalidated by an earlier merge are detected when
  popped and skipped. A new pair always has a higher rank than the merge that created it,
  because the merged token must exist before anything can merge with it. So popping in
  ``(rank, position)`` order applies merges in exactly the order ``simple`` does.

Normal text splits into short pre-tokens, so both are fast. The difference shows on long
pre-tokens with no spaces: minified code, base64, DNA, long runs of one character.

``Tokenizer.from_tiktoken`` loads an OpenAI encoding (``gpt2``, ``cl100k_base``, …).
tiktoken stores only ``token bytes -> rank``, so the merge table is recovered by running
BPE on each token's bytes using only lower ranks. Exactly two parts remain, and they are
the merge that created it.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Iterable

import regex

from .pretok import PATTERNS, compile_pattern
from .train import Pair


class Tokenizer:
    def __init__(
        self,
        merges: dict[Pair, int],
        byte_ids: list[int] | None = None,
        pattern: str = "gpt2",
        special: dict[str, int] | None = None,
        cache_size: int = 1 << 16,
    ):
        self.merges = merges
        self.byte_ids = byte_ids if byte_ids is not None else list(range(256))
        assert len(self.byte_ids) == 256
        self.pattern = PATTERNS.get(pattern, pattern)
        self.pat = compile_pattern(self.pattern)
        self.special = dict(special or {})
        self.vocab: dict[int, bytes] = {self.byte_ids[b]: bytes([b]) for b in range(256)}
        for (a, b), c in sorted(merges.items(), key=lambda kv: kv[1]):
            self.vocab[c] = self.vocab[a] + self.vocab[b]
        for s, i in self.special.items():
            self.vocab[i] = s.encode("utf-8")
        self.cache_size = cache_size
        self._cache: dict[bytes, list[int]] = {}
        self._special_pat = (
            regex.compile("|".join(regex.escape(s) for s in sorted(self.special, key=len)[::-1]))
            if self.special
            else None
        )

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_merges(cls, merges: Iterable[Pair], pattern: str = "gpt2", **kw) -> Tokenizer:
        """From a trainer's output: the i-th merge creates token ``256 + i``."""
        return cls({p: 256 + i for i, p in enumerate(merges)}, None, pattern, **kw)

    @classmethod
    def from_tiktoken(cls, name: str = "gpt2", **kw) -> Tokenizer:
        import tiktoken

        enc = tiktoken.get_encoding(name)
        ranks: dict[bytes, int] = enc._mergeable_ranks
        byte_ids = [ranks[bytes([b])] for b in range(256)]
        merges: dict[Pair, int] = {}
        for tok, rank in ranks.items():
            if len(tok) == 1:
                continue
            parts = _bpe_bytes(tok, ranks, rank)
            if len(parts) != 2:
                raise ValueError(f"token {tok!r} (rank {rank}) is not a merge of two tokens")
            merges[(ranks[parts[0]], ranks[parts[1]])] = rank
        return cls(merges, byte_ids, enc._pat_str, enc._special_tokens, **kw)

    # ------------------------------------------------------------------ BPE on one piece

    def bpe_simple(self, ids: list[int]) -> list[int]:
        ids = list(ids)
        get = self.merges.get
        while len(ids) >= 2:
            best_i, best_r = -1, None
            for i in range(len(ids) - 1):
                r = get((ids[i], ids[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best_i, best_r = i, r
            if best_r is None:
                break
            ids[best_i : best_i + 2] = [best_r]
        return ids

    def bpe_heap(self, ids: list[int]) -> list[int]:
        n = len(ids)
        if n < 2:
            return list(ids)
        get = self.merges.get
        tok = list(ids)
        nxt = list(range(1, n + 1))
        prv = list(range(-1, n - 1))
        alive = [True] * n
        heap = []
        for i in range(n - 1):
            r = get((tok[i], tok[i + 1]))
            if r is not None:
                heap.append((r, i, tok[i], tok[i + 1]))
        heapq.heapify(heap)
        while heap:
            r, i, a, b = heapq.heappop(heap)
            if not alive[i] or tok[i] != a:
                continue
            j = nxt[i]
            if j >= n or tok[j] != b:
                continue  # stale: a neighbour was merged since this entry was pushed
            tok[i] = r
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k < n:
                prv[k] = i
                r2 = get((r, tok[k]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, i, r, tok[k]))
            p = prv[i]
            if p >= 0:
                r2 = get((tok[p], r))
                if r2 is not None:
                    heapq.heappush(heap, (r2, p, tok[p], r))
        out, i = [], 0
        while i < n:
            out.append(tok[i])
            i = nxt[i]
        return out

    # ------------------------------------------------------------------ text

    def encode_ordinary(self, text: str, algo: str = "heap", cache: bool = True) -> list[int]:
        """Encode text, treating special-token strings as ordinary text."""
        bpe = self.bpe_heap if algo == "heap" else self.bpe_simple
        bid = self.byte_ids
        out: list[int] = []
        c = self._cache
        for piece in self.pat.findall(text):
            b = piece.encode("utf-8")
            ids = c.get(b) if cache else None
            if ids is None:
                ids = bpe([bid[x] for x in b])
                if cache and len(c) < self.cache_size:
                    c[b] = ids
            out.extend(ids)
        return out

    def encode(self, text: str, allowed_special: Iterable[str] | str = (), **kw) -> list[int]:
        """Encode text. Special tokens in ``allowed_special`` (or ``"all"``) become their
        ids; anywhere else they are ordinary text, which is the safe default."""
        allowed = set(self.special) if allowed_special == "all" else set(allowed_special)
        if not allowed or self._special_pat is None:
            return self.encode_ordinary(text, **kw)
        out, start = [], 0
        for m in self._special_pat.finditer(text):
            if m.group(0) not in allowed:
                continue
            out += self.encode_ordinary(text[start : m.start()], **kw)
            out.append(self.special[m.group(0)])
            start = m.end()
        return out + self.encode_ordinary(text[start:], **kw)

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pattern": self.pattern,
                    "byte_ids": self.byte_ids,
                    "merges": sorted([a, b, c] for (a, b), c in self.merges.items()),
                    "special": self.special,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, **kw) -> Tokenizer:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        merges = {(a, b): c for a, b, c in d["merges"]}
        return cls(merges, d["byte_ids"], d["pattern"], d["special"], **kw)


def _bpe_bytes(tok: bytes, ranks: dict[bytes, int], max_rank: int) -> list[bytes]:
    """BPE over byte strings using only ranks below ``max_rank`` (merge-table recovery)."""
    parts = [tok[i : i + 1] for i in range(len(tok))]
    while True:
        best_i, best_r = -1, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < max_rank and (best_r is None or r < best_r):
                best_i, best_r = i, r
        if best_r is None:
            return parts
        parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
