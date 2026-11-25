# bpe-tokenizer-from-scratch

[![CI](https://github.com/kishorhange111/bpe-tokenizer-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/bpe-tokenizer-from-scratch/actions/workflows/ci.yml)

Byte-level BPE, the tokenizer behind GPT-2, GPT-4 and Llama 3, written from scratch and
held to an exact standard:
- **The fast trainer produces the same merges as the textbook one, merge for merge.**
- **Loaded with GPT-2's merge table, the encoder produces byte-for-byte the same tokens as
  OpenAI's tiktoken.**

Around that, the repo times what the data-structure choices buy: training, encoding,
pathological inputs and compression.

```
text ──regex──► pre-tokens ──► counts ──train──► merges (a, b) → id ──encode──► token ids
```

## Pieces

**[`pretok.py`](src/bpetok/pretok.py): pre-tokenization.** This is GPT-2's split regex and
cl100k's (GPT-4's). Merges never cross a pre-token boundary. cl100k additionally splits
numbers into groups of at most 3 digits and keeps newlines apart from other whitespace.

**[`train.py`](src/bpetok/train.py): two trainers.** Both start from the bytes of each
unique pre-token, weighted by its count, and repeatedly turn the most frequent adjacent
pair into a new token. Ties go to the smallest pair, so the result is deterministic.
- `train_naive` recounts every pair after every merge.
- `train_fast` keeps the pair counts and an index from each pair to the words containing
  it. A merge only rewrites the words that contain the merged pair, subtracting their old
  pairs and adding their new ones. The next pair comes from a **lazy max-heap**: an entry
  whose count has changed is skipped when popped, and every change pushes a fresh entry.

**[`tokenizer.py`](src/bpetok/tokenizer.py): two encoders.** To encode a piece, start from
its bytes and keep applying the lowest-ranked merge available.
- `simple` scans every adjacent pair at each step, which is `O(n²)` per piece.
- `heap` keeps the piece as a **doubly linked list** and the candidate pairs in a
  **min-heap** keyed by `(rank, position)`, which is `O(n log n)`. After a merge only the
  two new neighbouring pairs are pushed. Invalidated entries are detected when popped. The
  heap applies merges in the same order as `simple` for a simple reason: a pair that
  includes a new token always ranks after the merge that created it.

A per-piece cache skips repeated words. Special tokens are encoded only when explicitly
allowed; otherwise they are plain text, which is the safe default against prompt
injection.

**Loading GPT-2.** tiktoken ships only `token bytes → rank`. `Tokenizer.from_tiktoken`
recovers the merge table by running BPE on each token's bytes using only lower ranks.
Exactly two parts remain, and they are the merge that created the token.

## Tests

There are 18. The GPT-2 tests download tiktoken's 1.6 MB BPE file and skip without
network. They check:
- the textbook example (Sennrich et al. 2016: *low, lower, newest, widest*) gives the
  merges `es, est, lo, low`;
- **`train_fast` equals `train_naive` exactly** on 6 random corpora (including runs of one
  byte, which create overlapping pairs) and on real text;
- **the heap encoder equals the simple encoder** on random inputs, including runs of up to
  1,000 identical bytes;
- `decode(encode(x)) == x` on random Unicode (CJK, emoji, combining marks, CRLF);
- special tokens, save/load, and the GPT-2 and cl100k split patterns;
- **GPT-2's 50,000 merges are recovered**, and encoding matches tiktoken exactly on
  mixed text, 30 random-alphabet strings and 30 strings of random code points, with both
  encoders.

```
pip install -e ".[dev]"
pytest -q                             # 18 tests
pip install -e ".[bench]"
python -m bpetok.bench --out results  # CPU benchmark
```

**Results are pending** from the benchmark run.
