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

## Results (CPU: Intel i5-1135G7 laptop, Python 3.12, single runs)

The output is in [`results/bench.md`](results/bench.md) and
[`results/bench.json`](results/bench.json). Every number below comes from a single run on a
laptop, so differences under about 2× in the small timings are noise.

### Training: 19.9 MB of TinyStories, 12,642 unique pre-tokens

| vocab | merges | fast (s) | naive (s) | speed-up | identical merges |
|---|---|---|---|---|---|
| 512 | 256 | 0.4 | 9.9 | 27× | yes |
| 1,024 | 768 | 0.6 | 22.8 | 41× | yes |
| 2,048 | 1,792 | 2.0 | 50.2 | 25× | yes |
| 4,096 | 3,840 | 0.6 | 110.7 | 179× | yes |
| 8,192 | 7,936 | 0.7 | 215.9 | 290× | yes |
| 16,384 | 13,045 | 1.2 | 364.5 | **305×** | yes |

- **The incremental trainer is 25–305× faster, and its merges are identical at every
  size.** Naive time grows linearly with the number of merges, since each one rescans
  every word. The fast trainer's cost per merge shrinks, because late merges are rare
  pairs that occur in only a few words.
- **Training stops at 13,045 merges.** After that, no pair occurs twice. TinyStories has a
  small vocabulary, so a 32,768-token target gives the same result as 16,384, and its row is
  left out of this table.
- Pre-tokenizing the corpus takes 2.6 s, so for both trainers the regex costs more than the
  merging. For reference, Hugging Face's Rust trainer, which pre-tokenizes in parallel, took
  2.5–6.6 s end to end. That is not like-for-like with the merge-only times above.

### Encoding: GPT-2's merges, output identical to tiktoken on every byte

| corpus | MB | tokens | heap + cache | heap | simple | tiktoken (Rust) |
|---|---|---|---|---|---|---|
| prose (TinyStories test) | 2.27 | 564,837 | **6.9 MB/s** | 1.3 | 1.9 | 18.2 |
| code (stdlib, held out) | 0.62 | 285,720 | **4.2 MB/s** | 1.5 | 1.9 | 10.6 |

- **On normal text the cache matters far more than the algorithm.** Repeated words skip BPE
  entirely, which makes the encoder 3–5× faster. Without the cache, the heap encoder is a
  bit *slower* than the simple one, because pieces are short (a word or two) and the
  heap's bookkeeping costs more than an `O(n²)` scan over a handful of bytes.
- Pure Python ends up 2.5–2.6× slower than tiktoken's Rust.

### One long pre-token: n random letters with no spaces

| n | simple | heap | tiktoken (Rust) |
|---|---|---|---|
| 1,024 | 0.040 s | 0.001 s | 0.000 s |
| 4,096 | 0.538 s | 0.003 s | 0.001 s |
| 16,384 | 12.1 s | 0.023 s | 0.002 s |
| 65,536 | **149.6 s** | **0.091 s** | 0.008 s |
| 262,144 | – | 0.61 s | 0.054 s |

**This is where the heap matters.** The simple encoder is quadratic: 4× the input takes about
12–23× the time, reaching 2.5 minutes for 64 KiB. The heap encoder grows 4–8× per 4× of input, close to
linear, taking 0.09 s for 64 KiB, which is **1,600× faster**. Inputs like this are real: minified
JavaScript, base64, hex dumps and DNA, or text crafted to stall a tokenizer. tiktoken copes
here too.

### Compression: bytes per token on held-out text (higher is better)

| tokenizer | prose | code |
|---|---|---|
| BPE-8192 trained on prose, GPT-2 pattern | 4.06 | 1.88 |
| BPE-8192 trained on prose, cl100k pattern | 4.18 | 1.62 |
| BPE-8192 trained on code, GPT-2 pattern | 2.77 | 3.69 |
| BPE-8192 trained on code, cl100k pattern | 2.77 | 3.67 |
| GPT-2 (50,257 tokens) | 4.02 | 2.16 |
| cl100k_base (100,277 tokens) | 4.15 | 3.91 |

- **The training domain matters more than vocabulary size.** An 8k vocabulary trained on
  code compresses code 1.7× better than GPT-2's 50k (3.69 against 2.16 bytes per token).
  GPT-2's training data had little code. On TinyStories, the prose-trained 8k vocabulary matches GPT-2's 50k
  (4.06 against 4.02), because this prose is simple.
- **The pattern matters less than the data.** cl100k's pattern helps on prose (4.18 against
  4.06) and hurts on code when trained on prose (1.62 against 1.88). Trained on code, the
  two patterns are within 1%.
- cl100k_base, with 100k tokens trained on broad data including code, is the best
  all-rounder here (4.15 on prose, 3.91 on code).
