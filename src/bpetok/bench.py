"""CPU benchmark: training speed, encoding speed, worst-case pieces, compression.

    python -m bpetok.bench --out results

1. **Training.** Pre-token counts of TinyStories train come from the GPT-2 pattern, then
   ``train_fast`` and ``train_naive`` run to several vocabulary sizes. ``train_naive``
   stops once a run exceeds the time budget. Wherever both run, the merge lists are
   checked to be identical. The Hugging Face ``tokenizers`` BPE trainer (Rust,
   multi-threaded, with its own pre-tokenization included) is timed for reference.
2. **Encoding.** GPT-2's merge table is loaded from tiktoken, and the run encodes the
   TinyStories test split and the held-out standard-library code with:
   - the heap encoder, with and without the pre-token cache;
   - the simple encoder;
   - tiktoken itself (Rust, one thread).
   The output must be identical to tiktoken's on every byte of both test sets.
3. **One long piece.** A string of ``n`` random letters is a single pre-token, so BPE sees
   ``n`` bytes at once. Times are for the simple and heap encoders and for tiktoken.
4. **Compression.** Tokenizers with 8,192 tokens are trained on prose or on code, with the
   GPT-2 or cl100k pattern. The run reports bytes per token on the held-out prose and code,
   next to GPT-2 (50k) and cl100k_base (100k).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import time

from .corpus import python_stdlib, tinystories
from .pretok import word_counts
from .tokenizer import Tokenizer
from .train import train_fast, train_naive


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def bench_train(prose: str, sizes, naive_budget: float) -> dict:
    counts, pre_s = timed(word_counts, prose, "gpt2")
    res = {"pretokenize_s": pre_s, "unique_words": len(counts), "corpus_MB": len(prose) / 1e6}
    rows, naive_on = [], True
    for v in sizes:
        fast, fs = timed(train_fast, counts, v)
        row = {"vocab": v, "fast_s": fs, "merges": len(fast)}
        if naive_on:
            naive, ns = timed(train_naive, counts, v)
            row.update(naive_s=ns, identical=naive == fast)
            naive_on = ns < naive_budget
        try:
            from tokenizers import Tokenizer as HFT
            from tokenizers import decoders, models, pre_tokenizers, trainers

            hf = HFT(models.BPE())
            hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            hf.decoder = decoders.ByteLevel()
            tr = trainers.BpeTrainer(
                vocab_size=v,
                show_progress=False,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            )
            lines = prose.split("\n")  # many items, so the Rust trainer can parallelise
            _, row["hf_s"] = timed(hf.train_from_iterator, lines, tr)
        except ImportError:
            pass
        rows.append(row)
        print(json.dumps(row), flush=True)
    res["rows"] = rows
    return res


def bench_encode(tok: Tokenizer, enc, texts: dict[str, str]) -> list[dict]:
    rows = []
    for name, text in texts.items():
        mb = len(text.encode()) / 1e6
        ref, ts = timed(enc.encode_ordinary, text)
        row = {"corpus": name, "MB": mb, "tokens": len(ref), "tiktoken_MBps": mb / ts}
        for label, kw in [
            ("heap+cache", {"algo": "heap", "cache": True}),
            ("heap", {"algo": "heap", "cache": False}),
            ("simple", {"algo": "simple", "cache": False}),
        ]:
            tok._cache.clear()
            ids, s = timed(tok.encode, text, **kw)
            assert ids == ref, f"{label} differs from tiktoken on {name}"
            row[f"{label}_MBps"] = mb / s
        row["identical_to_tiktoken"] = True
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def bench_long(tok: Tokenizer, enc, lengths, budget: float) -> list[dict]:
    rng = random.Random(0)
    rows, simple_on = [], True
    for n in lengths:
        text = "".join(rng.choice(string.ascii_lowercase) for _ in range(n))
        assert len(tok.pat.findall(text)) == 1
        ref, ts = timed(enc.encode_ordinary, text)
        ids, hs = timed(tok.encode, text, algo="heap", cache=False)
        assert ids == ref
        row = {"n": n, "tiktoken_s": ts, "heap_s": hs}
        if simple_on:
            ids, ss = timed(tok.encode, text, algo="simple", cache=False)
            assert ids == ref
            row["simple_s"] = ss
            simple_on = ss < budget
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def bench_compression(train: dict[str, str], test: dict[str, str], vocab: int) -> list[dict]:
    import tiktoken

    rows = []
    for (tname, ttext), pat in [(t, p) for t in train.items() for p in ("gpt2", "cl100k")]:
        tok = Tokenizer.from_merges(train_fast(word_counts(ttext, pat), vocab), pat)
        row = {"tokenizer": f"BPE-{vocab} on {tname}, {pat} pattern"}
        for name, text in test.items():
            row[name] = len(text.encode()) / len(tok.encode(text))
        rows.append(row)
        print(json.dumps(row), flush=True)
    for name in ("gpt2", "cl100k_base"):
        enc = tiktoken.get_encoding(name)
        row = {"tokenizer": f"{name} ({enc.n_vocab:,})"}
        for tn, text in test.items():
            row[tn] = len(text.encode()) / len(enc.encode_ordinary(text))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def report(r: dict) -> str:
    t = r["train"]
    L = [
        f"CPU: {r['cpu']}. Single-threaded Python {r['python']}.",
        "",
        f"### Training on TinyStories train ({t['corpus_MB']:.1f} MB,"
        f" {t['unique_words']:,} unique pre-tokens; pre-tokenization {t['pretokenize_s']:.1f} s)",
        "",
        "| vocab | fast (s) | naive (s) | speed-up | identical merges | HF tokenizers, Rust (s) |",
        "|---|---|---|---|---|---|",
    ]
    for x in t["rows"]:
        ns, fs = x.get("naive_s"), x["fast_s"]
        L.append(
            f"| {x['vocab']:,} | {fs:.1f} | {'–' if ns is None else f'{ns:.1f}'} |"
            f" {'–' if ns is None else f'{ns / fs:.0f}×'} |"
            f" {'–' if ns is None else ('yes' if x['identical'] else 'NO')} |"
            f" {x.get('hf_s', float('nan')):.1f} |"
        )
    L += [
        "",
        "### Encoding with GPT-2's merges (MB/s; output identical to tiktoken)",
        "",
        "| corpus | MB | tokens | heap + cache | heap | simple | tiktoken (Rust) |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in r["encode"]:
        L.append(
            f"| {x['corpus']} | {x['MB']:.2f} | {x['tokens']:,} | {x['heap+cache_MBps']:.2f} |"
            f" {x['heap_MBps']:.2f} | {x['simple_MBps']:.2f} | {x['tiktoken_MBps']:.2f} |"
        )
    L += [
        "",
        "### One long pre-token of n random letters (seconds)",
        "",
        "| n | simple | heap | tiktoken (Rust) |",
        "|---|---|---|---|",
    ]
    for x in r["long"]:
        s = x.get("simple_s")
        L.append(
            f"| {x['n']:,} | {'–' if s is None else f'{s:.3f}'} | {x['heap_s']:.3f} |"
            f" {x['tiktoken_s']:.3f} |"
        )
    names = [k for k in r["compression"][0] if k != "tokenizer"]
    L += [
        "",
        "### Compression: bytes per token on held-out text (higher is better)",
        "",
        "| tokenizer | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for x in r["compression"]:
        L.append(f"| {x['tokenizer']} | " + " | ".join(f"{x[n]:.2f}" for n in names) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--sizes", default="512,1024,2048,4096,8192,16384,32768")
    ap.add_argument("--naive-budget", type=float, default=600)
    ap.add_argument("--long", default="256,1024,4096,16384,65536,262144")
    ap.add_argument("--compression-vocab", type=int, default=8192)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    import platform

    import tiktoken

    prose_tr, prose_te = tinystories("train"), tinystories("test")
    code_tr, code_te = python_stdlib("train"), python_stdlib("test", max_bytes=2 << 20)
    r = {"cpu": platform.processor() or platform.machine(), "python": platform.python_version()}
    try:
        with open("/proc/cpuinfo") as f:
            r["cpu"] = next(x.split(":")[1].strip() for x in f if x.startswith("model name"))
    except (OSError, StopIteration):
        pass
    r["train"] = bench_train(prose_tr, [int(x) for x in a.sizes.split(",")], a.naive_budget)
    enc = tiktoken.get_encoding("gpt2")
    tok = Tokenizer.from_tiktoken("gpt2")
    r["encode"] = bench_encode(tok, enc, {"prose (TinyStories test)": prose_te, "code": code_te})
    r["long"] = bench_long(tok, enc, [int(x) for x in a.long.split(",")], budget=60)
    r["compression"] = bench_compression(
        {"prose": prose_tr, "code": code_tr},
        {"prose": prose_te, "code": code_te},
        a.compression_vocab,
    )
    with open(os.path.join(a.out, "bench.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    md = report(r)
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
