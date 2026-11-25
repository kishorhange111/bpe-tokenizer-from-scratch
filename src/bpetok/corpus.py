"""Benchmark corpora, with no heavy dependencies.

- **prose**: TinyStories V2 (the GPT-4-written validation file, 22.5 MB of plain text),
  downloaded once from the Hugging Face hub and cached under ``~/.cache/bpetok``. Stories
  are split 90/10 into train and test by a hash of their text. The English is simple, with a
  small vocabulary, which is why its compression numbers are high.
- **code**: the Python standard library's own ``.py`` files, already on disk. Files are
  split 90/10 by a hash of their path, so the split is stable.
"""

from __future__ import annotations

import hashlib
import os
import sysconfig
import urllib.request

CACHE = os.path.expanduser("~/.cache/bpetok")
TINYSTORIES = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt"
)


def _is_test(key: str) -> bool:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10 == 0


def tinystories(split: str = "train") -> str:
    path = os.path.join(CACHE, "TinyStoriesV2-GPT4-valid.txt")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(TINYSTORIES, path + ".part")
        os.replace(path + ".part", path)
    with open(path, encoding="utf-8") as f:
        stories = [s.strip() for s in f.read().split("<|endoftext|>") if s.strip()]
    keep = [s for s in stories if _is_test(s) == (split == "test")]
    return "\n\n".join(keep)


def python_stdlib(split: str = "train", max_bytes: int = 12 << 20) -> str:
    root = sysconfig.get_paths()["stdlib"]
    files = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in ("site-packages", "test", "__pycache__"))
        files += [os.path.join(d, f) for f in sorted(fs) if f.endswith(".py")]
    out, n = [], 0
    for p in files:
        if _is_test(os.path.relpath(p, root)) != (split == "test"):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            s = f.read()
        out.append(s)
        n += len(s)
        if n >= max_bytes:
            break
    return "\n".join(out)
