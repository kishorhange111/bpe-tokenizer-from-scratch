import random
from collections import Counter

import pytest

from bpetok import Tokenizer, merge_word, train_fast, train_naive, word_counts
from bpetok.pretok import compile_pattern

SAMPLE = """The quick brown fox jumps over the lazy dog. It's 2026; we've seen 1234567 foxes!
    def f(x):  return x**2 + 3.14159  # comment
Ünïcödé, 日本語のテキスト, emoji 🙂👍🏽, combining é, tabs\tand\r\nCRLF lines.
aaaaaaaaaaaaaaaa bbbbbbbb    spaces   at the end   \n\n\n"""


def random_text(rng: random.Random, n: int) -> str:
    alphabet = "aaaabbbcdeé日🙂 \n\t.,'0123456789"
    return "".join(rng.choice(alphabet) for _ in range(n))


# ------------------------------------------------------------------ training


def test_merge_word_is_left_to_right_and_non_overlapping():
    assert merge_word([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge_word([1, 1, 1, 1], (1, 1), 9) == [9, 9]
    assert merge_word([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert merge_word([2, 1], (1, 2), 9) == [2, 1]


def test_textbook_example():
    # Sennrich et al. (2016): low ×5, lower ×2, newest ×6, widest ×3
    counts = {b"low": 5, b"lower": 2, b"newest": 6, b"widest": 3}
    merges = train_naive(counts, 256 + 4)
    tok = Tokenizer.from_merges(merges)
    assert [tok.vocab[256 + i] for i in range(4)] == [b"es", b"est", b"lo", b"low"]


@pytest.mark.parametrize("seed", range(6))
def test_fast_trainer_matches_naive_exactly(seed):
    rng = random.Random(seed)
    counts = word_counts(random_text(rng, 3000))
    a = train_naive(counts, 256 + 120, min_freq=1)
    b = train_fast(counts, 256 + 120, min_freq=1)
    assert a == b and len(a) > 20


def test_fast_trainer_matches_naive_on_text():
    counts = word_counts(SAMPLE * 3)
    assert train_naive(counts, 400) == train_fast(counts, 400)


def test_training_stops_when_nothing_is_frequent():
    counts = Counter({b"ab": 1, b"cd": 1})
    assert train_fast(counts, 1000) == [] == train_naive(counts, 1000)
    assert train_fast(counts, 1000, min_freq=1) == [(97, 98), (99, 100)]


# ------------------------------------------------------------------ encoding


@pytest.fixture(scope="module")
def trained():
    rng = random.Random(0)
    text = SAMPLE + random_text(rng, 20000)
    return Tokenizer.from_merges(train_fast(word_counts(text), 256 + 400))


def test_heap_encoder_matches_simple(trained):
    rng = random.Random(1)
    for _ in range(200):
        ids = [rng.choice(b"aaaabbbcde") for _ in range(rng.randint(0, 300))]
        assert trained.bpe_heap(ids) == trained.bpe_simple(ids)
    for n in (1, 2, 3, 64, 1000):  # long runs of one byte: worst case for overlaps
        assert trained.bpe_heap([97] * n) == trained.bpe_simple([97] * n)


def test_roundtrip_and_algorithms_agree(trained):
    rng = random.Random(2)
    for _ in range(50):
        t = random_text(rng, rng.randint(0, 500))
        ids = trained.encode(t, algo="heap", cache=False)
        assert ids == trained.encode(t, algo="simple", cache=False) == trained.encode(t)
        assert trained.decode(ids) == t
    assert trained.decode(trained.encode(SAMPLE)) == SAMPLE


def test_compresses(trained):
    assert len(trained.encode(SAMPLE)) < 0.7 * len(SAMPLE.encode())


def test_special_tokens():
    tok = Tokenizer.from_merges([], special={"<|endoftext|>": 256})
    t = "hi<|endoftext|>there"
    assert 256 not in tok.encode(t)  # not allowed: plain text
    assert tok.encode(t, allowed_special="all") == [*b"hi", 256, *b"there"]
    assert tok.decode(tok.encode(t, allowed_special={"<|endoftext|>"})) == t


def test_save_load(tmp_path, trained):
    p = tmp_path / "tok.json"
    trained.save(str(p))
    tok2 = Tokenizer.load(str(p))
    assert tok2.encode(SAMPLE) == trained.encode(SAMPLE) and tok2.vocab == trained.vocab


def test_patterns():
    gpt2, cl = compile_pattern("gpt2"), compile_pattern("cl100k")
    assert gpt2.findall("I'm 12345 dogs!") == ["I", "'m", " 12345", " dogs", "!"]
    assert cl.findall("I'M 12345 dogs!") == ["I", "'M", " ", "123", "45", " dogs", "!"]
    assert gpt2.findall("a  b") == ["a", " ", " b"]  # \s+(?!\S) keeps the last space


# ------------------------------------------------------------------ GPT-2 compatibility


@pytest.fixture(scope="module")
def gpt2():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("gpt2")
        tok = Tokenizer.from_tiktoken("gpt2")
    except Exception as e:  # no network for the BPE files
        pytest.skip(f"gpt2 encoding unavailable: {e}")
    return enc, tok


def test_gpt2_merge_table_is_recovered(gpt2):
    enc, tok = gpt2
    assert len(tok.merges) == 50000 and tok.vocab_size == enc.n_vocab == 50257


def test_gpt2_encoding_is_identical_to_tiktoken(gpt2):
    enc, tok = gpt2
    rng = random.Random(3)
    texts = [SAMPLE, *SAMPLE.splitlines(), " " * 40, "x" * 500]
    texts += [random_text(rng, 400) for _ in range(30)]
    texts += ["".join(chr(rng.randint(32, 0x2FFF)) for _ in range(300)) for _ in range(30)]
    for t in texts:
        ids = enc.encode_ordinary(t)
        assert tok.encode(t, algo="heap") == ids, t
        assert tok.encode(t, algo="simple", cache=False) == ids, t
        assert tok.decode(ids) == t
    assert tok.encode("<|endoftext|>", allowed_special="all") == [50256]
