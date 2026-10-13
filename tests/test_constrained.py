"""The constrained decoder can never write a word outside the allowed set.

The adversary here is a random scorer: at every step it picks any token the
processor leaves open. If some sequence of choices could change a word, a
random walk over many seeds and inputs would find it.
"""

import random

import pytest
import torch

from trueform.constrained import VocabIndex, WordLattice, WordSafeLogitsProcessor, max_output_bytes
from trueform.labels import roundtrip_ok

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_REV = "7ae557604adf67be50417f59c2c2f167def9a775"

ADVERSARIAL = [
    "hey can you send me the report by friday i think john has it",
    "the meeting is at 3.30 pm on 12/05/2026 and costs $1,200",
    "a well-known state-of-the-art e-mail client",
    "don't won't it's rock 'n' roll o'neill's",
    "visit https://example.com/a?b=c&d=e or www.trueform.dev, ok",
    "i i i a a the the",
    "café naïve straße ǆungla",
    "emoji 👍 test 🙂",
    "x",
    "hello. world? already, punctuated",
    "UPPER Mixed iPhone mcdonald's",
    "1 2 3 4 5 6 7 8 9 10 11 12",
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REV)


@pytest.fixture(scope="module")
def index(tokenizer):
    return VocabIndex(tokenizer)


def test_byte_mapping_matches_tokenizer(tokenizer, index):
    rng = random.Random(0)
    ids = rng.sample(sorted(index.token_bytes), 2000)
    for i in ids:
        b = index.token_bytes[i]
        try:
            expected = b.decode("utf-8")
        except UnicodeDecodeError:
            continue  # partial UTF-8 byte tokens
        assert tokenizer.decode([i]) == expected


def test_lattice_language():
    lat = WordLattice("hey can you".split())
    ok = [b"hey can you", b"Hey, can you?", b"HEY can YOU.", b"hey. Can you"]
    bad = [b"hey can", b"hey  can you", b"hey can you!", b"Hey,, can you", b"hey can your", b"HeY can you",
           b" hey can you", b"hey can you "]
    for s in ok:
        st = lat.advance(lat.start(), s)
        assert st and lat.is_final(st), s
    for s in bad:
        st = lat.advance(lat.start(), s)
        assert not (st and lat.is_final(st)), s


@pytest.mark.parametrize("prefix", [b"", b"Hey", b"Hey,", b"hey can you", b"HEY C"])
def test_allowed_tokens_match_brute_force(index, prefix):
    """The trie search returns exactly the tokens a full vocabulary scan accepts."""
    lat = WordLattice("hey can you".split())
    st = lat.advance(lat.start(), prefix)
    assert st
    fast = set(lat.allowed_tokens(st, index))
    slow = {i for i, b in index.token_bytes.items() if lat.advance(st, b)}
    assert fast == slow
    assert fast


def random_walk(text, index, rng, eos=(151645,)):
    words = text.split()
    lat = WordLattice(words)
    proc = WordSafeLogitsProcessor(index, [lat], prompt_len=1, eos_ids=list(eos))
    ids = torch.tensor([[0]])
    for _ in range(max_output_bytes(words) + 1):
        scores = torch.rand(1, index.vocab_size, generator=torch.Generator().manual_seed(rng.randrange(1 << 30)))
        # Bias towards ending early, to also test that the end is refused while words remain.
        scores[0, list(eos)] += 0.5
        tok = int(proc(ids, scores).argmax())
        ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
        if tok in eos:
            break
    gen = [i for i in ids[0, 1:].tolist() if i not in eos]
    assert ids[0, -1].item() in eos, "generation must end with EOS within the token budget"
    return b"".join(index.token_bytes[i] for i in gen).decode("utf-8")


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_random_walk_never_changes_words(index, text):
    rng = random.Random(hash(text) & 0xFFFF)
    for _ in range(25):
        out = random_walk(text, index, rng)
        assert roundtrip_ok(text, out), (text, out)


def test_random_walk_random_lines(index):
    rng = random.Random(7)
    vocab = ["the", "report", "friday", "john", "3.5", "co-op", "isn't", "a", "i", "naïve", "www.x.org", "2026"]
    for _ in range(60):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 20)))
        out = random_walk(text, index, rng)
        assert roundtrip_ok(text, out), (text, out)


def test_generate_with_random_model(tokenizer):
    """End to end through transformers' generate() with an untrained model."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from trueform.llm import LLMRestorer

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    model = Qwen2ForCausalLM(cfg)
    model.generation_config.eos_token_id = [151645, 151643]
    restorer = LLMRestorer(model, tokenizer, constrained=True)
    outs = restorer.restore(ADVERSARIAL, batch_size=4)
    for text, out in zip(ADVERSARIAL, outs):
        assert roundtrip_ok(text, out), (text, out)
