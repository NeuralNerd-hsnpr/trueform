"""Tagger backend: the round trip holds whatever the weights predict, and
checkpoints load strictly."""

import json
import random
from pathlib import Path

import pytest
import torch

from trueform.labels import roundtrip_ok
from trueform.tagger import Tagger, TaggerModel, save_checkpoint, windows

ELECTRA = "google/electra-small-discriminator"
ELECTRA_REV = "fa8239aadc095e9164941d05878b98afe9b953c3"
ROOT = Path(__file__).resolve().parents[1]

ADVERSARIAL = [
    "hey can you send me the report by friday i think john has it",
    "the meeting is at 3.30 pm on 12/05/2026 and costs $1,200",
    "visit https://example.com/a?b=c&d=e or www.trueform.dev",
    "don't won't it's rock 'n' roll o'neill's",
    "café naïve straße emoji 👍 ​ zero width",
    "UPPER Mixed iPhone",
    "",
    "x",
]


@pytest.fixture(scope="module")
def tagger():
    from transformers import AutoTokenizer, ElectraConfig

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(ELECTRA, revision=ELECTRA_REV)
    cfg = ElectraConfig(vocab_size=tok.vocab_size, embedding_size=32, hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64)
    meta = {"base_model": ELECTRA, "base_revision": ELECTRA_REV, "pretrained": False,
            "max_length": 64, "window_words": 30}
    return Tagger(TaggerModel(cfg), tok, meta)


def test_roundtrip_adversarial(tagger):
    for text, out in zip(ADVERSARIAL, tagger.restore(ADVERSARIAL)):
        assert roundtrip_ok(text, out), (text, out)


def test_roundtrip_random_and_real_lines(tagger):
    rng = random.Random(0)
    real = [json.loads(line)["input"] for line in open(ROOT / "data" / "dev.jsonl")][:200]
    vocab = sorted({w for line in real for w in line.split()})
    randoms = [" ".join(rng.choice(vocab) for _ in range(rng.randint(1, 150))) for _ in range(50)]
    texts = real + randoms
    for text, out in zip(texts, tagger.restore(texts)):
        assert roundtrip_ok(text, out), (text, out)


def test_windows_cover_each_word_once():
    for n in (0, 1, 29, 30, 31, 100, 257):
        taken = []
        for s, e, ks, ke in windows(n, 30, 10):
            assert s <= ks <= ke <= e
            taken += range(ks, ke)
        assert taken == list(range(n))


def test_strict_loading(tagger, tmp_path):
    save_checkpoint(tagger.model, tagger.tokenizer, tagger.meta, tmp_path)
    loaded = Tagger.load(tmp_path)
    for k, v in tagger.model.state_dict().items():
        assert torch.equal(v, loaded.model.state_dict()[k])

    from safetensors.torch import load_file, save_file

    state = load_file(tmp_path / "model.safetensors")
    missing = dict(state)
    missing.pop("case_head.weight")
    save_file(missing, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        Tagger.load(tmp_path)

    extra = dict(state)
    extra["extra_head.weight"] = torch.zeros(1)
    save_file(extra, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        Tagger.load(tmp_path)
