import random

import pytest

from trueform.labels import (
    CASES,
    MARKS,
    casing_variants,
    decode,
    encode,
    labels_from_target,
    parse_token,
    render,
    roundtrip_ok,
)


def test_casing_variants_basic():
    assert casing_variants("friday") == {"lower": "friday", "cap": "Friday", "upper": "FRIDAY"}
    assert casing_variants("don't") == {"lower": "don't", "cap": "Don't", "upper": "DON'T"}
    assert casing_variants("2024")["cap"] == "2024"


def test_casing_variants_drop_length_changing_case():
    # "ß".upper() == "SS", which would not lowercase back to "ß".
    assert "upper" not in casing_variants("straße")


@pytest.mark.parametrize(
    "token,expected",
    [
        ("Friday?", ("Friday", "cap", "?")),
        ('said,"', ("said", "lower", ",")),
        ("(born", ("born", "lower", "")),
        ("U.S.", ("U.S", "upper", ".")),
        ("well-known;", ("well-known", "lower", ".")),
        ("NASA:", ("NASA", "upper", ",")),
        ("wow!", ("wow", "lower", ".")),
        ("iPhone", ("iPhone", "lower", "")),
        ("McDonald's", ("McDonald's", "cap", "")),
        ("I", ("I", "cap", "")),
        ("3.5,", ("3.5", "lower", ",")),
        ("--", ("", "lower", "")),
    ],
)
def test_parse_token(token, expected):
    assert parse_token(token) == expected


def test_labels_from_target():
    words, cases, marks = labels_from_target("Hey, can you send it by Friday? I think John has it.")
    assert words == ["hey", "can", "you", "send", "it", "by", "friday", "i", "think", "john", "has", "it"]
    assert cases[0] == "cap" and cases[6] == "cap" and cases[7] == "cap" and cases[9] == "cap"
    assert marks == [",", "", "", "", "", "", "?", "", "", "", "", "."]


def test_encode_decode_lossless():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(0, 30)
        cases = [rng.choice(CASES) for _ in range(n)]
        marks = [rng.choice(MARKS) for _ in range(n)]
        assert decode(*encode(cases, marks)) == (cases, marks)


def test_render_then_parse_is_lossless():
    rng = random.Random(1)
    vocab = ["hello", "world", "don't", "x", "3.5", "well-known", "nasa", "i", "a"]
    for _ in range(200):
        words = [rng.choice(vocab) for _ in range(rng.randint(1, 15))]
        cases = [rng.choice(CASES) for _ in words]
        marks = [rng.choice(MARKS) for _ in words]
        out = render(words, cases, marks)
        assert roundtrip_ok(" ".join(words), out)
        w2, c2, m2 = labels_from_target(out)
        assert w2 == words and m2 == marks
        # Single letters and numbers collapse cap/upper (or all three); compare rendered text.
        assert render(w2, c2, m2) == out


def test_roundtrip_rejects_changes():
    src = "hey can you send me the report"
    assert roundtrip_ok(src, "Hey, can you send me the report.")
    assert not roundtrip_ok(src, "Hey, could you send me the report.")  # changed
    assert not roundtrip_ok(src, "Hey, can you send me report.")  # dropped
    assert not roundtrip_ok(src, "Hey, can you please send me the report.")  # added
    assert not roundtrip_ok(src, "Hey, can you send me the report!")  # mark outside set
    assert not roundtrip_ok(src, "HeY can you send me the report")  # casing outside set
    assert not roundtrip_ok(src, "Hey,, can you send me the report")  # two marks
