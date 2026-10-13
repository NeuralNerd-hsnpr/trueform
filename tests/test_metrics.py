import pytest

from trueform.metrics import align, casing_scores, prf, punctuation_scores, score


def test_prf_hand_computed():
    # tp=2, fp=1, fn=3 -> p=2/3, r=2/5, f1=2*(2/3)(2/5)/(2/3+2/5)=0.5
    m = prf(2, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.4)
    assert m["f1"] == pytest.approx(0.5)
    assert m["support"] == 5
    assert prf(0, 0, 0)["f1"] == 0.0


def test_punctuation_scores_hand_computed():
    ref = [",", "", ".", "?", "", "."]
    pred = [",", ",", ".", ".", "", ""]
    s = punctuation_scores(ref, pred)
    # comma: tp=1 fp=1 fn=0 ; period: tp=1 fp=1 fn=1 ; question: tp=0 fp=0 fn=1
    assert s[","]["precision"] == pytest.approx(0.5) and s[","]["recall"] == 1.0
    assert s["."]["f1"] == pytest.approx(0.5)
    assert s["?"]["f1"] == 0.0
    # micro: tp=2 fp=2 fn=2 -> 0.5
    assert s["micro"]["precision"] == pytest.approx(0.5)
    assert s["micro"]["recall"] == pytest.approx(0.5)
    assert s["micro"]["f1"] == pytest.approx(0.5)


def test_casing_scores_hand_computed():
    ref = ["cap", "lower", "upper", "cap", "lower"]
    pred = ["cap", "cap", "upper", "lower", "lower"]
    s = casing_scores(ref, pred)
    assert s["accuracy"] == pytest.approx(3 / 5)
    # cap: tp=1 fp=1 fn=1 -> 0.5 ; upper: tp=1 -> 1.0
    assert s["cap"]["f1"] == pytest.approx(0.5)
    assert s["upper"]["f1"] == pytest.approx(1.0)
    # micro: tp=2 fp=1 fn=1 -> 2/3
    assert s["micro"]["f1"] == pytest.approx(2 / 3)


def test_align_exact_and_altered():
    words = "hey can you send it".split()
    c, m, alt = align(words, "Hey, can you send it?")
    assert not alt and c[0] == "cap" and m == [",", "", "", "", "?"]
    # "could" replaces "can": the unmatched word gets lower / no mark.
    c, m, alt = align(words, "Hey, could you send it?")
    assert alt and c[1] == "lower" and m[1] == "" and m[4] == "?"
    # Added quotes do not count as changed words.
    _, _, alt = align(words, '"Hey," can you send it?')
    assert not alt


def test_score_counts_failures():
    inputs = ["hey can you send it", "i think john has it"]
    refs = ["Hey, can you send it?", "I think John has it."]
    outputs = ["Hey, can you send it?", "I think John has got it."]
    metrics, failures = score(inputs, refs, outputs)
    assert metrics["lines_words_altered"] == 1
    assert metrics["lines_roundtrip_failed"] == 1
    assert failures[0]["line"] == 1
    assert metrics["punctuation"]["micro"]["f1"] == pytest.approx(1.0)
