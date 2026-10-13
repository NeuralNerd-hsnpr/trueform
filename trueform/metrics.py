"""Metrics shared by every backend.

Predictions are read back from the output text with the same parser used for
the references, so all approaches are scored the same way. If a backend
changed words, its output is aligned to the input with difflib and input
words without a match count as predicted "lower, no mark".
"""

from __future__ import annotations

import difflib

from .labels import parse_token, roundtrip_ok

PUNCT_CLASSES = (",", ".", "?")
CASE_F1_CLASSES = ("cap", "upper")


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn}


def punctuation_scores(ref: list[str], pred: list[str]) -> dict:
    """Per-mark P/R/F1 and the micro-average over marks ("" is the negative class)."""
    out = {}
    tot = [0, 0, 0]
    for m in PUNCT_CLASSES:
        tp = sum(1 for r, p in zip(ref, pred) if r == m and p == m)
        fp = sum(1 for r, p in zip(ref, pred) if r != m and p == m)
        fn = sum(1 for r, p in zip(ref, pred) if r == m and p != m)
        out[m] = prf(tp, fp, fn)
        tot = [tot[0] + tp, tot[1] + fp, tot[2] + fn]
    out["micro"] = prf(*tot)
    return out


def casing_scores(ref: list[str], pred: list[str]) -> dict:
    out = {"accuracy": sum(r == p for r, p in zip(ref, pred)) / len(ref) if ref else 0.0}
    tot = [0, 0, 0]
    for c in CASE_F1_CLASSES:
        tp = sum(1 for r, p in zip(ref, pred) if r == c and p == c)
        fp = sum(1 for r, p in zip(ref, pred) if r != c and p == c)
        fn = sum(1 for r, p in zip(ref, pred) if r == c and p != c)
        out[c] = prf(tp, fp, fn)
        tot = [tot[0] + tp, tot[1] + fp, tot[2] + fn]
    out["micro"] = prf(*tot)
    return out


def align(words: list[str], output: str) -> tuple[list[str], list[str], bool]:
    """Predicted (cases, marks) for each input word, and whether words changed."""
    parsed = [parse_token(t) for t in output.split()]
    parsed = [p for p in parsed if p[0]]
    cores = [p[0].lower() for p in parsed]
    if cores == words:
        return [p[1] for p in parsed], [p[2] for p in parsed], False
    cases = ["lower"] * len(words)
    marks = [""] * len(words)
    sm = difflib.SequenceMatcher(a=words, b=cores, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            cases[a + k] = parsed[b + k][1]
            marks[a + k] = parsed[b + k][2]
    return cases, marks, True


def score(inputs: list[str], refs: list[str], outputs: list[str]) -> tuple[dict, list[dict]]:
    """Score a test set. Returns (metrics, failures) where failures lists every
    line whose words differ from the input or that breaks the strict contract."""
    from .labels import labels_from_target

    ref_c, ref_m, pred_c, pred_m = [], [], [], []
    failures = []
    for i, (src, ref, out) in enumerate(zip(inputs, refs, outputs)):
        words, rc, rm = labels_from_target(ref)
        if words != src.split():
            raise ValueError(f"line {i}: reference words do not match input")
        pc, pm, altered = align(words, out)
        strict = roundtrip_ok(src, out)
        if altered or not strict:
            failures.append({"line": i, "input": src, "output": out, "words_altered": altered,
                             "roundtrip_ok": strict})
        ref_c += rc
        ref_m += rm
        pred_c += pc
        pred_m += pm
    metrics = {
        "lines": len(inputs),
        "words": len(ref_c),
        "punctuation": punctuation_scores(ref_m, pred_m),
        "casing": casing_scores(ref_c, pred_c),
        "lines_words_altered": sum(f["words_altered"] for f in failures),
        "lines_roundtrip_failed": sum(not f["roundtrip_ok"] for f in failures),
    }
    return metrics, failures
