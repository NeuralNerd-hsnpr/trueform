"""Word-level labels and the word-safety contract.

Every backend produces, for each input word, a casing (lower, Capitalized,
UPPER) and a punctuation mark to put after it. The output text is rebuilt
from the original words, so the words themselves can only change case.
"""

from __future__ import annotations

import unicodedata

CASES = ("lower", "cap", "upper")
MARKS = ("", ",", ".", "?")  # index 0 = no mark
MARK_CHARS = frozenset(",.?")

# Marks found in source text are folded into the four classes above.
_MARK_FOLD = {",": ",", ":": ",", ".": ".", ";": ".", "!": ".", "?": "?"}


def casing_variants(word: str) -> dict[str, str]:
    """The allowed spellings of `word`, keyed by casing label.

    A variant is only kept if lowercasing it gives back the lowercased word,
    so characters whose case mapping changes length (e.g. German sharp s)
    never break the round trip.
    """
    low = word.lower()
    out = {"lower": low}
    cap = low[:1].upper() + low[1:]
    if cap.lower() == low:
        out["cap"] = cap
    up = low.upper()
    if up.lower() == low:
        out["upper"] = up
    return out


def apply_case(word: str, case: str) -> str:
    variants = casing_variants(word)
    return variants.get(case, variants["lower"])


def render(words: list[str], cases: list[str], marks: list[str]) -> str:
    """Rebuild text from the input words and per-word labels."""
    if not (len(words) == len(cases) == len(marks)):
        raise ValueError("words, cases and marks must have the same length")
    return " ".join(apply_case(w, c) + m for w, c, m in zip(words, cases, marks))


def case_of(core: str) -> str:
    """Casing label of a word as it appears in reference text."""
    if not any(ch.isalpha() for ch in core):
        return "lower"
    if core == core.lower():
        return "lower"
    if core == core[:1].upper() + core[1:].lower():
        return "cap"
    if core == core.upper():
        return "upper"
    # Mixed case (iPhone, McDonald): nearest of the three classes.
    return "cap" if core[:1].isupper() else "lower"


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or unicodedata.category(ch) == "Mn"


def parse_token(token: str) -> tuple[str, str, str]:
    """Split a whitespace token from reference or model text into
    (core word, casing label, mark).

    Leading and trailing punctuation is removed; internal characters such as
    apostrophes, hyphens and decimal points stay part of the word. The mark is
    the last foldable punctuation character in the trailing part.
    """
    start, end = 0, len(token)
    while start < end and not _is_word_char(token[start]):
        start += 1
    while end > start and not _is_word_char(token[end - 1]):
        end -= 1
    core = token[start:end]
    mark = ""
    for ch in reversed(token[end:]):
        if ch in _MARK_FOLD:
            mark = _MARK_FOLD[ch]
            break
    return core, case_of(core), mark


def words_of(text: str) -> list[str]:
    """Input words: whitespace tokens, lowercased."""
    return [w.lower() for w in text.split()]


def roundtrip_ok(source: str, output: str) -> bool:
    """The word-safety contract.

    Each output token must be one of the source word's casing variants,
    optionally followed by a single mark. Equivalently: lowercase the output,
    strip the inserted marks, and you get the source words back.
    """
    src = source.split()
    out = output.split()
    if len(src) != len(out):
        return False
    for w, t in zip(src, out):
        allowed = set(casing_variants(w).values())
        if t in allowed:
            continue
        if len(t) > 1 and t[-1] in MARK_CHARS and t[:-1] in allowed:
            continue
        return False
    return True


def encode(cases: list[str], marks: list[str]) -> tuple[list[int], list[int]]:
    return [CASES.index(c) for c in cases], [MARKS.index(m) for m in marks]


def decode(case_ids: list[int], mark_ids: list[int]) -> tuple[list[str], list[str]]:
    return [CASES[i] for i in case_ids], [MARKS[i] for i in mark_ids]


def labels_from_target(target: str) -> tuple[list[str], list[str], list[str]]:
    """Input words and labels from a reference line."""
    words, cases, marks = [], [], []
    for tok in target.split():
        core, case, mark = parse_token(tok)
        if not core:
            continue
        words.append(core.lower())
        cases.append(case)
        marks.append(mark)
    return words, cases, marks
