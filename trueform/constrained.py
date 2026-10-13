"""Constrained decoding: the LLM can only spell out the input words.

The allowed outputs for input words w1..wn form a small regular language over
bytes:

    V(w1) M? ( " " V(wi) M? )*  then end of sequence

where V(w) is one of the word's casing variants (lower, Capitalized, UPPER)
and M is one of , . ?. A byte-level NFA tracks where the generated text is in
that language. At each step, a token is allowed only if its bytes keep the
text inside the language, and end-of-sequence is allowed only once every word
has been written. Skipping, inventing or changing a word is therefore never
an option the model can pick, whatever the weights are.

Works with byte-level BPE tokenizers (GPT-2 style: Qwen2.5, SmolLM2, ...).
"""

from __future__ import annotations

import bisect
from functools import lru_cache

import torch
from transformers import LogitsProcessor

from .labels import MARKS, casing_variants

SPACE = 0x20


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class VocabIndex:
    """Byte strings of every regular token, sorted for prefix lookups."""

    def __init__(self, tokenizer):
        dec = _byte_decoder()
        special = set(tokenizer.all_special_ids)
        special |= {int(i) for i in getattr(tokenizer, "added_tokens_decoder", {})}
        self.token_bytes: dict[int, bytes] = {}
        by_bytes: dict[bytes, list[int]] = {}
        for tok, i in tokenizer.get_vocab().items():
            if i in special:
                continue
            if any(ch not in dec for ch in tok):
                raise ValueError(f"not a byte-level BPE tokenizer (token {tok!r})")
            b = bytes(dec[ch] for ch in tok)
            self.token_bytes[i] = b
            by_bytes.setdefault(b, []).append(i)
        self.by_bytes = by_bytes
        self.sorted = sorted(by_bytes)
        self.vocab_size = max(len(tokenizer), max(self.token_bytes) + 1)

    def is_prefix(self, prefix: bytes) -> bool:
        k = bisect.bisect_left(self.sorted, prefix)
        return k < len(self.sorted) and self.sorted[k].startswith(prefix)


class WordLattice:
    """NFA over bytes for the outputs allowed for a list of input words.

    States:
      ("w", i, v, k)  inside word i, variant v, k bytes written (k < len)
      ("e", i)        word i written, a mark, a space or the end may follow
      ("m", i)        word i and its mark written, a space or the end may follow
    """

    def __init__(self, words: list[str], marks=MARKS[1:]):
        self.words = words
        self.variants = [sorted({v.encode() for v in casing_variants(w).values()}) for w in words]
        self.mark_bytes = frozenset(m.encode()[0] for m in marks)
        if any(len(m.encode()) != 1 for m in marks):
            raise ValueError("marks must be single bytes")
        self.n = len(words)

    def start(self) -> frozenset:
        if self.n == 0:
            return frozenset({("m", -1)})
        return self._enter(0)

    def _enter(self, i: int) -> frozenset:
        return frozenset(("w", i, v, 0) for v in range(len(self.variants[i])))

    def is_final(self, states: frozenset) -> bool:
        last = self.n - 1
        return ("e", last) in states or ("m", last) in states

    def next_bytes(self, state) -> list[int]:
        kind = state[0]
        if kind == "w":
            _, i, v, k = state
            return [self.variants[i][v][k]]
        i = state[1]
        out = [SPACE] if i < self.n - 1 else []
        if kind == "e":
            out += sorted(self.mark_bytes)
        return out

    def step(self, states: frozenset, byte: int) -> frozenset:
        out = set()
        for st in states:
            kind = st[0]
            if kind == "w":
                _, i, v, k = st
                var = self.variants[i][v]
                if var[k] == byte:
                    out.add(("e", i) if k + 1 == len(var) else ("w", i, v, k + 1))
            else:
                i = st[1]
                if byte == SPACE and i < self.n - 1:
                    out |= self._enter(i + 1)
                elif kind == "e" and byte in self.mark_bytes:
                    out.add(("m", i))
        return frozenset(out)

    def advance(self, states: frozenset, data: bytes) -> frozenset:
        for b in data:
            states = self.step(states, b)
            if not states:
                break
        return states

    def allowed_tokens(self, states: frozenset, index: VocabIndex) -> list[int]:
        """Every token whose bytes keep the text inside the language."""
        found: list[int] = []
        stack = [(b"", states)]
        while stack:
            prefix, sts = stack.pop()
            candidates = {b for st in sts for b in self.next_bytes(st)}
            for b in candidates:
                p = prefix + bytes([b])
                if not index.is_prefix(p):
                    continue
                nxt = self.step(sts, b)
                if not nxt:
                    continue
                found.extend(index.by_bytes.get(p, ()))
                stack.append((p, nxt))
        return found


class WordSafeLogitsProcessor(LogitsProcessor):
    """transformers LogitsProcessor that masks every token outside the lattice.

    One lattice per batch row. `prompt_len` is the (padded) prompt length, so
    generated tokens start at that column for every row.
    """

    def __init__(self, index: VocabIndex, lattices: list[WordLattice], prompt_len: int, eos_ids: list[int]):
        self.index = index
        self.lattices = lattices
        self.prompt_len = prompt_len
        self.eos_ids = list(eos_ids)
        self.states = [lat.start() for lat in lattices]
        self.consumed = [0] * len(lattices)
        self.done = [False] * len(lattices)

    def _update(self, row: int, generated: torch.Tensor) -> None:
        lat = self.lattices[row]
        for tok in generated[self.consumed[row] :].tolist():
            if self.done[row]:
                break
            if tok in self.eos_ids:
                self.done[row] = True
                continue
            self.states[row] = lat.advance(self.states[row], self.index.token_bytes[tok])
            if not self.states[row]:
                raise RuntimeError("generated text left the allowed language")
        self.consumed[row] = generated.shape[0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for row in range(input_ids.shape[0]):
            self._update(row, input_ids[row, self.prompt_len :])
            if self.done[row]:
                allowed = self.eos_ids
            else:
                lat, sts = self.lattices[row], self.states[row]
                allowed = lat.allowed_tokens(sts, self.index)
                if lat.is_final(sts):
                    allowed = allowed + self.eos_ids
                if not allowed:
                    raise RuntimeError("no token can continue the output")
            mask[row, allowed] = 0.0
        return scores + mask


def max_output_bytes(words: list[str]) -> int:
    """Longest possible output: every byte of every variant, a mark per word,
    and the spaces. Each token holds at least one byte, so generation with
    max_new_tokens = this + 1 always reaches the end of the words."""
    if not words:
        return 0
    longest = sum(max(len(v.encode()) for v in casing_variants(w).values()) for w in words)
    return longest + 2 * len(words) - 1
