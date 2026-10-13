"""Token-classification tagger: one casing and one punctuation label per word.

The encoder sees the words as a pre-split sequence. Each word is classified
from its first subword, and the output is rebuilt from the original words, so
the model has no way to change them.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .labels import CASES, MARKS, render

FORMAT_VERSION = 1


class TaggerModel(nn.Module):
    def __init__(self, encoder_config, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_config(encoder_config)
        hidden = encoder_config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.case_head = nn.Linear(hidden, len(CASES))
        self.mark_head = nn.Linear(hidden, len(MARKS))

    def forward(self, input_ids, attention_mask, first_pos):
        """first_pos: [batch, words] index of each word's first subword."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = first_pos.unsqueeze(-1).expand(-1, -1, h.size(-1))
        h = self.dropout(torch.gather(h, 1, idx))
        return self.case_head(h), self.mark_head(h)


def encode_batch(tokenizer, batch_words: list[list[str]], max_length: int):
    """Tokenize pre-split words. Returns tensors plus, per line, which words got
    a first subword (words cut by truncation or empty after normalization don't)."""
    enc = tokenizer(batch_words, is_split_into_words=True, truncation=True, max_length=max_length,
                    padding=True, return_tensors="pt")
    n_words = max(len(w) for w in batch_words)
    first_pos = torch.zeros(len(batch_words), max(n_words, 1), dtype=torch.long)
    has_pos = torch.zeros_like(first_pos, dtype=torch.bool)
    for b in range(len(batch_words)):
        seen = set()
        for t, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid not in seen:
                seen.add(wid)
                first_pos[b, wid] = t
                has_pos[b, wid] = True
    return enc["input_ids"], enc["attention_mask"], first_pos, has_pos


def windows(n: int, size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Split n words into overlapping windows. Returns (start, end, keep_start,
    keep_end): each word is taken from exactly one window, away from its edges."""
    if n <= size:
        return [(0, n, 0, n)]
    stride = size - overlap
    out = []
    start = 0
    while True:
        end = min(start + size, n)
        keep_start = 0 if start == 0 else start + overlap // 2
        keep_end = n if end == n else end - overlap // 2
        out.append((start, end, keep_start, keep_end))
        if end == n:
            return out
        start += stride


class Tagger:
    """Inference wrapper around a saved tagger checkpoint."""

    def __init__(self, model: TaggerModel, tokenizer, meta: dict, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.meta = meta
        self.device = device
        self.max_length = meta.get("max_length", 256)
        self.window_words = meta.get("window_words", 100)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", revision: str | None = None) -> "Tagger":
        """Load from a local directory or a Hugging Face Hub repo id.
        Missing or unexpected weights raise an error."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer

        path = Path(path)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(path), revision=revision))
        meta = json.loads((path / "trueform.json").read_text())
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format: {meta.get('format_version')}")
        if tuple(meta["cases"]) != CASES or tuple(meta["marks"]) != MARKS:
            raise ValueError("checkpoint label set does not match this version of trueform")
        config = AutoConfig.from_pretrained(path)
        model = TaggerModel(config)
        state = load_file(path / "model.safetensors")
        model.load_state_dict(state, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(path)
        return cls(model, tokenizer, meta, device)

    def save(self, path: str | Path) -> None:
        save_checkpoint(self.model, self.tokenizer, self.meta, path)

    @torch.inference_mode()
    def predict(self, batch_words: list[list[str]]) -> list[tuple[list[str], list[str]]]:
        """Labels for each line. Words the encoder never saw get lower / no mark."""
        jobs = []  # (line, start, keep_start, keep_end, words)
        for i, words in enumerate(batch_words):
            for s, e, ks, ke in windows(len(words), self.window_words, 20):
                jobs.append((i, s, ks, ke, words[s:e]))
        results = [(["lower"] * len(w), [""] * len(w)) for w in batch_words]
        jobs = [j for j in jobs if j[4]]
        for k in range(0, len(jobs), 32):
            chunk = jobs[k : k + 32]
            ids, mask, pos, has = encode_batch(self.tokenizer, [j[4] for j in chunk], self.max_length)
            case_logits, mark_logits = self.model(ids.to(self.device), mask.to(self.device), pos.to(self.device))
            case_ids = case_logits.argmax(-1).cpu()
            mark_ids = mark_logits.argmax(-1).cpu()
            for b, (i, s, ks, ke, words) in enumerate(chunk):
                cases, marks = results[i]
                for j in range(len(words)):
                    if s + j < ks or s + j >= ke or not has[b, j]:
                        continue
                    cases[s + j] = CASES[case_ids[b, j]]
                    marks[s + j] = MARKS[mark_ids[b, j]]
        return results

    def restore(self, texts: list[str]) -> list[str]:
        batch_words = [t.split() for t in texts]
        out = []
        for words, (cases, marks) in zip(batch_words, self.predict(batch_words)):
            out.append(render(words, cases, marks))
        return out


def save_checkpoint(model: TaggerModel, tokenizer, meta: dict, path: str | Path) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, path / "model.safetensors")
    model.encoder.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    meta = {**meta, "format_version": FORMAT_VERSION, "cases": list(CASES), "marks": list(MARKS)}
    (path / "trueform.json").write_text(json.dumps(meta, indent=2))
