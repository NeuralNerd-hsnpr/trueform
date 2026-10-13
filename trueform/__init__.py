"""trueform: word-safe punctuation and capitalization restoration.

    >>> from trueform import restore
    >>> restore("hey can you send me the report by friday i think john has it")
    'Hey, can you send me the report by Friday? I think John has it.'
"""

from __future__ import annotations

import os

from .labels import roundtrip_ok

__all__ = ["restore", "roundtrip_ok", "load"]
__version__ = "0.1.0"

# Hub ids of the published checkpoints. Override with a local path via the
# `model` argument or the TRUEFORM_TAGGER / TRUEFORM_LLM_ADAPTER variables.
DEFAULT_TAGGER = "NeuralNerd-hsnpr/trueform-tagger-electra-small"
DEFAULT_LLM_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LLM_BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_LLM_ADAPTER = "NeuralNerd-hsnpr/trueform-qwen2.5-0.5b-lora"

_cache: dict = {}


def load(backend: str = "tagger", model: str | None = None, device: str | None = None):
    """Load (and cache) a backend. Returns an object with .restore(list[str])."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (backend, model, device)
    if key in _cache:
        return _cache[key]
    if backend == "tagger":
        from .tagger import Tagger

        obj = Tagger.load(model or os.environ.get("TRUEFORM_TAGGER", DEFAULT_TAGGER), device=device)
    elif backend == "llm":
        from .llm import LLMRestorer

        adapter = model or os.environ.get("TRUEFORM_LLM_ADAPTER", DEFAULT_LLM_ADAPTER)
        obj = LLMRestorer.load(DEFAULT_LLM_BASE, DEFAULT_LLM_BASE_REVISION, adapter=adapter,
                               constrained=True, device=device)
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'tagger' or 'llm'")
    _cache[key] = obj
    return obj


def restore(text: str | list[str], backend: str = "tagger", model: str | None = None,
            device: str | None = None) -> str | list[str]:
    """Add punctuation and casing. Words are never added, dropped or changed:
    each output word is an input word in lower, Capitalized or UPPER case,
    optionally followed by one of , . ?"""
    single = isinstance(text, str)
    texts = [text] if single else list(text)
    out = load(backend, model, device).restore(texts)
    return out[0] if single else out
