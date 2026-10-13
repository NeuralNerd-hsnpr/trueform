"""LLM backend: a chat model rewrites the text, optionally under the
word-safe constraint from constrained.py.

The same prompt is used for the zero-shot baseline and for LoRA fine-tuning.
"""

from __future__ import annotations

import torch

from .constrained import VocabIndex, WordLattice, WordSafeLogitsProcessor, max_output_bytes
from .labels import roundtrip_ok

SYSTEM = "You restore punctuation and capitalization in transcribed speech."
INSTRUCTION = (
    "Add punctuation and capitalization to the text below. Do not add, remove or change any words. "
    "Reply with the corrected text only.\n\n"
)
CHUNK_WORDS = 80


def messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": INSTRUCTION + text}]


def prompt_text(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)


def eos_ids(tokenizer, model) -> list[int]:
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


class LLMRestorer:
    def __init__(self, model, tokenizer, constrained: bool = True, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.constrained = constrained
        self.device = device
        self.eos = eos_ids(tokenizer, model)
        self.index = VocabIndex(tokenizer) if constrained else None

    @classmethod
    def load(cls, base: str, revision: str | None = None, adapter: str | None = None,
             adapter_revision: str | None = None, constrained: bool = True, device: str = "cpu",
             dtype: torch.dtype | None = None) -> "LLMRestorer":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
        dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=dtype)
        if adapter:
            model = load_adapter_strict(model, adapter, adapter_revision)
        return cls(model, tokenizer, constrained, device)

    @torch.inference_mode()
    def _generate(self, texts: list[str]) -> list[str]:
        prompts = [prompt_text(self.tokenizer, t) for t in texts]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        words = [t.split() for t in texts]
        kwargs = dict(do_sample=False, pad_token_id=self.tokenizer.pad_token_id or self.eos[0],
                      eos_token_id=self.eos)
        if self.constrained:
            kwargs["max_new_tokens"] = max(max_output_bytes(w) for w in words) + 1
            proc = WordSafeLogitsProcessor(self.index, [WordLattice(w) for w in words], prompt_len, self.eos)
            kwargs["logits_processor"] = [proc]
        else:
            n_in = max(len(self.tokenizer(t)["input_ids"]) for t in texts)
            kwargs["max_new_tokens"] = 2 * n_in + 32
        out = self.model.generate(**enc, **kwargs)
        results = []
        for row in range(len(texts)):
            ids = [i for i in out[row, prompt_len:].tolist() if i not in self.eos and i != kwargs["pad_token_id"]]
            if self.constrained:
                text = b"".join(self.index.token_bytes[i] for i in ids).decode("utf-8")
                # Holds by construction; a failure here is a bug, not a filter.
                if not roundtrip_ok(texts[row], text):
                    raise RuntimeError(f"constrained output broke the word contract: {text!r}")
            else:
                text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            results.append(text)
        return results

    def restore(self, texts: list[str], batch_size: int = 8) -> list[str]:
        """Restore each text. Long texts are split into chunks of CHUNK_WORDS words."""
        pieces, owners = [], []
        for i, t in enumerate(texts):
            w = t.split()
            for k in range(0, max(len(w), 1), CHUNK_WORDS):
                pieces.append(" ".join(w[k : k + CHUNK_WORDS]))
                owners.append(i)
        # Group similar lengths to reduce padding.
        order = sorted(range(len(pieces)), key=lambda j: len(pieces[j]))
        restored = [""] * len(pieces)
        for k in range(0, len(order), batch_size):
            idx = order[k : k + batch_size]
            todo = [j for j in idx if pieces[j]]
            if todo:
                for j, r in zip(todo, self._generate([pieces[j] for j in todo])):
                    restored[j] = r
        out = [[] for _ in texts]
        for j, i in enumerate(owners):
            if restored[j]:
                out[i].append(restored[j])
        return [" ".join(parts) for parts in out]


def load_adapter_strict(model, adapter: str, revision: str | None = None):
    """Attach a LoRA adapter and fail if any adapter weight is missing or unexpected."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter, revision=revision)
    from peft.utils import load_peft_weights

    saved = load_peft_weights(adapter, revision=revision) if revision else load_peft_weights(adapter)
    expected = {k for k in model.state_dict() if "lora_" in k}
    # peft stores keys without the adapter name; normalise both sides.
    norm = lambda k: k.replace(".default", "").removeprefix("base_model.model.")  # noqa: E731
    saved_keys = {norm(k) for k in saved}
    expected_keys = {norm(k) for k in expected}
    missing = expected_keys - saved_keys
    unexpected = saved_keys - expected_keys
    if missing or unexpected:
        raise RuntimeError(f"adapter mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    return model.merge_and_unload()
