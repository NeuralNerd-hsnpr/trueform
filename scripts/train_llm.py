"""LoRA fine-tune of a small chat model to restore punctuation and casing.

Sized for one free Colab T4 (16 GB): fp16 autocast, gradient checkpointing,
loss on the answer tokens only.

    python scripts/train_llm.py --config configs/llm-qwen2.5-0.5b-lora.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trueform.llm import prompt_text  # noqa: E402
from trueform.runinfo import hardware, peak_memory_mb, set_seed  # noqa: E402


def build_example(tokenizer, row: dict, max_length: int, end_id: int):
    prompt = tokenizer(prompt_text(tokenizer, row["input"]), add_special_tokens=False)["input_ids"]
    answer = tokenizer(row["target"], add_special_tokens=False)["input_ids"] + [end_id]
    ids = (prompt + answer)[:max_length]
    labels = ([-100] * len(prompt) + answer)[:max_length]
    return ids, labels


def collate(batch, pad_id: int):
    n = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        mask[i, : len(ids)] = 1
    return input_ids, mask, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, help="override steps (smoke tests)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mc, tc, lc = cfg["model"], cfg["train"], cfg["lora"]
    if args.max_steps:
        tc["steps"] = args.max_steps
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(mc["base"], revision=mc["revision"])
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(mc["base"], revision=mc["revision"], dtype=torch.float32)
    if tc.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    model.to(device).train()
    model.print_trainable_parameters()

    steps, bs, accum = tc["steps"], tc["batch_size"], tc.get("grad_accum", 1)
    rows = []
    with open(ROOT / tc["data"]) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= steps * bs * accum:
                break
    rng = random.Random(cfg["seed"])
    rng.shuffle(rows)
    examples = [build_example(tokenizer, r, tc["max_length"], end_id) for r in rows]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc["lr"],
                            weight_decay=tc["weight_decay"])
    warmup = max(1, int(tc["warmup_ratio"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    use_amp = device == "cuda" and tc.get("fp16", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log, t0 = [], time.time()
    for step in range(steps):
        for a in range(accum):
            k = (step * accum + a) * bs
            input_ids, mask, labels = collate(examples[k : k + bs], tokenizer.pad_token_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                             labels=labels.to(device)).loss / accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % tc["log_every"] == 0 or step + 1 == steps:
            entry = {"step": step + 1, "loss": round(loss.item() * accum, 4), "elapsed_s": round(time.time() - t0, 1)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    train_seconds = time.time() - t0

    out = ROOT / cfg["output"]
    model.save_pretrained(out)
    info = {"base_model": mc["base"], "base_revision": mc["revision"], "steps": steps,
            "examples": steps * bs * accum, "train_seconds": round(train_seconds, 1),
            "peak_memory_mb": peak_memory_mb(device), "hardware": hardware(), "seed": cfg["seed"]}
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"saved adapter to {out}")


if __name__ == "__main__":
    main()
