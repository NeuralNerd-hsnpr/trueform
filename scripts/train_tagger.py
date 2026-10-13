"""Train the word tagger.

    python scripts/train_tagger.py --config configs/tagger-electra-small.yaml
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

from trueform.labels import encode, labels_from_target  # noqa: E402
from trueform.metrics import score  # noqa: E402
from trueform.runinfo import hardware, peak_memory_mb, set_seed  # noqa: E402
from trueform.tagger import Tagger, TaggerModel, encode_batch, save_checkpoint  # noqa: E402


def load_examples(path: Path, limit: int | None) -> list[tuple[list[str], list[int], list[int]]]:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            words, cases, marks = labels_from_target(r["target"])
            case_ids, mark_ids = encode(cases, marks)
            rows.append((words, case_ids, mark_ids))
            if limit and len(rows) >= limit:
                break
    return rows


def batches(rows, batch_size: int, rng: random.Random):
    """Shuffle, then sort within chunks of 50 batches by length to cut padding."""
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    chunk = batch_size * 50
    out = []
    for k in range(0, len(idx), chunk):
        part = sorted(idx[k : k + chunk], key=lambda i: len(rows[i][0]))
        out += [part[j : j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(out)
    return out


def evaluate_dev(model, tokenizer, meta, dev_rows, device) -> dict:
    tagger = Tagger(model, tokenizer, meta, device)
    outputs = tagger.restore([r["input"] for r in dev_rows])
    model.train()
    metrics, _ = score([r["input"] for r in dev_rows], [r["target"] for r in dev_rows], outputs)
    return {
        "punct_micro_f1": metrics["punctuation"]["micro"]["f1"],
        "case_micro_f1": metrics["casing"]["micro"]["f1"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tc, mc = cfg["train"], cfg["model"]
    set_seed(cfg["seed"])
    if tc.get("threads"):
        torch.set_num_threads(tc["threads"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(mc["base"], revision=mc["revision"])
    enc_config = AutoConfig.from_pretrained(mc["base"], revision=mc["revision"])
    model = TaggerModel(enc_config, dropout=tc.get("dropout", 0.1))
    if mc["pretrained"]:
        pretrained = AutoModel.from_pretrained(mc["base"], revision=mc["revision"])
        model.encoder.load_state_dict(pretrained.state_dict(), strict=True)
        del pretrained
    model.to(device).train()

    n_steps = tc["steps"]
    rows = load_examples(ROOT / tc["data"], tc["batch_size"] * n_steps)
    dev_rows = [json.loads(l) for l in open(ROOT / tc["dev_data"])][: tc.get("dev_lines", 500)]
    rng = random.Random(cfg["seed"])
    order = batches(rows, tc["batch_size"], rng)[:n_steps]

    no_decay = ("bias", "LayerNorm.weight", "layer_norm")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": tc["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=tc["lr"])
    warmup = int(tc["warmup_ratio"] * n_steps)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, n_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    meta = {"base_model": mc["base"], "base_revision": mc["revision"], "pretrained": mc["pretrained"],
            "max_length": 512, "window_words": 100}
    log = []
    t0 = time.time()
    for step, batch_idx in enumerate(order):
        batch = [rows[i] for i in batch_idx]
        ids, mask, pos, has = encode_batch(tokenizer, [b[0] for b in batch], tc["max_length"])
        case_t = torch.full(pos.shape, -100, dtype=torch.long)
        mark_t = torch.full(pos.shape, -100, dtype=torch.long)
        for b, (_, c, m) in enumerate(batch):
            case_t[b, : len(c)] = torch.tensor(c)
            mark_t[b, : len(m)] = torch.tensor(m)
        case_t[~has] = -100
        mark_t[~has] = -100
        case_logits, mark_logits = model(ids.to(device), mask.to(device), pos.to(device))
        loss = loss_fn(case_logits.flatten(0, 1), case_t.flatten().to(device)) + loss_fn(
            mark_logits.flatten(0, 1), mark_t.flatten().to(device)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % tc["log_every"] == 0 or step + 1 == n_steps:
            entry = {"step": step + 1, "loss": round(loss.item(), 4), "elapsed_s": round(time.time() - t0, 1)}
            if (step + 1) % tc["eval_every"] == 0 or step + 1 == n_steps:
                entry["dev"] = evaluate_dev(model, tokenizer, meta, dev_rows, device)
            log.append(entry)
            print(json.dumps(entry), flush=True)
    train_seconds = time.time() - t0

    out = ROOT / cfg["output"]
    meta["training"] = {
        "steps": n_steps,
        "examples": n_steps * tc["batch_size"],
        "train_seconds": round(train_seconds, 1),
        "peak_memory_mb": peak_memory_mb(device),
        "hardware": hardware(),
        "seed": cfg["seed"],
    }
    save_checkpoint(model, tokenizer, meta, out)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    # Re-load with strict checking as a smoke test of the saved checkpoint.
    Tagger.load(out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
