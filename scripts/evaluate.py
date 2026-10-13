"""Evaluate one run on the test sets with the shared metrics.

    python scripts/evaluate.py --config configs/eval/tagger-electra-small.yaml
    python scripts/evaluate.py --config configs/eval/ci-tagger.yaml --check

Writes results/<run>/metrics.json, predictions_<set>.jsonl, failures_<set>.jsonl
and the resolved config. With --check, nothing is written: the fresh metrics
are compared with the committed metrics.json and the script exits 1 on drift.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trueform.metrics import score  # noqa: E402
from trueform.runinfo import git_commit, hardware, peak_memory_mb, set_seed, sha256_dir, sha256_file  # noqa: E402

DEFAULTS = {"seed": 13, "threads": None, "batch_size": 32, "limit": None, "device": None,
            "price_per_hour_usd": None, "check_tolerance": 0.005}


def load_backend(cfg: dict, device: str):
    if cfg["backend"] == "tagger":
        from trueform.tagger import Tagger

        ckpt = cfg["checkpoint"]
        local = ROOT / ckpt
        tagger = Tagger.load(local if local.exists() else ckpt, device=device, revision=cfg.get("revision"))
        path = local if local.exists() else None
        info = {"checkpoint": ckpt, "base_model": tagger.meta["base_model"],
                "base_revision": tagger.meta["base_revision"], "pretrained": tagger.meta["pretrained"],
                "sha256": sha256_dir(path) if path else None, "hub_revision": cfg.get("revision")}
        return tagger, info, tagger.meta.get("training")
    if cfg["backend"] == "llm":
        from trueform.llm import LLMRestorer

        adapter = cfg.get("adapter")
        local = ROOT / adapter if adapter else None
        if local is not None and local.exists():
            adapter = str(local)
        restorer = LLMRestorer.load(cfg["base"], cfg["revision"], adapter=adapter,
                                    adapter_revision=cfg.get("adapter_revision"),
                                    constrained=cfg["constrained"], device=device)
        info = {"base_model": cfg["base"], "base_revision": cfg["revision"], "adapter": cfg.get("adapter"),
                "adapter_sha256": sha256_dir(local) if local is not None and local.exists() else None,
                "constrained": cfg["constrained"]}
        training = None
        if local is not None and (local / "train_info.json").exists():
            training = json.loads((local / "train_info.json").read_text())
        return restorer, info, training
    raise ValueError(cfg["backend"])


def compare(fresh: dict, committed: dict, tol: float) -> list[str]:
    problems = []
    for name, m in fresh["test_sets"].items():
        old = committed["test_sets"].get(name)
        if old is None:
            problems.append(f"{name}: not in committed metrics")
            continue
        pairs = [("punct micro F1", m["punctuation"]["micro"]["f1"], old["punctuation"]["micro"]["f1"]),
                 ("casing micro F1", m["casing"]["micro"]["f1"], old["casing"]["micro"]["f1"]),
                 ("casing accuracy", m["casing"]["accuracy"], old["casing"]["accuracy"])]
        for label, a, b in pairs:
            if abs(a - b) > tol:
                problems.append(f"{name}: {label} {a:.4f} vs committed {b:.4f}")
        for key in ("lines", "lines_words_altered", "lines_roundtrip_failed"):
            if m[key] != old[key]:
                problems.append(f"{name}: {key} {m[key]} vs committed {old[key]}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", help="override the device in the config")
    ap.add_argument("--check", action="store_true", help="compare with committed metrics instead of writing")
    ap.add_argument("--speed-only", type=int, metavar="N",
                    help="only time the first N in-domain lines on this device and update the speed entry")
    args = ap.parse_args()
    cfg = {**DEFAULTS, **yaml.safe_load(open(args.config))}
    if args.device:
        cfg["device"] = args.device
    set_seed(cfg["seed"])
    if cfg["threads"]:
        torch.set_num_threads(cfg["threads"])
    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results" / cfg["run"]

    restorer, model_info, training = load_backend(cfg, device)
    dev_key = "gpu" if device.startswith("cuda") else "cpu"
    out_path = out_dir / "metrics.json"

    if args.speed_only:
        rows = [json.loads(line) for line in open(ROOT / cfg["test_sets"]["in_domain"])][: args.speed_only]
        restorer.restore([r["input"] for r in rows[:4]], batch_size=cfg["batch_size"])  # warm-up
        t0 = time.perf_counter()
        restorer.restore([r["input"] for r in rows], batch_size=cfg["batch_size"])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        words = sum(len(r["input"].split()) for r in rows)
        entry = {"words_per_second": words / (time.perf_counter() - t0), "lines": len(rows),
                 "batch_size": cfg["batch_size"], "peak_memory": peak_memory_mb(device), "hardware": hardware()}
        metrics = json.loads(out_path.read_text())
        metrics["efficiency"][dev_key] = entry
        out_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"{dev_key}: {entry['words_per_second']:.0f} words/s on {len(rows)} lines")
        return

    results, data_hashes, timing = {}, {}, {"words": 0, "seconds": 0.0}
    predictions = {}
    for name, path in cfg["test_sets"].items():
        rows = [json.loads(line) for line in open(ROOT / path)]
        if cfg["limit"]:
            rows = rows[: cfg["limit"]]
        data_hashes[name] = sha256_file(ROOT / path)
        inputs = [r["input"] for r in rows]
        t0 = time.perf_counter()
        outputs = restorer.restore(inputs, batch_size=cfg["batch_size"])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        metrics, failures = score(inputs, [r["target"] for r in rows], outputs)
        metrics["words_per_second"] = metrics["words"] / elapsed
        timing["words"] += metrics["words"]
        timing["seconds"] += elapsed
        results[name] = metrics
        predictions[name] = (rows, outputs, failures)
        print(f"{name}: punct F1 {metrics['punctuation']['micro']['f1']:.4f}  "
              f"case F1 {metrics['casing']['micro']['f1']:.4f}  altered {metrics['lines_words_altered']}"
              f"/{metrics['lines']}  {metrics['words_per_second']:.0f} words/s", flush=True)

    manifest = ROOT / "data" / "manifest.json"
    if manifest.exists():
        data_hashes["train"] = json.loads(manifest.read_text())["output_sha256"]["train/train.jsonl"]

    efficiency = {dev_key: {
        "words_per_second": timing["words"] / timing["seconds"],
        "batch_size": cfg["batch_size"],
        "peak_memory": peak_memory_mb(device),
        "hardware": hardware(),
    }}
    if training:
        gpu = training.get("hardware", {}).get("gpu")
        hours = training.get("train_seconds", 0) / 3600
        training = {**training, "gpu_hours": round(hours, 3) if gpu else 0.0,
                    "cost_usd": round(hours * cfg["price_per_hour_usd"], 2) if cfg["price_per_hour_usd"] else None}

    metrics = {
        "run": cfg["run"],
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(ROOT),
        "seed": cfg["seed"],
        "backend": cfg["backend"],
        "model": model_info,
        "data_sha256": data_hashes,
        "training": training,
        "efficiency": efficiency,
        "test_sets": results,
    }

    committed_path = out_path
    if args.check:
        committed = json.loads(committed_path.read_text())
        problems = compare(metrics, committed, cfg["check_tolerance"])
        if problems:
            print("DRIFT:\n  " + "\n  ".join(problems))
            sys.exit(1)
        print("metrics match the committed results")
        return

    # Keep speed numbers measured on another device (e.g. a later GPU run).
    if committed_path.exists():
        old = json.loads(committed_path.read_text())
        for dev, entry in old.get("efficiency", {}).items():
            metrics["efficiency"].setdefault(dev, entry)
    out_dir.mkdir(parents=True, exist_ok=True)
    committed_path.write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "device": device}, sort_keys=False))
    for name, (rows, outputs, failures) in predictions.items():
        with open(out_dir / f"predictions_{name}.jsonl", "w") as f:
            for r, o in zip(rows, outputs):
                f.write(json.dumps({"input": r["input"], "target": r["target"], "output": o},
                                   ensure_ascii=False) + "\n")
        with open(out_dir / f"failures_{name}.jsonl", "w") as f:
            for fl in failures:
                f.write(json.dumps(fl, ensure_ascii=False) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
