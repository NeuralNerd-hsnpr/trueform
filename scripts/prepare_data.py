"""Download the pinned sources and build train/dev/test pairs.

Each output line is a JSON object {"input": ..., "target": ...}. The input is
the target lowercased with its punctuation stripped; the target is the clean
reference rebuilt from the same words, so it only differs from the input in
casing and the marks , . ? after words.

    python scripts/prepare_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trueform.labels import labels_from_target, render  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(cfg: dict) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths = {}
    wt = cfg["wikitext"]
    for split, name in wt["files"].items():
        p = hf_hub_download(
            wt["repo"], name, repo_type="dataset", revision=wt["revision"], local_dir=RAW / "wikitext"
        )
        paths[f"wikitext/{split}"] = Path(p)
    for book_id in cfg["gutenberg"]["books"]:
        p = RAW / f"pg{book_id}.txt"
        if not p.exists():
            url = cfg["gutenberg"]["url"].format(id=book_id)
            urllib.request.urlretrieve(url, p)
        paths[f"gutenberg/{book_id}"] = p
    return paths


# ---------------------------------------------------------------- cleaning

_WT_FIXES = [
    (re.compile(r" @(.)@ "), r"\1"),  # 1 @,@ 000 -> 1,000 ; well @-@ known
    (re.compile(r"\s*\([^()]*\)"), ""),  # parentheticals: mostly pronunciations and dates
    (re.compile(r'"'), " "),
    (re.compile(r" (n't|'s|'re|'ve|'ll|'d|'m)\b"), r"\1"),
    (re.compile(r" ([,.;:?!%])"), r"\1"),
]


def clean_wikitext(line: str) -> str:
    for pat, rep in _WT_FIXES:
        line = pat.sub(rep, line)
    return " ".join(line.split())


def wikitext_paragraphs(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    paras = []
    for line in pq.read_table(path).column("text").to_pylist():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        paras.append(clean_wikitext(line))
    return paras


_GB_FIXES = [
    (re.compile(r"[’‘]"), "'"),
    (re.compile(r"[“”\"_]"), " "),
    (re.compile(r"--+|[—–]"), " "),
    (re.compile(r"\[[^\]]*\]"), " "),  # [Illustration] etc.
    (re.compile(r"(?<!\w)'|'(?!\w)"), " "),  # quote marks, not apostrophes
]


def gutenberg_paragraphs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    start = re.search(r"^\*\*\* START OF.*$", text, re.M)
    end = re.search(r"^\*\*\* END OF.*$", text, re.M)
    text = text[start.end() : end.start()]
    paras = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())
        if not block or block.upper() == block:  # headings, chapter titles
            continue
        for pat, rep in _GB_FIXES:
            block = pat.sub(rep, block)
        block = " ".join(block.split())
        if len(block.split()) >= 3:
            paras.append(block)
    return paras


# ---------------------------------------------------------------- segments

_SENT_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9])")
_ALLOWED = re.compile(r"^[\x20-\x7eÀ-ſ]+$")  # Latin script only


def segments(paras: list[str], rng: random.Random, min_words: int, max_words: int) -> list[str]:
    """Pack consecutive sentences into lines of a random target length,
    so lines hold one to several sentences like an ASR transcript chunk."""
    out = []
    for para in paras:
        if not _ALLOWED.match(para):
            continue
        sents = _SENT_END.split(para)
        cur: list[str] = []
        target = rng.randint(min_words, max_words)
        for s in sents:
            n = len(s.split())
            if n > max_words:
                continue  # skip run-on sentences and lists
            if cur and sum(len(x.split()) for x in cur) + n > target:
                out.append(" ".join(cur))
                cur = []
                target = rng.randint(min_words, max_words)
            cur.append(s)
        if cur:
            out.append(" ".join(cur))
    pairs = []
    for seg in out:
        words, cases, marks = labels_from_target(seg)
        if len(words) < min_words:
            continue
        target = render(words, cases, marks)
        pairs.append({"input": " ".join(words), "target": target})
    return pairs


def sentence_keys(pair: dict) -> list[str]:
    """Normalized sentences of a pair, for the train/test overlap check."""
    keys, cur = [], []
    for tok in pair["target"].split():
        cur.append(tok.lower().rstrip(",.?"))
        if tok.endswith((".", "?")):
            keys.append(" ".join(cur))
            cur = []
    if cur:
        keys.append(" ".join(cur))
    return keys


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--update", action="store_true", help="accept source files whose hashes changed")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    rng = random.Random(cfg["seed"])
    seg = cfg["segments"]
    sizes = cfg["sizes"]

    paths = download(cfg)
    hashes = {k: sha256(p) for k, p in paths.items()}
    pinned_path = OUT / "manifest.json"
    if pinned_path.exists() and not args.update:
        pinned = json.loads(pinned_path.read_text())["source_sha256"]
        changed = [k for k in hashes if pinned.get(k) not in (None, hashes[k])]
        if changed:
            raise SystemExit(f"source files changed since they were pinned: {changed} (use --update to accept)")

    split_pairs = {}
    for split in ("train", "validation", "test"):
        paras = wikitext_paragraphs(paths[f"wikitext/{split}"])
        split_pairs[split] = segments(paras, rng, seg["min_words"], seg["max_words"])
    ood_paras = []
    for book_id in cfg["gutenberg"]["books"]:
        ood_paras += gutenberg_paragraphs(paths[f"gutenberg/{book_id}"])
    ood = segments(ood_paras, rng, seg["min_words"], seg["max_words"])

    train = split_pairs["train"]
    rng.shuffle(train)
    train = train[: sizes["train"]]
    dev = rng.sample(split_pairs["validation"], min(len(split_pairs["validation"]), sizes["dev"] * 2))
    test_in = rng.sample(split_pairs["test"], min(len(split_pairs["test"]), sizes["test_in_domain"] * 2))
    test_out = rng.sample(ood, sizes["test_out_of_domain"] * 2)

    # Overlap check: drop any test or dev line that shares a sentence with training.
    train_sents = {k for p in train for k in sentence_keys(p) if len(k.split()) >= 4}
    train_lines = {p["input"] for p in train}
    report = {}

    def dedup(name: str, rows: list[dict], n: int) -> list[dict]:
        kept, dropped = [], 0
        for r in rows:
            if r["input"] in train_lines or any(k in train_sents for k in sentence_keys(r)):
                dropped += 1
                continue
            kept.append(r)
        kept = kept[:n]
        report[name] = {"candidates": len(rows), "dropped_overlap": dropped, "kept": len(kept)}
        if len(kept) < n:
            raise SystemExit(f"{name}: only {len(kept)} lines left after overlap check")
        return kept

    dev = dedup("dev", dev, sizes["dev"])
    test_in = dedup("test_in_domain", test_in, sizes["test_in_domain"])
    test_out = dedup("test_out_of_domain", test_out, sizes["test_out_of_domain"])

    # Second pass on the final sets: must be zero by construction.
    final_overlap = {
        name: sum(1 for r in rows if r["input"] in train_lines or any(k in train_sents for k in sentence_keys(r)))
        for name, rows in (("dev", dev), ("test_in_domain", test_in), ("test_out_of_domain", test_out))
    }
    assert all(v == 0 for v in final_overlap.values()), final_overlap

    write_jsonl(OUT / "train" / "train.jsonl", train)
    write_jsonl(OUT / "dev.jsonl", dev)
    write_jsonl(OUT / "test_in_domain.jsonl", test_in)
    write_jsonl(OUT / "test_out_of_domain.jsonl", test_out)
    write_jsonl(OUT / "ci_sample.jsonl", test_in[: sizes["ci_sample"]])

    outputs = {
        p: sha256(OUT / p)
        for p in ("train/train.jsonl", "dev.jsonl", "test_in_domain.jsonl", "test_out_of_domain.jsonl", "ci_sample.jsonl")
    }
    manifest = {
        "config": cfg,
        "source_sha256": hashes,
        "output_sha256": outputs,
        "counts": {
            "train": len(train),
            "train_sentences_indexed": len(train_sents),
            "dev": len(dev),
            "test_in_domain": len(test_in),
            "test_out_of_domain": len(test_out),
        },
        "overlap_check": {"method": "exact match on normalized sentences (>=4 words) and whole lines", **report,
                          "final_overlap": final_overlap},
    }
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["overlap_check"], indent=2))


if __name__ == "__main__":
    main()
