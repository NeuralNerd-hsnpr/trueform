"""Build the README results table from results/*/metrics.json.

    python scripts/make_table.py          # rewrite the table in README.md
    python scripts/make_table.py --check  # exit 1 if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"

# Display order and labels. Runs without a metrics.json are left out.
RUNS = [
    ("tagger-electra-small", "Tagger, pretrained electra-small"),
    ("tagger-electra-small-scratch", "Tagger, same architecture from random init"),
    ("llm-qwen2.5-0.5b-lora-constrained", "Qwen2.5-0.5B + LoRA, constrained"),
    ("llm-qwen2.5-0.5b-zeroshot-constrained", "Qwen2.5-0.5B zero-shot, constrained"),
    ("llm-qwen2.5-0.5b-zeroshot", "Qwen2.5-0.5B zero-shot, unconstrained"),
    ("llm-qwen2.5-1.5b-zeroshot", "Qwen2.5-1.5B zero-shot, unconstrained"),
]
SETS = [("in_domain", "in"), ("out_of_domain", "out")]


def cell(m: dict | None, fn) -> str:
    return "–" if m is None else fn(m)


def build() -> str:
    head = ("| Approach | Punct F1 in | Punct F1 out | Casing F1 in | Casing F1 out "
            "| Words altered in | Words altered out | CPU words/s |")
    lines = [head, "|" + "---|" * 8]
    for run, label in RUNS:
        path = ROOT / "results" / run / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        ts = m["test_sets"]
        row = [label]
        for key in ("punct", "case", "altered"):
            for s, _ in SETS:
                t = ts.get(s)
                if key == "punct":
                    row.append(cell(t, lambda t: f"{t['punctuation']['micro']['f1']:.3f}"))
                elif key == "case":
                    row.append(cell(t, lambda t: f"{t['casing']['micro']['f1']:.3f}"))
                else:
                    row.append(cell(t, lambda t: f"{t['lines_words_altered']} / {t['lines']}"))
        cpu = m["efficiency"].get("cpu")
        row.append(f"{cpu['words_per_second']:,.0f}" if cpu else "–")
        lines.append("| " + " | ".join(row) + " |")
    footer = ("\nPunctuation F1 is the micro-average over , . ? ; casing F1 is the micro-average over "
              "the Capitalized and UPPER classes. \"Words altered\" counts test lines whose words differ "
              "from the input. Per-mark and per-class scores, hardware and hashes are in "
              "`results/<run>/metrics.json`; every altered line is listed in `results/<run>/failures_*.jsonl`.")
    return "\n".join(lines) + "\n" + footer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    readme = (ROOT / "README.md").read_text()
    a, b = readme.index(START) + len(START), readme.index(END)
    new = readme[:a] + "\n" + build() + "\n" + readme[b:]
    if args.check:
        if new != readme:
            print("README.md results table is out of date; run scripts/make_table.py")
            sys.exit(1)
        return
    (ROOT / "README.md").write_text(new)


if __name__ == "__main__":
    main()
