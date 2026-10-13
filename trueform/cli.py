"""Command line: restore text from arguments, a file, or stdin (one line per input).

    trueform "hey can you send me the report by friday"
    cat transcript.txt | trueform --backend llm
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trueform", description=__doc__.split("\n")[0])
    ap.add_argument("text", nargs="*", help="text to restore; reads stdin when omitted")
    ap.add_argument("--backend", choices=("tagger", "llm"), default="tagger")
    ap.add_argument("--model", help="local checkpoint path or Hugging Face Hub id")
    ap.add_argument("--file", help="read lines from this file instead of stdin")
    ap.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    from . import load

    restorer = load(args.backend, args.model, args.device)
    if args.text:
        print(restorer.restore([" ".join(args.text)])[0])
        return 0
    stream = open(args.file, encoding="utf-8") if args.file else sys.stdin
    buf: list[str] = []
    for line in stream:
        buf.append(line.rstrip("\n"))
        if len(buf) >= args.batch_size:
            print("\n".join(restorer.restore(buf)), flush=True)
            buf = []
    if buf:
        print("\n".join(restorer.restore(buf)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
