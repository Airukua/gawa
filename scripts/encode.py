from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path for local execution.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from eval.encode import encode_words
from training.data import load_words


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode words")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained checkpoint (e.g., checkpoints/gawa_small/best.pt)",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-path",
        type=str,
        help="Path to input words file (one word per line)",
    )
    input_group.add_argument(
        "--words",
        type=str,
        help='Comma-separated words, e.g. "hello,world,python"',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default from checkpoint config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (default from checkpoint config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (.jsonl or .npy). Default: stdout (jsonl)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "npy"],
        default=None,
        help="Output format. Inferred from --output if not specified.",
    )
    args = parser.parse_args()

    if args.data_path:
        words = load_words(args.data_path)
    else:
        words = [w.strip() for w in args.words.split(",") if w.strip()]
        if not words:
            raise ValueError("No valid words provided via --words")

    fmt = args.format
    if fmt is None and args.output:
        fmt = "npy" if args.output.lower().endswith(".npy") else "jsonl"
    if fmt is None:
        fmt = "jsonl"

    kept_words, embeddings = encode_words(
        config_path=None,
        checkpoint_path=args.checkpoint,
        words=words,
        batch_size=args.batch_size,
        device=args.device,
    )

    if fmt == "npy":
        if args.output is None:
            raise ValueError("Output path required for npy format")
        np.save(args.output, embeddings)
    else:
        out_fh = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        for word, emb in zip(kept_words, embeddings):
            out_fh.write(json.dumps({"word": word, "embedding": emb.tolist()}) + "\n")
        if args.output:
            out_fh.close()

    total = len(words)
    kept = len(kept_words)
    if kept < total:
        print(
            f"Encoded {kept}/{total} words (some skipped for length).",
            file=sys.stderr,
        )
    else:
        print(f"Encoded {kept}/{total} words.", file=sys.stderr)


if __name__ == "__main__":
    main()
