from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add repo root to sys.path for local execution.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.prepare import prepare_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw UTF-8 text file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/train.txt",
        help="Output path for word list (default: data/processed/train.txt)",
    )
    parser.add_argument(
        "--lower",
        action="store_true",
        help="Lowercase all tokens",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="Minimum token length (inclusive)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=64,
        help="Maximum token length (inclusive)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate tokens while preserving order",
    )
    parser.add_argument(
        "--allow-redup",
        action="store_true",
        help="Keep reduplicated hyphenated tokens (e.g., go-go)",
    )
    args = parser.parse_args()

    words = prepare_file(
        args.input,
        output_path=args.output,
        lower=args.lower,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        dedupe=args.dedupe,
        allow_redup=args.allow_redup,
    )
    print(f"Wrote {len(words)} words to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
