from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add repo root to sys.path for local execution.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.evaluate import evaluate_dataset
from training.data import WordDataset, load_words
from model.char_vocab import CharVocab
import torch


def _load_config_from_checkpoint(checkpoint_path: str) -> dict:
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    cfg = state.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(
            "Checkpoint does not contain an embedded config. "
            "Re-train with config saving enabled."
        )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GAWA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g., checkpoints/gawa_small/best.pt)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to evaluation dataset (one word per line)",
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
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader workers (default from checkpoint config)",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Force pin_memory=True (default from checkpoint config)",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Force pin_memory=False (default from checkpoint config)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Print N sample reconstructions from first N words in eval set",
    )
    args = parser.parse_args()

    if args.pin_memory and args.no_pin_memory:
        raise ValueError("Use only one of --pin-memory or --no-pin-memory")
    pin_memory = None
    if args.pin_memory:
        pin_memory = True
    elif args.no_pin_memory:
        pin_memory = False

    loss = evaluate_dataset(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        sample_count=args.sample_count,
    )

    # Best-effort dataset stats.
    cfg = _load_config_from_checkpoint(args.checkpoint)
    max_word_len = int(cfg["data"]["max_word_len"])
    words = load_words(args.data_path)
    valid = len(WordDataset(words, CharVocab(), max_len=max_word_len))

    print(f"Eval Loss: {loss:.4f} (dataset={len(words)} words, valid={valid})")


if __name__ == "__main__":
    main()
