from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add repo root to sys.path for local execution.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.config import load_config
from training.trainer import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GAWA (wrapper)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., training/configs/gawa_small.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_from_config(cfg)


if __name__ == "__main__":
    main()
