"""General utilities for GAWA training."""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility and device selection
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Integer seed value.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_str: str) -> torch.device:
    """Resolve a device string into a valid :class:`torch.device`.

    Falls back to CPU when CUDA is requested but unavailable.

    Args:
        device_str: A device string such as ``"cpu"`` or ``"cuda"``.

    Returns:
        A resolved :class:`torch.device` instance.

    """
    device_str = str(device_str)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Metrics and logging
# ---------------------------------------------------------------------------


def accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
) -> float:
    """Compute character-level accuracy with padding masked out.

    Args:
        logits: Logits of shape ``(batch, seq_len, vocab_size)``.
        targets: Target ids of shape ``(batch, seq_len)``.
        pad_idx: Padding index to ignore.

    Returns:
        Accuracy in the range ``[0, 1]``.

    """
    preds = logits.argmax(dim=-1)
    mask = targets.ne(pad_idx)
    if mask.sum().item() == 0:
        return 0.0
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item()


# ---------------------------------------------------------------------------
# Optional integrations
# ---------------------------------------------------------------------------


def maybe_init_wandb(cfg: dict[str, Any]) -> Optional[Any]:
    """Initialize a Weights & Biases run if enabled.

    Args:
        cfg: The ``wandb`` section of the training config.

    Returns:
        A W&B run object if enabled, otherwise ``None``.
    """
    if not cfg.get("enabled", False):
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "wandb is enabled in config but not installed. "
            "Install with `pip install wandb` or disable wandb.enabled."
        ) from exc

    return wandb.init(
        project=cfg.get("project"),
        entity=cfg.get("entity"),
        name=cfg.get("run_name"),
        config=cfg,
    )
