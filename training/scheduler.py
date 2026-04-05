"""
Learning-rate schedulers with optional warmup.

Supports cosine and linear decay, with an optional linear warmup phase.
"""

from __future__ import annotations
from typing import Optional
import math
import torch


class SchedulerConfig:
    """Configuration container for learning rate scheduler settings."""

    def __init__(
        self,
        name: str = "cosine",
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        self.name = name
        self.warmup_steps = max(int(warmup_steps), 0)
        self.min_lr = float(min_lr)


def _build_cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup and an optional min_lr floor."""
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    max_base_lr = max(base_lrs) if base_lrs else 1.0

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            # Linear warmup to 1.0 over warmup_steps.
            return float(step + 1) / float(max(1, warmup_steps))

        # Cosine decay phase.
        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Apply floor if min_lr > 0.
        if max_base_lr > 0 and min_lr > 0:
            cosine_decay = max(cosine_decay, min_lr / max_base_lr)

        return cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _build_linear_with_warmup(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear decay to zero with linear warmup."""

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        # Linear decay to 0.
        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    config: SchedulerConfig,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create a scheduler instance from the provided config."""
    name = config.name.lower()

    if name in ("none", "off", "disabled", "constant"):
        return None

    if name in ("cosine", "cosine_anneal"):
        # WARNING: No warmup. Use "cosine_warmup" for warmup support.
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
        )

    if name == "cosine_warmup":
        # Warmup + min_lr floor.
        return _build_cosine_with_warmup(
            optimizer,
            total_steps=total_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
        )

    if name == "linear":
        return _build_linear_with_warmup(
            optimizer,
            total_steps=total_steps,
            warmup_steps=config.warmup_steps,
        )

    raise ValueError(
        f"Unknown scheduler: '{config.name}'. "
        f"Supported: cosine, cosine_warmup, linear, none"
    )
