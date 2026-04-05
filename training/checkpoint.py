"""Checkpoint utilities for saving and restoring GAWA training state.

Provides a safe, atomic save/load cycle that guards against partial writes,
unknown keys, and silent state mismatches.
"""

from __future__ import annotations

__all__ = ["CheckpointState", "save_checkpoint", "load_checkpoint"]

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Optional, Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Versioning — bump when the checkpoint schema changes to detect stale files.
# ---------------------------------------------------------------------------

_CHECKPOINT_VERSION: Final[int] = 1

# Keys that must be present in every valid checkpoint file.
_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {"version", "model", "optimizer", "epoch", "step"}
)


# ---------------------------------------------------------------------------
# CheckpointState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointState:
    """Immutable metadata returned after loading a checkpoint.

    Attributes:
        epoch:       The epoch index at which the checkpoint was saved
                     (1-based by convention).
        step:        The global gradient-update step counter.
        best_metric: Best validation metric recorded so far, or ``None`` if
                     not tracked.  Lower is assumed better (e.g., val loss).
        version:     Schema version of the checkpoint file, used to detect
                     incompatible legacy checkpoints.
        config:      Training configuration dict embedded at save time, or
                     ``None`` if not provided.
    """

    epoch: int
    step: int
    best_metric: Optional[float] = None
    version: int = _CHECKPOINT_VERSION
    config: Optional[dict[str, Any]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {self.epoch}")
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")

    def __str__(self) -> str:
        metric = (
            f"{self.best_metric:.6f}"
            if self.best_metric is not None
            else "n/a"
        )
        return (
            f"CheckpointState(epoch={self.epoch}, step={self.step}, "
            f"best_metric={metric}, version={self.version})"
        )


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    best_metric: Optional[float] = None,
    config: Optional[dict[str, Any]] = None,
) -> Path:
    """Atomically save a training checkpoint to *path*.

    The file is written to a sibling temporary file first and then renamed
    into place.  This guarantees that a checkpoint at *path* is either
    complete and valid or absent — a partially written file is never visible
    to :func:`load_checkpoint`.

    Args:
        path:         Destination file path.  Parent directories are created
                      automatically.
        model:        Model whose ``state_dict`` will be saved.
        optimizer:    Optimizer whose ``state_dict`` will be saved.
        epoch:        Current epoch index (1-based recommended).
        step:         Global gradient-update step counter.
        scheduler:    Optional LR scheduler; its state is saved when provided.
        best_metric:  Best validation metric so far.  ``None`` if not tracked.
        config:       Arbitrary config dict embedded for reproducibility.
                      Must be ``torch.save``-serialisable (plain Python types
                      work best).

    Returns:
        The resolved :class:`~pathlib.Path` where the checkpoint was written.

    Raises:
        ValueError:   If ``epoch`` or ``step`` is negative.
        OSError:      If the destination directory cannot be created or the
                      file cannot be written.
    """
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "version":      _CHECKPOINT_VERSION,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "epoch":        epoch,
        "step":         step,
        "best_metric":  best_metric,
        "config":       config,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    # Write atomically: temp file → fsync → rename.
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=dest.parent, prefix=".ckpt_tmp_", suffix=".pt"
    )
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, dest)  # atomic on POSIX; best-effort on Windows
    except BaseException:
        # Clean up the orphaned temp file on any failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    log.info("Checkpoint saved → %s  (epoch=%d, step=%d)", dest, epoch, step)
    return dest


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> CheckpointState:
    """Load a checkpoint from *path* and restore module states in-place.

    Args:
        path:          Source checkpoint file.
        model:         Model to restore.  Weights are loaded with
                       ``strict=strict``; see below.
        optimizer:     If provided, its state is restored from the checkpoint.
                       Silently skipped when the checkpoint contains no
                       optimizer state (e.g., inference-only checkpoints).
        scheduler:     If provided, its state is restored when present in the
                       checkpoint.
        map_location:  Device onto which tensors are mapped during loading.
                       Pass ``"cuda"`` to load directly onto GPU.
        strict:        Forwarded to :meth:`~torch.nn.Module.load_state_dict`.
                       ``True`` (default) raises on missing or unexpected keys.
                       Set to ``False`` for partial checkpoint loading or
                       fine-tuning from a differently structured checkpoint.

    Returns:
        A :class:`CheckpointState` with the metadata embedded at save time.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If the checkpoint is missing required keys or was
                           saved with an incompatible schema version.
        RuntimeError:      Propagated from ``load_state_dict`` when ``strict``
                           is ``True`` and keys do not match.
    """
    src = Path(path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    state: dict[str, Any] = torch.load(src, map_location=map_location, weights_only=True)

    _validate_checkpoint_schema(state, src)

    # Restore model weights.
    missing, unexpected = model.load_state_dict(
        state["model"], strict=strict
    )
    if missing:
        log.warning("Missing keys in model state_dict: %s", missing)
    if unexpected:
        log.warning("Unexpected keys in model state_dict: %s", unexpected)

    # Restore optimizer state (optional).
    if optimizer is not None:
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        else:
            log.warning(
                "Optimizer provided but checkpoint contains no optimizer state; "
                "optimizer state was NOT restored."
            )

    # Restore scheduler state (optional).
    if scheduler is not None:
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        else:
            log.warning(
                "Scheduler provided but checkpoint contains no scheduler state; "
                "scheduler state was NOT restored."
            )

    ckpt_state = CheckpointState(
        epoch=int(state.get("epoch", 0)),
        step=int(state.get("step", 0)),
        best_metric=state.get("best_metric"),
        version=int(state.get("version", _CHECKPOINT_VERSION)),
        config=state.get("config"),
    )

    log.info(
        "Checkpoint loaded ← %s  (%s)",
        src,
        ckpt_state,
    )
    return ckpt_state


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _validate_checkpoint_schema(
    state: dict[str, Any],
    path: Path,
) -> None:
    """Raise ``ValueError`` if *state* is missing required keys or has an
    incompatible schema version.

    Args:
        state: Raw dict loaded from a checkpoint file.
        path:  Source path, included in error messages for context.

    Raises:
        ValueError: On missing keys or version mismatch.
    """
    missing_keys = _REQUIRED_KEYS - state.keys()
    if missing_keys:
        raise ValueError(
            f"Checkpoint at {path} is missing required keys: {sorted(missing_keys)}"
        )

    file_version: int = int(state["version"])
    if file_version != _CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch: expected {_CHECKPOINT_VERSION}, "
            f"got {file_version} (from {path}).  "
            "Re-save or write a migration path."
        )
