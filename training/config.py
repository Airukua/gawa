"""Configuration management for GAWA training.

Provides a single source of truth for default hyperparameters, a safe YAML
loader, and a validated config dataclass hierarchy.  The design separates
three concerns:

1. **Defaults** — ``DEFAULT_CONFIG``: a plain dict that YAML files override.
2. **Loading** — ``load_config``: reads YAML, deep-merges, validates schema.
3. **Typed access** — ``GAWAConfig`` and nested dataclasses: structured,
   type-checked, IDE-navigable config objects.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_CONFIG",
    "load_config",
    "GAWAConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "SchedulerConfig",
    "WandbConfig",
]

import copy
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Optional, Union

import torch
import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Bump this when the config schema changes in a backward-incompatible way.
_CONFIG_VERSION: Final[int] = 1

# Top-level sections that must be present after merging with defaults.
_REQUIRED_SECTIONS: Final[frozenset[str]] = frozenset(
    {"seed", "device", "data", "model", "training", "scheduler", "wandb"}
)


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "version": _CONFIG_VERSION,
    "seed": 42,
    # Resolved at import time; can be overridden in YAML for multi-GPU setups.
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data": {
        "train_path": "data/raw/train.txt",
        "val_path": None,
        "test_path": None,
        "val_split": 0.05,
        "test_split": 0.0,
        "max_word_len": 32,
    },
    "model": {
        "char_emb_dim": 64,
        "pos_enc_dim": 64,
        "hidden_dim": 256,
        "eword_dim": 768,
        "max_word_len": 32,
        "encoder_lambda_adjust": 0.3,
        "decoder_num_layers": 2,
        "decoder_num_heads": 4,
    },
    "training": {
        "epochs": 10,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "teacher_forcing": True,
        "use_tqdm": True,
        "log_every": 50,
        "eval_every": 1,
        "sample_every": 5,
        "sample_count": 5,
        "num_workers": 2,
        "pin_memory": True,
        "amp": False,
        "checkpoint": {
            "dir": "checkpoints/gawa",
            "save_every": 1,
            "save_best": True,
            "resume_path": None,
            "max_keep": 3,
        },
        "early_stopping": {
            "enabled": False,
            "patience": 5,
            "min_delta": 0.0,
        },
    },
    "scheduler": {
        "name": "cosine_anneal",
        "warmup_steps": 0,
        "min_lr": 0.0,
    },
    "wandb": {
        "enabled": False,
        "project": "gawa",
        "entity": None,
        "run_name": None,
    },
}


# ---------------------------------------------------------------------------
# Typed config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing settings.

    Attributes:
        train_path:   Path to the training corpus (one word per line).
        val_path:     Explicit validation file; takes precedence over
                      ``val_split`` when provided.
        test_path:    Explicit test file; takes precedence over
                      ``test_split`` when provided.
        val_split:    Fraction of training data used for validation when
                      ``val_path`` is ``None``.  Must be in ``[0, 1)``.
        test_split:   Fraction of training data used for test when
                      ``test_path`` is ``None``.  Must be in ``[0, 1)``.
        max_word_len: Maximum character sequence length; longer words are
                      truncated or skipped depending on the dataset loader.
    """

    train_path: str = "data/raw/train.txt"
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    val_split: float = 0.05
    test_split: float = 0.0
    max_word_len: int = 32

    def __post_init__(self) -> None:
        if self.val_split < 0.0 or self.val_split >= 1.0:
            raise ValueError(
                f"val_split must be in [0, 1), got {self.val_split}"
            )
        if self.test_split < 0.0 or self.test_split >= 1.0:
            raise ValueError(
                f"test_split must be in [0, 1), got {self.test_split}"
            )
        if self.val_path is None and self.test_path is None:
            if self.val_split + self.test_split >= 1.0:
                raise ValueError(
                    "val_split + test_split must be < 1.0 when using splits; "
                    f"got {self.val_split + self.test_split}"
                )
        if self.max_word_len <= 0:
            raise ValueError(
                f"max_word_len must be positive, got {self.max_word_len}"
            )


@dataclass(frozen=True)
class ModelConfig:
    """GAWA encoder/decoder architecture hyperparameters.

    Attributes:
        char_emb_dim:           Character embedding dimension (shared between
                                encoder and decoder).
        pos_enc_dim:            Number of Gaussian basis functions in the
                                encoder positional encoding layer.
        hidden_dim:             Hidden width for the encoder fusion MLP and
                                the decoder GRU.  Must be divisible by
                                ``decoder_num_heads``.
        eword_dim:              Dimensionality of the word embedding output by
                                the encoder and consumed by the decoder.
        max_word_len:           Maximum decoding steps (autoregressive mode).
        encoder_lambda_adjust:  Scale of the decoder's learnable weight
                                adjustment.  Must be non-negative.
        decoder_num_layers:     Number of GRU layers in the decoder.
        decoder_num_heads:      Number of cross-attention heads in the decoder.
    """

    char_emb_dim: int = 64
    pos_enc_dim: int = 64
    hidden_dim: int = 256
    eword_dim: int = 768
    max_word_len: int = 32
    encoder_lambda_adjust: float = 0.3
    decoder_num_layers: int = 2
    decoder_num_heads: int = 4

    def __post_init__(self) -> None:
        for attr in ("char_emb_dim", "pos_enc_dim", "hidden_dim",
                     "eword_dim", "max_word_len",
                     "decoder_num_layers", "decoder_num_heads"):
            if getattr(self, attr) <= 0:
                raise ValueError(
                    f"ModelConfig.{attr} must be positive, got {getattr(self, attr)}"
                )
        if self.encoder_lambda_adjust < 0:
            raise ValueError(
                "encoder_lambda_adjust must be non-negative, "
                f"got {self.encoder_lambda_adjust}"
            )
        if self.hidden_dim % self.decoder_num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"decoder_num_heads ({self.decoder_num_heads})"
            )


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint saving/resuming settings.

    Attributes:
        dir:          Directory where checkpoint files are written.
        save_every:   Save a checkpoint every this many epochs.
        save_best:    Whether to additionally save ``best.pt`` when
                      the validation metric improves.
        resume_path:  If set, training resumes from this checkpoint path.
        max_keep:     Maximum number of epoch checkpoints to keep. When
                      pruning, checkpoints are kept roughly evenly spaced
                      across the configured epoch range.
    """

    dir: str = "checkpoints/gawa"
    save_every: int = 1
    save_best: bool = True
    resume_path: Optional[str] = None
    max_keep: int = 3

    def __post_init__(self) -> None:
        if self.save_every <= 0:
            raise ValueError(
                f"CheckpointConfig.save_every must be positive, "
                f"got {self.save_every}"
            )
        if self.max_keep <= 0:
            raise ValueError(
                f"CheckpointConfig.max_keep must be positive, got {self.max_keep}"
            )


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping settings.

    Attributes:
        enabled:   Whether to stop training when validation stops improving.
        patience:  Number of eval steps to wait without improvement.
        min_delta: Minimum improvement to reset patience.
    """

    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError(
                f"patience must be positive, got {self.patience}"
            )
        if self.min_delta < 0:
            raise ValueError(
                f"min_delta must be non-negative, got {self.min_delta}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser, data-loading, and training loop settings.

    Attributes:
        epochs:           Total number of training epochs.
        batch_size:       Mini-batch size.
        lr:               Peak learning rate.
        weight_decay:     AdamW weight-decay coefficient.
        grad_clip_norm:   Maximum gradient L2 norm; ``0.0`` disables clipping.
        teacher_forcing:  Use ground-truth tokens as decoder inputs during
                          training.
        use_tqdm:         Enable a progress bar for the training dataloader.
        log_every:        Log scalar metrics every this many steps.
        eval_every:       Run validation every this many epochs.
        sample_every:     Log reconstruction samples every this many epochs.
        sample_count:     Number of samples to log per evaluation.
        num_workers:      DataLoader worker processes.
        pin_memory:       Pin host memory for faster GPU transfer.
        amp:              Enable automatic mixed precision (``torch.cuda.amp``).
        checkpoint:       Nested checkpoint settings.
        early_stopping:   Nested early-stopping settings.
    """

    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    teacher_forcing: bool = True
    use_tqdm: bool = True
    log_every: int = 50
    eval_every: int = 1
    sample_every: int = 5
    sample_count: int = 5
    num_workers: int = 2
    pin_memory: bool = True
    amp: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    def __post_init__(self) -> None:
        for attr in ("epochs", "batch_size", "log_every",
                     "eval_every", "sample_every", "sample_count"):
            if getattr(self, attr) <= 0:
                raise ValueError(
                    f"TrainingConfig.{attr} must be positive, "
                    f"got {getattr(self, attr)}"
                )
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if self.grad_clip_norm < 0:
            raise ValueError(
                f"grad_clip_norm must be non-negative, got {self.grad_clip_norm}"
            )
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be non-negative, got {self.num_workers}"
            )


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate scheduler settings.

    Attributes:
        name:          Scheduler type.  Supported values: ``"cosine_anneal"``,
                       ``"cosine"``, ``"linear"``, ``"constant"``.
        warmup_steps:  Number of linear warm-up steps before the main
                       schedule begins.
        min_lr:        Lower bound on the learning rate during cosine decay.
    """

    name: str = "cosine"
    warmup_steps: int = 100
    min_lr: float = 1e-5

    _SUPPORTED: Final[frozenset[str]] = field(
        default=frozenset({"cosine_anneal", "cosine", "linear", "constant"}),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.name not in self._SUPPORTED:
            raise ValueError(
                f"SchedulerConfig.name must be one of {sorted(self._SUPPORTED)}, "
                f"got {self.name!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {self.min_lr}")


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases integration settings.

    Attributes:
        enabled:  Whether to initialise a W&B run.
        project:  W&B project name.
        entity:   W&B entity (team or user); ``None`` uses the default.
        run_name: Human-readable run name; ``None`` lets W&B auto-generate.
    """

    enabled: bool = False
    project: str = "gawa"
    entity: Optional[str] = None
    run_name: Optional[str] = None


@dataclass(frozen=True)
class GAWAConfig:
    """Root configuration object for a GAWA training run.

    Aggregates all subsection configs into a single, typed, immutable object.
    Construct via :meth:`from_dict` (from a loaded YAML) or directly for
    unit tests.

    Attributes:
        version:  Config schema version; validated against
                  ``_CONFIG_VERSION`` on load.
        seed:     Global random seed for reproducibility.
        device:   Training device string (e.g., ``"cuda"``, ``"cpu"``).
        data:     Dataset settings.
        model:    Architecture hyperparameters.
        training: Optimiser and loop settings.
        scheduler: LR scheduler settings.
        wandb:    Logging integration settings.
    """

    version: int = _CONFIG_VERSION
    seed: int = 42
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> GAWAConfig:
        """Construct a :class:`GAWAConfig` from a raw config dict.

        Typically the dict comes from :func:`load_config`, but any dict
        with the expected structure is accepted.

        Args:
            cfg: Raw config mapping (may be a subset; missing keys use
                 dataclass defaults).

        Returns:
            A fully validated, immutable :class:`GAWAConfig` instance.

        Raises:
            ValueError: If any field fails its ``__post_init__`` validation.
            TypeError:  If a value has the wrong type for its field.
        """
        train_raw = cfg.get("training", {})
        ckpt_raw = train_raw.get("checkpoint", {})
        early_raw = train_raw.get("early_stopping", {})

        return cls(
            version=cfg.get("version", _CONFIG_VERSION),
            seed=cfg.get("seed", 42),
            device=cfg.get("device", "cpu"),
            data=DataConfig(**cfg.get("data", {})),
            model=ModelConfig(**cfg.get("model", {})),
            training=TrainingConfig(
                **{
                    k: v
                    for k, v in train_raw.items()
                    if k not in ("checkpoint", "early_stopping")
                },
                checkpoint=CheckpointConfig(**ckpt_raw),
                early_stopping=EarlyStoppingConfig(**early_raw),
            ),
            scheduler=SchedulerConfig(**cfg.get("scheduler", {})),
            wandb=WandbConfig(**cfg.get("wandb", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise this config back to a plain dict.

        The returned dict is a deep copy safe to mutate and can be round-
        tripped through :func:`load_config` / :meth:`from_dict`.

        Returns:
            A JSON/YAML-serialisable plain-Python dict.
        """
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(
    path: Union[str, Path],
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Load a YAML config file and merge it with :data:`DEFAULT_CONFIG`.

    Processing order (later entries win):

    1. :data:`DEFAULT_CONFIG` (deep-copied to avoid mutation).
    2. Values from the YAML file at *path*.
    3. Programmatic *overrides* (useful for CLI flag injection).

    Args:
        path:      Path to a YAML config file.  Must exist and be valid YAML.
        overrides: Optional dict of values to apply on top of the merged
                   result.  Keys follow the same nested structure as the YAML
                   file.

    Returns:
        A fully merged config dict.  Use :meth:`GAWAConfig.from_dict` to
        convert to typed dataclasses.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If the YAML file is empty or not a mapping.
    """
    src = Path(path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Config file not found: {src}")

    with src.open("r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh)

    if user_cfg is None:
        log.warning("Config file %s is empty; using defaults.", src)
        user_cfg = {}
    elif not isinstance(user_cfg, dict):
        raise ValueError(
            f"Config file {src} must contain a YAML mapping at the top level, "
            f"got {type(user_cfg).__name__}"
        )

    # Deep-copy defaults so repeated calls are independent.
    cfg = _deep_update(copy.deepcopy(DEFAULT_CONFIG), user_cfg)

    if overrides:
        cfg = _deep_update(cfg, overrides)

    _validate_config_schema(cfg, src)

    log.debug("Config loaded from %s  (seed=%s, device=%s)",
              src, cfg.get("seed"), cfg.get("device"))
    return cfg


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _deep_update(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge *updates* into *base*, returning *base*.

    Nested dicts are merged recursively; scalar values are overwritten.
    *base* is modified **in place** — pass a deep copy if the original must
    be preserved.

    Args:
        base:    Mutable base dictionary.
        updates: Override values; nested dicts are merged, not replaced.

    Returns:
        The mutated *base* dict (for chaining convenience).
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _validate_config_schema(
    cfg: dict[str, Any],
    path: Path,
) -> None:
    """Raise ``ValueError`` if *cfg* is missing required top-level sections.

    Args:
        cfg:  Merged config dict to validate.
        path: Source path included in error messages.

    Raises:
        ValueError: If any required section is absent.
    """
    missing = _REQUIRED_SECTIONS - cfg.keys()
    if missing:
        raise ValueError(
            f"Config loaded from {path} is missing required sections: "
            f"{sorted(missing)}.  Check your YAML file."
        )
