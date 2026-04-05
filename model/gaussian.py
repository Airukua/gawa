"""Gaussian positional encoding utilities for sequence and word-level models.

This module provides two complementary modules:

  - GaussianPositionalEncoding: A fixed (non-trainable) positional encoding
    that represents each integer position as a vector of Gaussian basis
    function responses. Useful as a drop-in replacement for sinusoidal
    encodings when a compact, smooth basis is preferred.

  - GaussianPositionPrior: A normalized prior over character positions within
    a word, derived from the GAWA paper. Assigns higher importance to early
    characters (which tend to be more discriminative) and can be used to
    weight character-level embeddings.

Typical usage::

    enc = GaussianPositionalEncoding(dim=64)
    positions = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
    encoded = enc(positions)  # (1, seq_len, 64)

    prior = GaussianPositionPrior()
    weights = prior(n=word_len, device=device)  # (word_len,)
"""

from __future__ import annotations

__all__ = ["GaussianPositionalEncoding", "GaussianPositionPrior"]

from typing import Final, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Device = Union[torch.device, str, None]


# ---------------------------------------------------------------------------
# GaussianPositionalEncoding
# ---------------------------------------------------------------------------


class GaussianPositionalEncoding(nn.Module):
    """Fixed Gaussian positional encoding for integer sequence positions.

    Each position ``i`` is projected onto ``dim`` Gaussian basis functions::

        enc[i, j] = exp(-(i - μ_j)² / (2 · σ_j²))

    where ``μ_j = j`` and ``σ_j = √j`` for ``j ∈ {1, …, dim}``.

    The encoding is **deterministic** and contains **no trainable parameters**.
    Basis centers and widths are registered as non-persistent buffers so the
    module moves correctly with ``.to(device)`` and ``.half()``.

    Args:
        dim: Number of Gaussian basis functions (output feature dimension).
            Must be a positive integer.

    Raises:
        ValueError: If ``dim`` is not a positive integer.

    Example::

        >>> enc = GaussianPositionalEncoding(dim=4)
        >>> positions = torch.tensor([[1, 2, 3]])  # (batch=1, seq_len=3)
        >>> enc(positions).shape
        torch.Size([1, 3, 4])
    """

    def __init__(self, dim: int = 64) -> None:
        super().__init__()

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                f"dim must be a positive integer, got {dim!r}"
            )

        self.dim: Final[int] = dim

        j = torch.arange(1, dim + 1, dtype=torch.float32)  # (dim,)
        # Buffers follow the module's device/dtype automatically.
        self.register_buffer("_mu", j, persistent=False)
        self.register_buffer("_sigma", j.sqrt(), persistent=False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian positional encodings for a batch of sequences.

        Args:
            positions: Integer positions with shape ``(batch, seq_len)``.
                Values should be non-negative integers. Floating-point tensors
                are accepted and will be used as-is (no rounding is applied).

        Returns:
            A float tensor of shape ``(batch, seq_len, dim)`` whose values
            lie in the open interval ``(0, 1]``.

        Raises:
            ValueError: If ``positions`` does not have exactly two dimensions.

        Note:
            The denominator is clamped to ``1e-8`` to avoid division by zero
            for the first basis function (``σ_1 = 1``, always safe, but
            custom subclasses may override ``_sigma``).
        """
        if positions.ndim != 2:
            raise ValueError(
                "positions must be a 2-D tensor with shape (batch, seq_len), "
                f"got shape {tuple(positions.shape)}"
            )

        # (batch, seq_len, 1) – broadcast over dim axis
        pos = positions.unsqueeze(-1).float()

        # (1, 1, dim) – broadcast over batch and seq_len axes
        mu: torch.Tensor = self._mu.view(1, 1, -1)      # type: ignore[attr-defined]
        sigma: torch.Tensor = self._sigma.view(1, 1, -1) # type: ignore[attr-defined]

        # Clamp prevents division by zero more precisely than additive ε.
        denominator = 2.0 * sigma.pow(2).clamp(min=1e-8)
        enc = torch.exp(-((pos - mu).pow(2)) / denominator)
        return enc  # (batch, seq_len, dim)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.dim})"


# ---------------------------------------------------------------------------
# GaussianPositionPrior
# ---------------------------------------------------------------------------


class GaussianPositionPrior(nn.Module):
    """Position-importance prior for characters within a word.

    Models the intuition that characters earlier in a word carry more
    discriminative signal, using a decaying-sigma Gaussian-inspired schedule
    from the GAWA paper (Üstün et al., 2024):

    .. math::

        σ_i = d - (d - s_0) \\cdot e^{-r \\cdot i}

        w_i = \\frac{1 / σ_i}{\\sum_k 1 / σ_k}

    where ``i ∈ {1, …, n}`` indexes character positions.

    Args:
        d:  Asymptotic sigma value (long-word limit). Default: ``1.617``.
        s0: Initial sigma at position 1. Default: ``0.5``.
        r:  Decay rate controlling how quickly sigma approaches ``d``.
            Default: ``1.105``.

    All three hyperparameters are stored as plain Python floats (not tensors),
    so the module has **no trainable or buffered state**.

    Raises:
        ValueError: If any of ``d``, ``s0``, or ``r`` are non-positive, or if
            ``s0 >= d`` (which would make sigma non-monotone).

    Example::

        >>> prior = GaussianPositionPrior()
        >>> weights = prior(n=5, device="cpu")
        >>> weights.shape
        torch.Size([5])
        >>> torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    """

    def __init__(
        self,
        d: float = 1.617,
        s0: float = 0.5,
        r: float = 1.105,
    ) -> None:
        super().__init__()

        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        if s0 <= 0:
            raise ValueError(f"s0 must be positive, got {s0}")
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")
        if s0 >= d:
            raise ValueError(
                f"s0 must be strictly less than d for sigma to be "
                f"monotonically increasing, got s0={s0}, d={d}"
            )

        self.d: Final[float] = d
        self.s0: Final[float] = s0
        self.r: Final[float] = r

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, n: int, device: Device = None) -> torch.Tensor:
        """Return normalized position-importance weights of length ``n``.

        Args:
            n: Word length (number of character positions). Must be positive.
            device: Target device for the returned tensor. Accepts a
                :class:`torch.device`, a device string such as ``"cuda:0"``,
                or ``None`` (defaults to CPU).

        Returns:
            A 1-D float tensor of shape ``(n,)`` whose entries sum to ``1.0``.

        Raises:
            ValueError: If ``n`` is not a positive integer.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError(
                f"n must be a positive integer, got {n!r}"
            )

        i = torch.arange(1, n + 1, dtype=torch.float32, device=device)

        # σ_i = d − (d − s₀) · exp(−r · i)
        sigma = self.d - (self.d - self.s0) * torch.exp(-self.r * i)

        # Importance ∝ 1/σ; normalize to a proper probability vector.
        importance = sigma.reciprocal()
        return importance / importance.sum()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"d={self.d}, s0={self.s0}, r={self.r})"
        )
