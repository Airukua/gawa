"""Gaussian Attention-Weighted Aggregation (GAWA) encoder.

Converts a padded batch of character-ID sequences into fixed-dimensional
word embeddings by:

  1. Projecting each character through a learned embedding table.
  2. Augmenting each position with a fixed Gaussian positional encoding.
  3. Fusing both signals through a two-layer MLP.
  4. Pooling character representations with weights that blend a
     closed-form Gaussian prior with a small learnable adjustment.
  5. Projecting the pooled vector to the final output dimension.

Typical usage::

    encoder = GAWAEncoder(vocab_size=64)
    char_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    lengths  = torch.tensor([3, 2])
    eword    = encoder(char_ids, lengths)  # (2, 768)
"""

from __future__ import annotations

__all__ = ["GAWAEncoder"]

from typing import Final

import torch
import torch.nn as nn

from model.gaussian import GaussianPositionalEncoding, GaussianPositionPrior

# ---------------------------------------------------------------------------
# Module defaults — centralised so callers can reference them in configs.
# ---------------------------------------------------------------------------

_DEFAULT_CHAR_EMB_DIM: Final[int] = 64
_DEFAULT_POS_ENC_DIM: Final[int] = 64
_DEFAULT_HIDDEN_DIM: Final[int] = 256
_DEFAULT_OUTPUT_DIM: Final[int] = 768
_DEFAULT_LAMBDA_ADJUST: Final[float] = 0.3


# ---------------------------------------------------------------------------
# GAWAEncoder
# ---------------------------------------------------------------------------


class GAWAEncoder(nn.Module):
    """Gaussian Attention-Weighted Aggregation encoder.

    Produces a single word embedding from a variable-length character sequence
    using a Gaussian-prior weighted pooling scheme with a learnable correction.

    Architecture summary::

        char_ids  ──► Embedding ──────────────────────┐
                                                       ├──► cat ──► FusionMLP ──► WeightedPool ──► OutputProj ──► eword
        positions ──► GaussianPosEnc (fixed) ──────────┘
                                     ▲
              GaussianPositionPrior ─┤ (prior weights)
              AdjustMLP             ─┘ (learnable δ)

    Args:
        vocab_size:     Number of entries in the character vocabulary.
        char_emb_dim:   Dimension of the character embedding table.
        pos_enc_dim:    Number of Gaussian basis functions in the positional
                        encoding (= positional feature dimension).
        hidden_dim:     Width of the fusion MLP's hidden layers.
        output_dim:     Dimension of the output word embedding (``eword``).
        lambda_adjust:  Scale factor applied to the learnable weight
                        adjustment.  Larger values allow the model to deviate
                        more from the Gaussian prior.  Must be non-negative.
        padding_idx:    Vocabulary index treated as padding; its embedding is
                        fixed at zero and excluded from pooling via ``lengths``.

    Raises:
        ValueError: If ``vocab_size`` or any dimension argument is not
                    a positive integer, or if ``lambda_adjust`` is negative.

    Example::

        >>> encoder = GAWAEncoder(vocab_size=64)
        >>> char_ids = torch.zeros(2, 5, dtype=torch.long)
        >>> lengths  = torch.tensor([3, 2])
        >>> encoder(char_ids, lengths).shape
        torch.Size([2, 768])
    """

    def __init__(
        self,
        vocab_size: int,
        char_emb_dim: int = _DEFAULT_CHAR_EMB_DIM,
        pos_enc_dim: int = _DEFAULT_POS_ENC_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        output_dim: int = _DEFAULT_OUTPUT_DIM,
        lambda_adjust: float = _DEFAULT_LAMBDA_ADJUST,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        _validate_positive_int(vocab_size, "vocab_size")
        _validate_positive_int(char_emb_dim, "char_emb_dim")
        _validate_positive_int(pos_enc_dim, "pos_enc_dim")
        _validate_positive_int(hidden_dim, "hidden_dim")
        _validate_positive_int(output_dim, "output_dim")
        if lambda_adjust < 0:
            raise ValueError(
                f"lambda_adjust must be non-negative, got {lambda_adjust}"
            )

        self.output_dim: Final[int] = output_dim
        self.lambda_adjust: Final[float] = lambda_adjust

        # -- Submodules -------------------------------------------------------

        # (1) Character embedding table.
        self.char_emb = nn.Embedding(
            vocab_size, char_emb_dim, padding_idx=padding_idx
        )

        # (2) Fixed Gaussian positional encoding (no trainable parameters).
        self.pos_enc = GaussianPositionalEncoding(dim=pos_enc_dim)

        # (3) Fusion MLP: [char_emb ‖ pos_enc] → hidden representation.
        fusion_in_dim = char_emb_dim + pos_enc_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # (4a) Gaussian positional prior (closed-form, no parameters).
        self.prior = GaussianPositionPrior()

        # (4b) Learnable weight adjustment: [position, word_len] → δ.
        self.adjust_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # (5) Output projection → eword.
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        char_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a padded batch of character sequences into word embeddings.

        Args:
            char_ids: Padded integer token ids with shape ``(batch, max_len)``.
                      Positions beyond ``lengths[b]`` must be set to
                      ``padding_idx`` (0 by default).
            lengths:  True sequence length for each example in the batch,
                      shape ``(batch,)``.  All values must satisfy
                      ``1 ≤ lengths[b] ≤ max_len``.

        Returns:
            Word-embedding matrix of shape ``(batch, output_dim)``.

        Raises:
            ValueError: If tensor ranks or length bounds are violated.
        """
        _validate_encoder_inputs(char_ids, lengths)

        batch, max_len = char_ids.shape
        device = char_ids.device

        # ------------------------------------------------------------------ #
        # Step 1 – Character embeddings                                        #
        # ------------------------------------------------------------------ #
        x_emb = self.char_emb(char_ids)  # (batch, max_len, char_emb_dim)

        # ------------------------------------------------------------------ #
        # Step 2 – Fixed Gaussian positional encoding                          #
        # ------------------------------------------------------------------ #
        # Positions are 1-indexed to match the Gaussian basis (μ_j = j ≥ 1).
        positions = (
            torch.arange(1, max_len + 1, device=device)
            .unsqueeze(0)
            .expand(batch, -1)
        )  # (batch, max_len)
        pos_enc = self.pos_enc(positions)  # (batch, max_len, pos_enc_dim)

        # ------------------------------------------------------------------ #
        # Step 3 – Fusion MLP                                                  #
        # ------------------------------------------------------------------ #
        fused = torch.cat([x_emb, pos_enc], dim=-1)  # (batch, max_len, fusion_in)
        c = self.fusion_mlp(fused)                    # (batch, max_len, hidden_dim)

        # ------------------------------------------------------------------ #
        # Step 4 – Dynamic positional weighting (vectorised over batch)        #
        # ------------------------------------------------------------------ #
        eword = self._weighted_pool(c, lengths, device)  # (batch, hidden_dim)

        # ------------------------------------------------------------------ #
        # Step 5 – Output projection                                           #
        # ------------------------------------------------------------------ #
        return self.output_proj(eword)  # (batch, output_dim)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _weighted_pool(
        self,
        c: torch.Tensor,
        lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute Gaussian-prior-weighted pooling for each sequence.

        The final per-character weight is::

            w_final[i] = w_gauss[i] · (1 + λ · tanh(adjust_mlp([i, n])))
            w_norm      = w_final / Σ w_final          (normalised)

        Args:
            c:       Fused character representations, shape
                     ``(batch, max_len, hidden_dim)``.
            lengths: True lengths per example, shape ``(batch,)``.
            device:  Target device.

        Returns:
            Pooled representations of shape ``(batch, hidden_dim)``.
        """
        eword_list: list[torch.Tensor] = []

        for b in range(c.size(0)):
            n: int = int(lengths[b].item())

            # Closed-form Gaussian prior weights.
            w_gauss: torch.Tensor = self.prior(n, device)  # (n,)

            # Raw position features: [i, n].
            i_idx = torch.arange(1, n + 1, dtype=torch.float32, device=device)
            n_tensor = torch.full((n,), float(n), device=device)
            inp = torch.stack([i_idx, n_tensor], dim=-1)  # (n, 2)

            # Learnable adjustment capped via tanh → δ ∈ (-λ, +λ).
            delta = self.lambda_adjust * torch.tanh(
                self.adjust_mlp(inp).squeeze(-1)
            )  # (n,)

            # Blend prior with adjustment and re-normalise.
            w_final = w_gauss * (1.0 + delta)
            w_norm  = w_final / w_final.sum().clamp(min=1e-8)  # (n,)

            # Weighted sum over the valid character positions.
            c_b   = c[b, :n, :]                             # (n, hidden_dim)
            eword = (w_norm.unsqueeze(-1) * c_b).sum(dim=0) # (hidden_dim,)
            eword_list.append(eword)

        return torch.stack(eword_list, dim=0)  # (batch, hidden_dim)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"output_dim={self.output_dim}, "
            f"lambda_adjust={self.lambda_adjust})"
        )


# ---------------------------------------------------------------------------
# Module-private validation helpers
# ---------------------------------------------------------------------------


def _validate_positive_int(value: object, name: str) -> None:
    """Raise ``ValueError`` if *value* is not a positive ``int``."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        )


def _validate_encoder_inputs(
    char_ids: torch.Tensor,
    lengths: torch.Tensor,
) -> None:
    """Validate shapes and length-bound constraints for :meth:`GAWAEncoder.forward`.

    Raises:
        ValueError: On any shape or value violation.
    """
    if char_ids.ndim != 2:
        raise ValueError(
            "char_ids must be a 2-D tensor (batch, max_len), "
            f"got shape {tuple(char_ids.shape)}"
        )
    if lengths.ndim != 1:
        raise ValueError(
            "lengths must be a 1-D tensor (batch,), "
            f"got shape {tuple(lengths.shape)}"
        )

    batch, max_len = char_ids.shape
    if lengths.size(0) != batch:
        raise ValueError(
            f"lengths.size(0) must equal char_ids.size(0) (batch={batch}), "
            f"got {lengths.size(0)}"
        )

    max_length = int(lengths.max().item())
    if max_length > max_len:
        raise ValueError(
            f"lengths contains a value ({max_length}) that exceeds "
            f"char_ids max_len ({max_len})"
        )
    if int(lengths.min().item()) < 1:
        raise ValueError(
            "All values in lengths must be ≥ 1; "
            f"got min={int(lengths.min().item())}"
        )
