"""Gaussian Attention-Weighted Aggregation (GAWA) autoregressive decoder.

Decodes a word embedding (``eword``) produced by :class:`GAWAEncoder` back
into a character sequence, step-by-step, using a GRU with cross-attention
over the encoded word representation.

Decoding modes
--------------
Teacher forcing (training)::

    decoder = GAWADecoder(vocab_size=64)
    logits  = decoder(eword, target_ids=target_ids, teacher_forcing=True)
    # logits: (batch, seq_len, vocab_size)

Autoregressive greedy (inference)::

    logits = decoder(eword, teacher_forcing=False)
    ids    = logits.argmax(dim=-1)          # (batch, max_len)
"""

from __future__ import annotations

__all__ = ["GAWADecoder"]

from typing import Final, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module defaults
# ---------------------------------------------------------------------------

_DEFAULT_EWORD_DIM: Final[int] = 768
_DEFAULT_CHAR_EMB_DIM: Final[int] = 64
_DEFAULT_HIDDEN_DIM: Final[int] = 256
_DEFAULT_MAX_LEN: Final[int] = 32
_DEFAULT_NUM_LAYERS: Final[int] = 2
_DEFAULT_NUM_HEADS: Final[int] = 4

_BOS_IDX: Final[int] = 1
_EOS_IDX: Final[int] = 2
_PAD_IDX: Final[int] = 0


# ---------------------------------------------------------------------------
# GAWADecoder
# ---------------------------------------------------------------------------


class GAWADecoder(nn.Module):
    """Autoregressive GRU decoder with cross-attention over a word embedding.

    Given a word-level embedding ``eword`` produced by :class:`GAWAEncoder`,
    this module generates a character sequence token-by-token.  At each step
    ``t`` the decoder:

    1. Embeds the previous character (or ``<BOS>`` at step 0).
    2. Concatenates the character embedding with ``eword`` as a persistent
       context signal and feeds the pair into a multi-layer GRU.
    3. Refines the GRU output with a cross-attention query over the projected
       ``eword`` (key and value), adding a residual connection.
    4. Projects the attended hidden state to vocabulary logits.

    During **training** the next ground-truth token is fed back (teacher
    forcing).  During **inference** the argmax of the current logits is used
    (greedy decoding).  Beam search can be layered on top by calling
    :meth:`decode_step` directly.

    Args:
        vocab_size:    Size of the character vocabulary (including control
                       tokens).
        eword_dim:     Dimensionality of the input word embedding.
        char_emb_dim:  Character embedding dimension.
        hidden_dim:    GRU hidden state width.  Must be divisible by
                       ``num_heads``.
        max_len:       Maximum number of decoding steps used during
                       autoregressive (non-teacher-forced) inference.
        num_layers:    Number of GRU layers.
        num_heads:     Number of attention heads in the cross-attention layer.
        padding_idx:   Vocabulary index for the padding token.  Its embedding
                       row is kept at zero.

    Raises:
        ValueError: If ``hidden_dim`` is not divisible by ``num_heads``, or
                    if any dimension argument is not a positive integer.

    Example::

        >>> decoder   = GAWADecoder(vocab_size=64)
        >>> eword     = torch.randn(2, 768)
        >>> target    = torch.randint(0, 64, (2, 10))
        >>> logits    = decoder(eword, target_ids=target, teacher_forcing=True)
        >>> logits.shape
        torch.Size([2, 10, 64])
    """

    def __init__(
        self,
        vocab_size: int,
        eword_dim: int = _DEFAULT_EWORD_DIM,
        char_emb_dim: int = _DEFAULT_CHAR_EMB_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        max_len: int = _DEFAULT_MAX_LEN,
        num_layers: int = _DEFAULT_NUM_LAYERS,
        num_heads: int = _DEFAULT_NUM_HEADS,
        padding_idx: int = _PAD_IDX,
    ) -> None:
        super().__init__()

        _validate_positive_int(vocab_size, "vocab_size")
        _validate_positive_int(eword_dim, "eword_dim")
        _validate_positive_int(char_emb_dim, "char_emb_dim")
        _validate_positive_int(hidden_dim, "hidden_dim")
        _validate_positive_int(max_len, "max_len")
        _validate_positive_int(num_layers, "num_layers")
        _validate_positive_int(num_heads, "num_heads")

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.vocab_size: Final[int] = vocab_size
        self.max_len: Final[int] = max_len
        self.hidden_dim: Final[int] = hidden_dim
        self.num_layers: Final[int] = num_layers

        # -- Submodules ---------------------------------------------------

        # (1) Project eword → initial GRU hidden state.
        self.eword_proj = nn.Linear(eword_dim, hidden_dim)

        # (2) Character embedding table (decoder inputs).
        self.char_emb = nn.Embedding(
            vocab_size, char_emb_dim, padding_idx=padding_idx
        )

        # (3) Multi-layer GRU.
        #     Input: [char_emb ‖ eword] — fusing context at every step.
        self.gru = nn.GRU(
            input_size=char_emb_dim + eword_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # (4) Cross-attention: query = GRU output, key/value = eword.
        #     Separate linear projections keep eword_dim and hidden_dim
        #     decoupled (they need not be equal).
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.eword_to_key = nn.Linear(eword_dim, hidden_dim)
        self.eword_to_val = nn.Linear(eword_dim, hidden_dim)

        # (5) Output projection → vocabulary logits.
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        eword: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
    ) -> torch.Tensor:
        """Decode ``eword`` into a sequence of vocabulary logits.

        Args:
            eword:           Word embedding tensor of shape
                             ``(batch, eword_dim)``.
            target_ids:      Ground-truth character ids of shape
                             ``(batch, seq_len)``.  Required when
                             ``teacher_forcing=True``; may be ``None`` for
                             pure autoregressive inference.
            teacher_forcing: If ``True``, feed ground-truth tokens as the
                             next input at each step (training mode).
                             If ``False``, feed the argmax prediction
                             (inference mode).

        Returns:
            Logit tensor of shape ``(batch, max_steps, vocab_size)``, where
            ``max_steps = target_ids.shape[1]`` when ``target_ids`` is
            provided, else ``self.max_len``.

        Raises:
            ValueError: If ``teacher_forcing=True`` but ``target_ids`` is
                        ``None``, or if tensor shapes are inconsistent.
        """
        _validate_decoder_inputs(eword, target_ids, teacher_forcing)

        batch, device = eword.size(0), eword.device
        max_steps: int = (
            target_ids.size(1) if target_ids is not None else self.max_len
        )

        # Pre-compute context projections (constant across all time steps).
        hidden = self._init_hidden(eword)           # (num_layers, batch, hidden)
        eword_k = self.eword_to_key(eword).unsqueeze(1)  # (batch, 1, hidden)
        eword_v = self.eword_to_val(eword).unsqueeze(1)  # (batch, 1, hidden)

        # Decoding loop — one character per step.
        input_char = torch.full(
            (batch,), _BOS_IDX, dtype=torch.long, device=device
        )
        all_logits: list[torch.Tensor] = []

        for t in range(max_steps):
            logits, hidden = self.decode_step(
                input_char, eword, eword_k, eword_v, hidden
            )
            all_logits.append(logits)

            # Determine next input token.
            if teacher_forcing and target_ids is not None:
                input_char = target_ids[:, t]
            else:
                input_char = logits.argmax(dim=-1)  # greedy

        return torch.stack(all_logits, dim=1)  # (batch, max_steps, vocab_size)

    def decode_step(
        self,
        input_char: torch.Tensor,
        eword: torch.Tensor,
        eword_k: torch.Tensor,
        eword_v: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute a single autoregressive decoding step.

        Exposed as a public method to support custom decoding strategies such
        as beam search or nucleus sampling without duplicating GRU/attention
        logic.

        Args:
            input_char: Token ids for the current step, shape ``(batch,)``.
            eword:      Word embedding, shape ``(batch, eword_dim)``.
                        Concatenated with the character embedding as a
                        persistent context signal.
            eword_k:    Pre-projected key tensor, shape ``(batch, 1, hidden_dim)``.
            eword_v:    Pre-projected value tensor, shape ``(batch, 1, hidden_dim)``.
            hidden:     Current GRU hidden state,
                        shape ``(num_layers, batch, hidden_dim)``.

        Returns:
            A ``(logits, hidden)`` tuple where ``logits`` has shape
            ``(batch, vocab_size)`` and ``hidden`` is the updated GRU state.
        """
        # Character embedding concatenated with eword context.
        char_emb = self.char_emb(input_char)                       # (batch, char_emb_dim)
        gru_in   = torch.cat([char_emb, eword], dim=-1).unsqueeze(1)  # (batch, 1, gru_in)

        # GRU step.
        gru_out, hidden = self.gru(gru_in, hidden)  # (batch, 1, hidden)

        # Cross-attention over the eword context (single key/value vector).
        attn_out, _ = self.cross_attn(
            query=gru_out,
            key=eword_k,
            value=eword_v,
        )  # (batch, 1, hidden)

        # Residual connection (mirrors the reference implementation).
        out = (gru_out + attn_out).squeeze(1)  # (batch, hidden)

        logits = self.output_proj(out)  # (batch, vocab_size)
        return logits, hidden

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_hidden(self, eword: torch.Tensor) -> torch.Tensor:
        """Project ``eword`` into the initial GRU hidden state.

        All layers share the same initialisation; this is equivalent to
        broadcasting a single projected vector across all GRU layers, which
        is common practice and avoids adding ``num_layers`` separate
        projection heads.

        Args:
            eword: Shape ``(batch, eword_dim)``.

        Returns:
            Shape ``(num_layers, batch, hidden_dim)``.
        """
        h0 = self.eword_proj(eword)          # (batch, hidden)
        h0 = F.tanh(h0).unsqueeze(0)         # (1, batch, hidden) — bound init range
        return h0.expand(self.num_layers, -1, -1).contiguous()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"vocab_size={self.vocab_size}, "
            f"hidden_dim={self.hidden_dim}, "
            f"max_len={self.max_len}, "
            f"num_layers={self.num_layers})"
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


def _validate_decoder_inputs(
    eword: torch.Tensor,
    target_ids: Optional[torch.Tensor],
    teacher_forcing: bool,
) -> None:
    """Validate shapes and mode consistency for :meth:`GAWADecoder.forward`.

    Raises:
        ValueError: On any shape, rank, or mode violation.
    """
    if eword.ndim != 2:
        raise ValueError(
            f"eword must be a 2-D tensor (batch, eword_dim), "
            f"got shape {tuple(eword.shape)}"
        )
    if teacher_forcing and target_ids is None:
        raise ValueError(
            "target_ids must be provided when teacher_forcing=True"
        )
    if target_ids is not None:
        if target_ids.ndim != 2:
            raise ValueError(
                f"target_ids must be a 2-D tensor (batch, seq_len), "
                f"got shape {tuple(target_ids.shape)}"
            )
        if target_ids.size(0) != eword.size(0):
            raise ValueError(
                f"Batch size mismatch: eword has batch={eword.size(0)}, "
                f"target_ids has batch={target_ids.size(0)}"
            )
