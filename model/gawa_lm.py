"""GAWA: top-level character-level autoencoder combining GAWAEncoder and GAWADecoder.

``GAWAModel`` is the root :class:`~torch.nn.Module` that ties together the
Gaussian Attention-Weighted Aggregation encoder and decoder into a single,
trainable unit.  It is the primary interface for:

- **Training**: supervised reconstruction via teacher-forced decoding.
- **Representation**: extracting ``eword`` embeddings for use in a downstream
  Global Transformer.
- **Inference**: end-to-end word reconstruction from character ids.

Architecture overview::

    char_ids ──► GAWAEncoder ──► eword ──► GAWADecoder ──► logits
                                   │
                                   └──────────────────────► (returned for downstream use)

Typical training usage::

    model  = GAWAModel(vocab_size=len(vocab))
    logits, eword = model(char_ids, lengths, target_ids=targets)
    loss   = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1),
                             ignore_index=CharVocab.PAD)

Typical inference usage::

    words = model.reconstruct(char_ids, lengths, vocab=vocab)
"""

from __future__ import annotations

__all__ = ["GAWAModel"]

from typing import Final, Optional

import torch
import torch.nn as nn

from model.char_vocab import CharVocab
from model.decoder import GAWADecoder
from model.encoder import GAWAEncoder

# ---------------------------------------------------------------------------
# Module defaults
# ---------------------------------------------------------------------------

_DEFAULT_CHAR_EMB_DIM: Final[int] = 64
_DEFAULT_POS_ENC_DIM: Final[int] = 64
_DEFAULT_HIDDEN_DIM: Final[int] = 256
_DEFAULT_EWORD_DIM: Final[int] = 768
_DEFAULT_MAX_WORD_LEN: Final[int] = 32
_DEFAULT_ENCODER_LAMBDA_ADJUST: Final[float] = 0.3
_DEFAULT_DECODER_NUM_LAYERS: Final[int] = 2
_DEFAULT_DECODER_NUM_HEADS: Final[int] = 4


# ---------------------------------------------------------------------------
# GAWAModel
# ---------------------------------------------------------------------------


class GAWAModel(nn.Module):
    """Character-level autoencoder for word-level representation learning.

    Encodes a padded batch of character sequences into dense ``eword``
    embeddings via :class:`GAWAEncoder`, then decodes them back to character
    logits via :class:`GAWADecoder`.

    The encoder and decoder share neither weights nor vocabulary mapping; they
    are independently parameterised and can be used separately after training
    (e.g., :meth:`encode` alone in a larger pipeline).

    Args:
        vocab_size:    Size of the character vocabulary.  Must match the
                       :class:`CharVocab` instance used to tokenise inputs.
        char_emb_dim:  Character embedding dimension (shared between encoder
                       and decoder so the embedding spaces are comparable).
        pos_enc_dim:   Number of Gaussian basis functions in the encoder's
                       positional encoding layer.
        hidden_dim:    Hidden width for both the encoder fusion MLP and the
                       decoder GRU.
        eword_dim:     Dimensionality of the intermediate word embedding.
                       This is the encoder's output and the decoder's input.
        max_word_len:  Maximum number of decoding steps during autoregressive
                       (non-teacher-forced) inference.
        encoder_lambda_adjust: Scale factor for the encoder's learnable
                       weight adjustment. Must be non-negative.
        decoder_num_layers: Number of GRU layers in the decoder.
        decoder_num_heads:  Number of attention heads in the decoder's
                       cross-attention block.

    Raises:
        ValueError: If any dimension argument is not a positive integer.

    Example::

        >>> vocab = CharVocab()
        >>> model = GAWAModel(vocab_size=vocab.vocab_size)
        >>> char_ids = torch.zeros(2, 8, dtype=torch.long)
        >>> lengths  = torch.tensor([5, 3])
        >>> logits, eword = model(char_ids, lengths)
        >>> logits.shape, eword.shape
        (torch.Size([2, 32, 64]), torch.Size([2, 768]))
    """

    def __init__(
        self,
        vocab_size: int,
        char_emb_dim: int = _DEFAULT_CHAR_EMB_DIM,
        pos_enc_dim: int = _DEFAULT_POS_ENC_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        eword_dim: int = _DEFAULT_EWORD_DIM,
        max_word_len: int = _DEFAULT_MAX_WORD_LEN,
        encoder_lambda_adjust: float = _DEFAULT_ENCODER_LAMBDA_ADJUST,
        decoder_num_layers: int = _DEFAULT_DECODER_NUM_LAYERS,
        decoder_num_heads: int = _DEFAULT_DECODER_NUM_HEADS,
    ) -> None:
        super().__init__()

        _validate_positive_int(vocab_size, "vocab_size")
        _validate_positive_int(char_emb_dim, "char_emb_dim")
        _validate_positive_int(pos_enc_dim, "pos_enc_dim")
        _validate_positive_int(hidden_dim, "hidden_dim")
        _validate_positive_int(eword_dim, "eword_dim")
        _validate_positive_int(max_word_len, "max_word_len")
        if encoder_lambda_adjust < 0:
            raise ValueError(
                f"encoder_lambda_adjust must be non-negative, got {encoder_lambda_adjust}"
            )
        _validate_positive_int(decoder_num_layers, "decoder_num_layers")
        _validate_positive_int(decoder_num_heads, "decoder_num_heads")

        self.eword_dim: Final[int] = eword_dim
        self.max_word_len: Final[int] = max_word_len

        self.encoder = GAWAEncoder(
            vocab_size=vocab_size,
            char_emb_dim=char_emb_dim,
            pos_enc_dim=pos_enc_dim,
            hidden_dim=hidden_dim,
            output_dim=eword_dim,
            lambda_adjust=encoder_lambda_adjust,
        )
        self.decoder = GAWADecoder(
            vocab_size=vocab_size,
            eword_dim=eword_dim,
            char_emb_dim=char_emb_dim,
            hidden_dim=hidden_dim,
            max_len=max_word_len,
            num_layers=decoder_num_layers,
            num_heads=decoder_num_heads,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        char_ids: torch.Tensor,
        lengths: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full encode → decode pass; primary entry point for training.

        Args:
            char_ids:        Padded character id tensor, shape
                             ``(batch, max_len)``.
            lengths:         True sequence lengths, shape ``(batch,)``.
            target_ids:      Ground-truth character ids for teacher forcing,
                             shape ``(batch, seq_len)``.  Required when
                             ``teacher_forcing=True``.
            teacher_forcing: Feed ground-truth tokens as decoder inputs at
                             each step when ``True`` (training); feed argmax
                             predictions when ``False`` (inference).

        Returns:
            A ``(logits, eword)`` tuple:

            - ``logits``:  Shape ``(batch, seq_len, vocab_size)``.  Pass to
              :func:`torch.nn.functional.cross_entropy` for reconstruction
              loss.
            - ``eword``:   Shape ``(batch, eword_dim)``.  Word-level
              representation; can be forwarded to a Global Transformer.

        Raises:
            ValueError: Propagated from encoder / decoder input validation.
        """
        eword  = self.encoder(char_ids, lengths)
        logits = self.decoder(eword, target_ids, teacher_forcing)
        return logits, eword

    def encode(
        self,
        char_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode character sequences into word embeddings.

        A thin, explicitly named wrapper around the encoder for use in
        inference pipelines where only the ``eword`` representation is needed
        (e.g., feeding a Global Transformer without decoding).

        Args:
            char_ids: Padded character id tensor, shape ``(batch, max_len)``.
            lengths:  True sequence lengths, shape ``(batch,)``.

        Returns:
            Word embedding tensor of shape ``(batch, eword_dim)``.
        """
        return self.encoder(char_ids, lengths)

    @torch.inference_mode()
    def reconstruct(
        self,
        char_ids: torch.Tensor,
        lengths: torch.Tensor,
        vocab: CharVocab,
    ) -> list[str]:
        """Encode then greedily decode back to word strings.

        Convenience method for qualitative evaluation and debugging.  Runs
        the full encode → decode pipeline under :func:`torch.inference_mode`
        (no gradient tape; faster than ``no_grad`` on recent PyTorch builds).

        Args:
            char_ids: Padded character id tensor, shape ``(batch, max_len)``.
            lengths:  True sequence lengths, shape ``(batch,)``.
            vocab:    :class:`CharVocab` instance used to map predicted ids
                      back to characters.  The same vocab must have been used
                      to produce ``char_ids``.

        Returns:
            A list of ``batch`` decoded strings.  Each string is terminated
            at the first ``<EOS>`` token; ``<PAD>`` and ``<BOS>`` tokens are
            stripped.  Out-of-range indices are decoded as ``"?"``.

        Example::

            >>> words = model.reconstruct(char_ids, lengths, vocab)
            >>> words
            ['hello', 'world']
        """
        if not isinstance(vocab, CharVocab):
            raise TypeError(
                f"vocab must be a CharVocab instance, got {type(vocab).__name__!r}"
            )

        eword    = self.encoder(char_ids, lengths)
        logits   = self.decoder(eword, teacher_forcing=False)
        pred_ids = logits.argmax(dim=-1)  # (batch, max_len)

        return [vocab.decode(ids.tolist()) for ids in pred_ids]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"eword_dim={self.eword_dim}, "
            f"max_word_len={self.max_word_len})"
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
