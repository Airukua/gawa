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

from typing import Final, Optional, Iterator, List, Tuple

import torch
import torch.nn as nn

from model.char_vocab import CharVocab
from model.decoder import GAWADecoder
from model.encoder import GAWAEncoder
from training.checkpoint import load_checkpoint
from training.config import load_config
from training.utils import select_device, set_seed

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
    # Hugging Face helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: str = "best.pt",
        config_path: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> "GAWAModel":
        """Load a pretrained GAWA checkpoint from Hugging Face Hub.

        Example::
            >>> model = GAWAModel.from_pretrained("AiRukua/gawa")
        """
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "huggingface_hub is required for from_pretrained(). "
                "Install with: pip install huggingface_hub"
            ) from exc

        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )

        if config_path:
            cfg = load_config(config_path)
        else:
            cfg = _load_config_from_checkpoint(checkpoint_path)

        if "seed" in cfg:
            set_seed(int(cfg["seed"]))

        device_obj = select_device(device or cfg.get("device", "cpu"))
        model_cfg = cfg["model"]
        vocab = CharVocab()

        model = cls(
            vocab_size=vocab.vocab_size,
            char_emb_dim=int(model_cfg["char_emb_dim"]),
            pos_enc_dim=int(model_cfg["pos_enc_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            eword_dim=int(model_cfg["eword_dim"]),
            max_word_len=int(model_cfg["max_word_len"]),
            encoder_lambda_adjust=float(model_cfg["encoder_lambda_adjust"]),
            decoder_num_layers=int(model_cfg["decoder_num_layers"]),
            decoder_num_heads=int(model_cfg["decoder_num_heads"]),
        ).to(device_obj)

        load_checkpoint(
            checkpoint_path,
            model,
            optimizer=None,
            scheduler=None,
            map_location=device_obj,
        )

        model.eval()
        # Store helpers for encode/decode convenience methods
        model._gawa_cfg = cfg
        model._gawa_vocab = vocab
        model._gawa_device = device_obj
        model._checkpoint_path = checkpoint_path
        model._config_path = config_path
        return model

    @torch.inference_mode()
    def encode_words(
        self,
        words: List[str],
        batch_size: int | None = None,
    ) -> Tuple[List[str], torch.Tensor]:
        """Encode a list of words into eword embeddings.

        Uses the same logic as `eval.encode.encode_words`.
        Returns (kept_words, embeddings) where embeddings is a CPU tensor.
        """
        if not hasattr(self, "_checkpoint_path"):
            raise ValueError("encode_words requires a model loaded via from_pretrained().")
        from eval.encode import encode_words as _encode_words

        device_str = str(getattr(self, "_gawa_device", "cpu"))
        kept_words, embeddings = _encode_words(
            checkpoint_path=self._checkpoint_path,
            config_path=getattr(self, "_config_path", None),
            words=words,
            batch_size=batch_size,
            device=device_str,
        )
        return kept_words, torch.from_numpy(embeddings)

    @torch.inference_mode()
    def decode_words(
        self,
        words: List[str],
        batch_size: int | None = None,
    ) -> Tuple[List[str], List[str]]:
        """Decode / reconstruct a list of words using `eval.decode.decode_words`."""
        if not hasattr(self, "_checkpoint_path"):
            raise ValueError("decode_words requires a model loaded via from_pretrained().")
        from eval.decode import decode_words as _decode_words

        device_str = str(getattr(self, "_gawa_device", "cpu"))
        return _decode_words(
            checkpoint_path=self._checkpoint_path,
            config_path=getattr(self, "_config_path", None),
            words=words,
            batch_size=batch_size,
            device=device_str,
        )

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


def _load_config_from_checkpoint(checkpoint_path: str) -> dict:
    """Extract training configuration from checkpoint metadata."""
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    cfg = state.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(
            "Checkpoint does not contain an embedded config. "
            "Provide config_path or use a checkpoint saved with config."
        )
    return cfg


def _iter_word_batches(
    words: List[str],
    vocab: CharVocab,
    max_len: int,
    batch_size: int,
) -> Iterator[Tuple[List[str], torch.Tensor, torch.Tensor]]:
    """Batch words with padding and length tracking."""
    batch_words: List[str] = []
    batch_ids: List[List[int]] = []
    batch_lengths: List[int] = []

    for word in words:
        word = word.strip()
        if not word:
            continue
        if len(word) < 1 or len(word) > max_len - 2:
            continue

        ids = vocab.encode(word)
        length = len(ids)
        pad_len = max_len - length
        batch_words.append(word)
        batch_ids.append(ids + [vocab.PAD] * pad_len)
        batch_lengths.append(length)

        if len(batch_words) >= batch_size:
            yield (
                batch_words,
                torch.tensor(batch_ids, dtype=torch.long),
                torch.tensor(batch_lengths, dtype=torch.long),
            )
            batch_words, batch_ids, batch_lengths = [], [], []

    if batch_words:
        yield (
            batch_words,
            torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_lengths, dtype=torch.long),
        )
