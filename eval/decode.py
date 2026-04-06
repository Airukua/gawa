"""
Programmatic decoding / reconstruction for the GAWA language model.

Use :func:`decode_words` from Python. This mirrors the batching and
checkpoint-loading logic of :mod:`eval.encode`, but returns reconstructed
strings instead of embeddings.
"""

from __future__ import annotations
from typing import Iterable, Iterator, List, Tuple

import torch

from model.char_vocab import CharVocab
from model.gawa_lm import GAWAModel
from training.checkpoint import load_checkpoint
from training.config import load_config
from training.utils import select_device, set_seed


def _iter_batches(
    words: Iterable[str],
    vocab: CharVocab,
    max_len: int,
    batch_size: int,
) -> Iterator[Tuple[List[str], torch.Tensor, torch.Tensor, int]]:
    """Batch words with padding and length tracking.

    Yields batches of encoded words suitable for model input. Handles
    filtering of words that are too short or too long, and pads sequences
    to max_len using the PAD token.
    """
    batch_words: List[str] = []
    batch_ids: List[List[int]] = []
    batch_lengths: List[int] = []
    skipped = 0

    for word in words:
        word = word.strip()
        if not word:
            continue
        if len(word) < 1 or len(word) > max_len - 2:
            skipped += 1
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
                skipped,
            )
            batch_words, batch_ids, batch_lengths = [], [], []
            skipped = 0

    if batch_words:
        yield (
            batch_words,
            torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_lengths, dtype=torch.long),
            skipped,
        )
    elif skipped > 0:
        yield ([], torch.empty(0, max_len, dtype=torch.long), torch.empty(0), skipped)


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


def decode_words(
    *,
    config_path: str | None = None,
    checkpoint_path: str,
    words: List[str],
    batch_size: int | None = None,
    device: str | None = None,
) -> tuple[List[str], List[str]]:
    """Decode words by reconstructing them from a trained checkpoint.

    Args:
        config_path: Optional YAML config file path. If None, extracts config
            from checkpoint metadata (requires checkpoint saved with config).
        checkpoint_path: Path to trained model checkpoint (.pt file).
        words: List of input words to decode.
        batch_size: Number of words to process simultaneously. If None,
            uses the batch size from training config.
        device: Device string ("cuda", "cpu", etc.). If None, uses device
            from config.

    Returns:
        Tuple of (kept_words, reconstructed) where:
        - kept_words: List of words that passed length filtering (subset of input)
        - reconstructed: List of decoded strings aligned with kept_words
    """
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = _load_config_from_checkpoint(checkpoint_path)
    set_seed(int(cfg["seed"]))

    device_obj = select_device(device or cfg["device"])
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    vocab = CharVocab()
    max_word_len = int(data_cfg["max_word_len"])
    model_max_len = int(model_cfg["max_word_len"])
    if max_word_len != model_max_len:
        raise ValueError(
            "data.max_word_len must match model.max_word_len "
            f"(data={max_word_len}, model={model_max_len})"
        )

    if not words:
        raise ValueError("No words provided for decoding.")

    eff_batch_size = (
        int(batch_size)
        if batch_size is not None
        else int(train_cfg["batch_size"])
    )

    model = GAWAModel(
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

    kept_words: List[str] = []
    reconstructed: List[str] = []

    model.eval()
    with torch.inference_mode():
        for batch_words, char_ids, lengths, _skipped in _iter_batches(
            words, vocab, max_word_len, eff_batch_size
        ):
            if not batch_words:
                continue
            char_ids = char_ids.to(device_obj)
            lengths = lengths.to(device_obj)
            decoded = model.reconstruct(char_ids, lengths, vocab)
            kept_words.extend(batch_words)
            reconstructed.extend(decoded)

    if not kept_words:
        raise ValueError(
            "No valid words after length filtering. "
            "Check max_word_len or input content."
        )

    return kept_words, reconstructed
