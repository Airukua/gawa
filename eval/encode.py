"""
Programmatic embedding extraction for GAWA language model.

Use :func:`encode_words` from Python. For CLI usage, see `scripts/encode.py`.
"""

from __future__ import annotations
from typing import Iterable, Iterator, List, Tuple

import numpy as np
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
    
    Args:
        words: Iterable of raw word strings.
        vocab: Character vocabulary for encoding.
        max_len: Maximum sequence length (including BOS/EOS tokens).
        batch_size: Number of words per batch.
    
    Yields:
        Tuple of (word_strings, char_ids, lengths, skipped_count) where:
        - word_strings: List of original words that passed filtering
        - char_ids: Tensor of shape (batch_size, max_len) with padded token IDs
        - lengths: Tensor of shape (batch_size,) with original sequence lengths
        - skipped_count: Number of words skipped in this batch due to length
    
    Example::
        >>> vocab = CharVocab()
        >>> words = ["hi", "hello", "a", "verylongwordthatexceedslimits"]
        >>> for batch_words, ids, lengths, skipped in _iter_batches(
        ...     words, vocab, max_len=10, batch_size=2
        ... ):
        ...     print(f"Batch: {batch_words}, shapes: {ids.shape}, skipped: {skipped}")
        Batch: ['hi', 'hello'], shapes: torch.Size([2, 10]), skipped: 0
        Batch: ['a'], shapes: torch.Size([1, 10]), skipped: 1  # verylongword filtered
    """
    batch_words: List[str] = []
    batch_ids: List[List[int]] = []
    batch_lengths: List[int] = []
    skipped = 0

    for word in words:
        word = word.strip()
        if not word:
            continue
        # Filter words outside valid length range (accounting for BOS/EOS)
        if len(word) < 1 or len(word) > max_len - 2:
            skipped += 1
            continue
        
        # Encode and pad to fixed length
        ids = vocab.encode(word)
        length = len(ids)
        pad_len = max_len - length
        batch_words.append(word)
        batch_ids.append(ids + [vocab.PAD] * pad_len)
        batch_lengths.append(length)

        # Yield complete batch
        if len(batch_words) >= batch_size:
            yield (
                batch_words,
                torch.tensor(batch_ids, dtype=torch.long),
                torch.tensor(batch_lengths, dtype=torch.long),
                skipped,
            )
            batch_words, batch_ids, batch_lengths = [], [], []
            skipped = 0

    # Yield final partial batch
    if batch_words:
        yield (
            batch_words,
            torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(batch_lengths, dtype=torch.long),
            skipped,
        )
    elif skipped > 0:
        # Edge case: all remaining words were filtered, report skip count
        yield ([], torch.empty(0, max_len, dtype=torch.long), torch.empty(0), skipped)


def _load_config_from_checkpoint(checkpoint_path: str) -> dict:
    """Extract training configuration from checkpoint metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file containing embedded config.
    
    Returns:
        Configuration dictionary.
    
    Raises:
        ValueError: If checkpoint does not contain config metadata.
    
    Example::
        >>> cfg = _load_config_from_checkpoint("model.pt")
        >>> print(cfg["model"]["eword_dim"])
        256
    """
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


def encode_words(
    *,
    config_path: str | None = None,
    checkpoint_path: str,
    words: List[str],
    batch_size: int | None = None,
    device: str | None = None,
) -> tuple[List[str], np.ndarray]:
    """Encode words into embedding vectors using trained GAWA model.
    
    Loads a trained checkpoint and extracts eword (encoder word) embeddings
    for a list of input words. Handles batching, padding, and device placement
    automatically.
    
    Args:
        config_path: Optional YAML config file path. If None, extracts config
            from checkpoint metadata (requires checkpoint saved with config).
        checkpoint_path: Path to trained model checkpoint (.pt file).
        words: List of input words to encode.
        batch_size: Number of words to process simultaneously. If None,
            uses the batch size from training config.
        device: Device string ("cuda", "cpu", etc.). If None, uses device
            from config.
    
    Returns:
        Tuple of (kept_words, embeddings) where:
        - kept_words: List of words that passed length filtering (subset of input)
        - embeddings: NumPy array of shape (N, eword_dim) where N is len(kept_words)
    
    Raises:
        ValueError: If no words provided, or all words filtered due to length,
            or checkpoint lacks embedded config when config_path is None.
    
    Example - basic encoding::
        >>> words = ["hello", "world", "test"]
        >>> kept, embs = encode_words(
        ...     config_path=None,
        ...     checkpoint_path="model.pt",
        ...     words=words
        ... )
        >>> print(f"Encoded {len(kept)} words to vectors of dim {embs.shape[1]}")
        Encoded 3 words to vectors of dim 256
    
    Example - with filtering::
        >>> words = ["hi", "supercalifragilisticexpialidocious"]  # second too long
        >>> kept, embs = encode_words(
        ...     config_path=None,
        ...     checkpoint_path="model.pt",
        ...     words=words,
        ...     max_word_len=32  # model's max length
        ... )
        >>> print(kept)
        ['hi']  # Long word filtered out
    """
    # Load configuration
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = _load_config_from_checkpoint(checkpoint_path)
    set_seed(int(cfg["seed"]))

    # Setup device and extract config sections
    device_obj = select_device(device or cfg["device"])
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # Initialize vocabulary and validate length constraints
    vocab = CharVocab()
    max_word_len = int(data_cfg["max_word_len"])
    model_max_len = int(model_cfg["max_word_len"])
    if max_word_len != model_max_len:
        raise ValueError(
            "data.max_word_len must match model.max_word_len "
            f"(data={max_word_len}, model={model_max_len})"
        )

    if not words:
        raise ValueError("No words provided for encoding.")

    # Determine batch size
    eff_batch_size = (
        int(batch_size)
        if batch_size is not None
        else int(train_cfg["batch_size"])
    )

    # Initialize and load model
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

    # Extract embeddings in batches
    all_embeddings: List[np.ndarray] = []
    kept_words: List[str] = []

    model.eval()
    with torch.inference_mode():
        for batch_words, char_ids, lengths, _skipped in _iter_batches(
            words, vocab, max_word_len, eff_batch_size
        ):
            if not batch_words:
                continue
            char_ids = char_ids.to(device_obj)
            lengths = lengths.to(device_obj)
            # Extract encoder output (eword) and move to CPU
            eword = model.encode(char_ids, lengths).cpu().numpy()
            all_embeddings.append(eword)
            kept_words.extend(batch_words)

    if not kept_words:
        raise ValueError(
            "No valid words after length filtering. "
            f"Check max_word_len={max_word_len} and dataset contents."
        )

    embeddings = np.concatenate(all_embeddings, axis=0)
    return kept_words, embeddings
