from __future__ import annotations
import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from model.char_vocab import CharVocab


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WordDataset(Dataset):
    """One-word-per-line dataset with BOS/EOS targets.

    Each item returns a dictionary with:
      - ``char_ids``: padded character ids for the encoder
      - ``lengths``: true character length (without padding)
      - ``target``: BOS + chars + EOS, padded to ``max_len``

    Args:
        words: List of raw word strings.
        vocab: :class:`CharVocab` instance.
        max_len: Maximum word length (encoder input length).
    """

    def __init__(
        self,
        words: list[str],
        vocab: CharVocab,
        max_len: int = 32,
    ) -> None:
        self.vocab = vocab
        self.max_len = max_len
        # Filter words that exceed the maximum length budget.
        self.words = [w for w in words if 1 <= len(w) <= max_len - 2]

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        word = self.words[idx]
        char_ids = self.vocab.encode(word)
        length = len(char_ids)

        # Target sequence: BOS + chars + EOS.
        target = [self.vocab.BOS] + char_ids + [self.vocab.EOS]

        # Pad encoder input to max_len.
        pad_len = self.max_len - length
        char_ids_padded = char_ids + [self.vocab.PAD] * pad_len

        # Pad target to (max_len + 1) and truncate to max_len for alignment.
        target_pad_len = (self.max_len + 1) - len(target)
        target_padded = target + [self.vocab.PAD] * target_pad_len

        return {
            "char_ids": torch.tensor(char_ids_padded, dtype=torch.long),
            "lengths": torch.tensor(length, dtype=torch.long),
            "target": torch.tensor(
                target_padded[: self.max_len], dtype=torch.long
            ),
        }


# ---------------------------------------------------------------------------
# Word list utilities
# ---------------------------------------------------------------------------


def load_words(path: str) -> list[str]:
    """Load a one-word-per-line text file.

    Args:
        path: Path to the word list file.

    Returns:
        A list of words.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dataset file: {path}. "
            "Provide data.train_path or create the file."
        )
    words: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    if not words:
        raise ValueError(f"Dataset file is empty: {path}")
    return words


def split_words(
    words: list[str],
    val_split: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split words into train and validation sets.

    Args:
        words: Input word list.
        val_split: Fraction of words to reserve for validation.
        seed: Random seed for deterministic split.

    Returns:
        A ``(train_words, val_words)`` tuple.
    """
    if val_split <= 0:
        return words, []
    rng = random.Random(seed)
    indices = list(range(len(words)))
    rng.shuffle(indices)
    val_size = max(1, int(len(words) * val_split))
    val_idx = set(indices[:val_size])
    train_words, val_words = [], []
    for i, w in enumerate(words):
        if i in val_idx:
            val_words.append(w)
        else:
            train_words.append(w)
    return train_words, val_words


def split_words_three_way(
    words: list[str],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split words into train, validation, and test sets.

    Args:
        words: Input word list.
        val_split: Fraction reserved for validation.
        test_split: Fraction reserved for test.
        seed: Random seed for deterministic split.

    Returns:
        A ``(train_words, val_words, test_words)`` tuple.
    """
    if val_split <= 0 and test_split <= 0:
        return words, [], []
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split + test_split must be < 1.0, got {val_split + test_split}"
        )
    rng = random.Random(seed)
    indices = list(range(len(words)))
    rng.shuffle(indices)
    val_size = int(len(words) * val_split)
    test_size = int(len(words) * test_split)
    val_idx = set(indices[:val_size])
    test_idx = set(indices[val_size: val_size + test_size])
    train_words, val_words, test_words = [], [], []
    for i, w in enumerate(words):
        if i in val_idx:
            val_words.append(w)
        elif i in test_idx:
            test_words.append(w)
        else:
            train_words.append(w)
    return train_words, val_words, test_words


# ---------------------------------------------------------------------------
# Dataloader builder
# ---------------------------------------------------------------------------


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader with common defaults.

    Args:
        dataset: Dataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle each epoch.
        num_workers: Number of background workers.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
