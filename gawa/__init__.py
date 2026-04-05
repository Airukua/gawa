"""Public API for the GAWA package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from eval.encode import encode_words
from model.char_vocab import CharVocab
from model.gawa_lm import GAWAModel
from training.config import load_config
from training.trainer import train_from_config

__all__ = [
    "GAWAModel",
    "CharVocab",
    "encode_words",
    "load_config",
    "train_from_config",
]


def __getattr__(name: str):
    if name == "__version__":
        try:
            return version("gawa")
        except PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(name)
