"""
Programmatic evaluation helpers for GAWA language model.

Use from code via :func:`evaluate_dataset`. For CLI usage, see
`scripts/evaluate.py`.
"""

from __future__ import annotations

import torch

from model.char_vocab import CharVocab
from model.gawa_lm import GAWAModel
from training.checkpoint import load_checkpoint
from training.data import WordDataset, build_dataloader, load_words
from training.loop import evaluate, sample_reconstructions
from training.utils import select_device, set_seed


def _load_config_from_checkpoint(checkpoint_path: str) -> dict:
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    cfg = state.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(
            "Checkpoint does not contain an embedded config. "
            "Re-train with config saving enabled."
        )
    return cfg


def evaluate_dataset(
    *,
    checkpoint_path: str,
    data_path: str,
    batch_size: int | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    sample_count: int = 0,
) -> float:
    """Evaluate a trained checkpoint on a separate dataset."""
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

    words = load_words(data_path)
    eval_ds = WordDataset(words, vocab, max_len=max_word_len)
    if len(eval_ds) == 0:
        raise ValueError(
            "No valid words after length filtering. "
            f"Check max_word_len={max_word_len} and dataset contents."
        )

    eff_batch_size = (
        int(batch_size)
        if batch_size is not None
        else int(train_cfg["batch_size"])
    )
    eff_num_workers = (
        int(num_workers)
        if num_workers is not None
        else int(train_cfg["num_workers"])
    )
    if pin_memory is None:
        eff_pin_memory = bool(train_cfg["pin_memory"])
    else:
        eff_pin_memory = bool(pin_memory)

    eval_loader = build_dataloader(
        eval_ds,
        batch_size=eff_batch_size,
        shuffle=False,
        num_workers=eff_num_workers,
        pin_memory=eff_pin_memory,
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

    eval_loss = evaluate(
        model=model,
        loader=eval_loader,
        device=device_obj,
        pad_idx=vocab.PAD,
    )

    if sample_count > 0:
        sample_words = words[:sample_count]
        sample_reconstructions(
            model=model,
            vocab=vocab,
            words=sample_words,
            max_len=max_word_len,
            device=device_obj,
        )

    return eval_loss
