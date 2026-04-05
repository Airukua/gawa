"""
Programmatic training utilities for GAWA.

Use :func:`train_from_config` from Python. For CLI usage, see `scripts/train.py`.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any

import torch

from model.char_vocab import CharVocab
from model.gawa_lm import GAWAModel
from training.checkpoint import load_checkpoint, save_checkpoint
from training.data import (
    WordDataset,
    build_dataloader,
    load_words,
    split_words_three_way,
)
from training.loop import (
    evaluate,
    print_epoch_summary,
    run_epoch,
    sample_reconstructions,
)
from training.scheduler import SchedulerConfig, build_scheduler
from training.utils import maybe_init_wandb, select_device, set_seed


def train_from_config(cfg: dict) -> None:
    """Run training programmatically from an in-memory config dict."""
    set_seed(int(cfg["seed"]))

    device = select_device(cfg["device"])
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

    if data_cfg.get("val_path") or data_cfg.get("test_path"):
        train_words = load_words(data_cfg["train_path"])
        val_words = (
            load_words(data_cfg["val_path"])
            if data_cfg.get("val_path")
            else []
        )
        test_words = (
            load_words(data_cfg["test_path"])
            if data_cfg.get("test_path")
            else []
        )
    else:
        all_words = load_words(data_cfg["train_path"])
        train_words, val_words, test_words = split_words_three_way(
            all_words,
            float(data_cfg["val_split"]),
            float(data_cfg.get("test_split", 0.0)),
            int(cfg["seed"]),
        )

    train_ds = WordDataset(train_words, vocab, max_len=max_word_len)
    val_ds = WordDataset(val_words, vocab, max_len=max_word_len) if val_words else None
    test_ds = WordDataset(test_words, vocab, max_len=max_word_len) if test_words else None

    train_loader = build_dataloader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
    )

    val_loader = None
    test_loader = None
    if val_ds is not None:
        val_loader = build_dataloader(
            val_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=bool(train_cfg["pin_memory"]),
        )
    if test_ds is not None:
        test_loader = build_dataloader(
            test_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            pin_memory=bool(train_cfg["pin_memory"]),
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
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    scheduler_cfg = SchedulerConfig(**cfg["scheduler"])
    scheduler = build_scheduler(
        optimizer,
        total_steps=int(train_cfg["epochs"]),
        config=scheduler_cfg,
    )

    scaler = None
    if bool(train_cfg["amp"]) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    wandb_run = maybe_init_wandb(cfg.get("wandb", {}))

    ckpt_cfg = train_cfg["checkpoint"]
    early_cfg = train_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 5))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = None
    epochs_no_improve = 0
    start_epoch = 1
    global_step = 0

    if ckpt_cfg.get("resume_path"):
        state = load_checkpoint(
            ckpt_cfg["resume_path"],
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = state.epoch + 1
        global_step = state.step
        best_val = state.best_metric
        print(f"Resumed from {ckpt_cfg['resume_path']} at epoch={state.epoch}")

    total_epochs = int(train_cfg["epochs"])

    def _select_evenly_spaced_epochs(
        epochs: list[int],
        total_epochs: int,
        max_keep: int,
    ) -> set[int]:
        if max_keep <= 0 or len(epochs) <= max_keep:
            return set(epochs)
        if max_keep == 1:
            return {max(epochs)}

        targets = [
            1 + i * (total_epochs - 1) / (max_keep - 1)
            for i in range(max_keep)
        ]
        remaining = set(epochs)
        keep: list[int] = []
        for target in targets:
            if not remaining:
                break
            closest = min(
                remaining,
                key=lambda e: (abs(e - target), e),
            )
            keep.append(closest)
            remaining.remove(closest)

        if len(keep) < max_keep:
            for epoch in sorted(remaining, reverse=True):
                keep.append(epoch)
                if len(keep) == max_keep:
                    break

        return set(keep)

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            grad_clip_norm=float(train_cfg["grad_clip_norm"]),
            log_every=int(train_cfg["log_every"]),
            pad_idx=vocab.PAD,
            teacher_forcing=bool(train_cfg["teacher_forcing"]),
            use_tqdm=bool(train_cfg.get("use_tqdm", True)),
            wandb_run=wandb_run,
            epoch=epoch,
            global_step=global_step,
        )

        val_loss = None
        if val_loader is not None and epoch % int(train_cfg["eval_every"]) == 0:
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                pad_idx=vocab.PAD,
            )

        elapsed = time.time() - epoch_start
        print_epoch_summary(
            epoch=epoch,
            total_epochs=int(train_cfg["epochs"]),
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            elapsed=elapsed,
        )

        if scheduler is not None:
            scheduler.step()

        if wandb_run is not None:
            payload: dict[str, Any] = {
                "train/epoch_loss": train_loss,
                "train/epoch_acc": train_acc,
                "epoch": epoch,
            }
            if val_loss is not None:
                payload["val/loss"] = val_loss
            wandb_run.log(payload, step=global_step)

        if epoch % int(ckpt_cfg["save_every"]) == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                str(ckpt_path),
                model,
                optimizer,
                epoch=epoch,
                step=global_step,
                scheduler=scheduler,
                best_metric=best_val,
                config=cfg,
            )
            max_keep = int(ckpt_cfg.get("max_keep", 0))
            if max_keep > 0:
                ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
                if len(ckpts) > max_keep:
                    epoch_map: dict[int, Path] = {}
                    for path in ckpts:
                        stem = path.stem
                        if stem.startswith("epoch_"):
                            try:
                                epoch_idx = int(stem.split("_", 1)[1])
                            except ValueError:
                                continue
                            epoch_map[epoch_idx] = path
                    keep_epochs = _select_evenly_spaced_epochs(
                        sorted(epoch_map),
                        total_epochs=total_epochs,
                        max_keep=max_keep,
                    )
                    for epoch_idx, path in epoch_map.items():
                        if epoch_idx not in keep_epochs:
                            path.unlink(missing_ok=True)

        if val_loss is not None:
            prev_best = best_val
            improved = best_val is None or val_loss < best_val
            if improved:
                best_val = val_loss
                if ckpt_cfg.get("save_best", True):
                    best_path = ckpt_dir / "best.pt"
                    save_checkpoint(
                        str(best_path),
                        model,
                        optimizer,
                        epoch=epoch,
                        step=global_step,
                        scheduler=scheduler,
                        best_metric=best_val,
                        config=cfg,
                    )

            if early_enabled:
                if prev_best is None:
                    epochs_no_improve = 0
                else:
                    delta = prev_best - val_loss
                    if delta > early_min_delta:
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= early_patience:
                            print(
                                "Early stopping triggered: no improvement for "
                                f"{early_patience} eval steps."
                            )
                            break

        sample_every = int(train_cfg.get("sample_every", 0))
        sample_count = int(train_cfg.get("sample_count", 5))
        if sample_every > 0 and epoch % sample_every == 0:
            sample_words = train_words[:sample_count]
            sample_reconstructions(
                model=model,
                vocab=vocab,
                words=sample_words,
                max_len=max_word_len,
                device=device,
            )

    if wandb_run is not None:
        wandb_run.finish()

    if test_loader is not None:
        test_loss = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            pad_idx=vocab.PAD,
        )
        print(f"Test Loss: {test_loss:.4f}")


def _select_evenly_spaced_epochs(
    epochs: list[int],
    total_epochs: int,
    max_keep: int,
) -> set[int]:
    """Select a subset of epoch indices to retain for checkpoint cleanup.

    Uses a greedy algorithm to keep checkpoints spread across the timeline.

    Args:
        epochs: Sorted list of epoch indices that have checkpoints.
        total_epochs: Total number of epochs in training run.
        max_keep: Maximum number of checkpoints to retain.
    
    Returns:
        Set of epoch indices to keep.
    """
    if max_keep <= 0 or len(epochs) <= max_keep:
        return set(epochs)
    if max_keep == 1:
        return {max(epochs)}

    # Target positions evenly distributed across timeline.
    targets = [
        1 + i * (total_epochs - 1) / (max_keep - 1)
        for i in range(max_keep)
    ]
    
    remaining = set(epochs)
    keep: list[int] = []

    # Greedy selection: pick closest available epoch to each target.
    for target in targets:
        if not remaining:
            break
        closest = min(
            remaining,
            key=lambda e: (abs(e - target), e),
        )
        keep.append(closest)
        remaining.remove(closest)

    # Fill any remaining slots with most recent epochs if needed.
    if len(keep) < max_keep:
        for epoch in sorted(remaining, reverse=True):
            keep.append(epoch)
            if len(keep) == max_keep:
                break

    return set(keep)
