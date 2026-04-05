"""
Training loop utilities for GAWA.

Provides epoch training, evaluation, logging helpers, and reconstructions.
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.char_vocab import CharVocab
from model.gawa_lm import GAWAModel
from training.utils import accuracy_from_logits


def run_epoch(
    model: GAWAModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip_norm: float,
    log_every: int,
    pad_idx: int,
    teacher_forcing: bool,
    use_tqdm: bool,
    wandb_run: Optional[Any],
    epoch: int,
    global_step: int,
) -> tuple[float, float, int]:
    """Run one training epoch.

    Handles optimization, optional LR scheduling, and AMP via ``scaler``.

    Args:
        model: GAWAModel instance.
        loader: Training dataloader yielding dicts with keys
            ``char_ids``, ``lengths``, ``target``.
        optimizer: Optimizer with model parameters registered.
        scheduler: Optional LR scheduler (stepped per batch).
        device: Target device for tensors.
        scaler: Optional GradScaler for mixed precision.
        grad_clip_norm: Maximum gradient norm; <= 0 disables clipping.
        log_every: Log interval in steps. Set to 0 to disable step logging.
        pad_idx: Padding index in target tensor (ignored in loss computation).
        teacher_forcing: Use ground truth as decoder input.
        use_tqdm: Wrap dataloader with progress bar showing epoch progress.
        wandb_run: Weights & Biases run object for metric logging.
        epoch: Current epoch number (for display/logging purposes).
        global_step: Global step counter (updated and returned).

    Returns:
        Tuple of ``(average_loss, average_accuracy, updated_global_step)``.

    Raises:
        RuntimeError: If ``scaler`` is provided but CUDA is not available.
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    count = 0

    iterator = tqdm(loader, desc=f"epoch {epoch}") if use_tqdm else loader

    for step, batch in enumerate(iterator, start=1):
        char_ids = batch["char_ids"].to(device)
        lengths = batch["lengths"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # AMP context enabled only when scaler is provided.
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(autocast_device, enabled=scaler is not None):
            logits, _ = model(
                char_ids,
                lengths,
                target_ids=target,
                teacher_forcing=teacher_forcing,
            )
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=pad_idx,
            )

        # Backward pass with optional gradient scaling.
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Update metrics.
        acc = accuracy_from_logits(logits.detach(), target, pad_idx)
        running_loss += loss.item()
        running_acc += acc
        count += 1
        global_step += 1

        # Periodic logging.
        if log_every > 0 and step % log_every == 0:
            avg_loss = running_loss / count
            avg_acc = running_acc / count
            print(f"epoch={epoch} step={step} loss={avg_loss:.4f} acc={avg_acc:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {"train/loss": avg_loss, "train/acc": avg_acc},
                    step=global_step,
                )

    return running_loss / max(1, count), running_acc / max(1, count), global_step


def evaluate(
    model: GAWAModel,
    loader: DataLoader,
    device: torch.device,
    pad_idx: int,
) -> float:
    """Compute mean validation loss in inference mode.

    Args:
        model: GAWAModel instance.
        loader: Validation dataloader with same structure as training.
        device: Target device.
        pad_idx: Padding index for loss masking.

    Returns:
        Mean cross-entropy loss across all validation batches.
    """
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            char_ids = batch["char_ids"].to(device)
            lengths = batch["lengths"].to(device)
            target = batch["target"].to(device)

            logits, _ = model(
                char_ids,
                lengths,
                target_ids=target,
                teacher_forcing=True,
            )
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=pad_idx,
            )
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float],
    elapsed: float,
) -> None:
    """Print formatted single-line epoch summary to stdout.
    """
    msg = (
        f"Epoch {epoch:3d}/{total_epochs} | Loss: {train_loss:.4f} | "
        f"Char Acc: {train_acc * 100:.2f}%"
    )
    if val_loss is not None:
        msg += f" | Val Loss: {val_loss:.4f}"
    msg += f" time={elapsed:.1f}s"
    print(msg)


def sample_reconstructions(
    model: GAWAModel,
    vocab: CharVocab,
    words: list[str],
    max_len: int,
    device: torch.device,
) -> None:
    """Display model reconstructions for qualitative evaluation.
    """
    # Filter to valid length range (accounting for BOS/EOS tokens)
    valid_words = [w for w in words if 1 <= len(w) <= max_len - 2]
    if not valid_words:
        print("No valid words in sample list (check length constraints)")
        return

    model.eval()
    from training.data import WordDataset

    dataset = WordDataset(valid_words, vocab, max_len=max_len)
    batch = {
        "char_ids": torch.stack(
            [dataset[i]["char_ids"] for i in range(len(dataset))]
        ).to(device),
        "lengths": torch.stack(
            [dataset[i]["lengths"] for i in range(len(dataset))]
        ).to(device),
    }

    reconstructed = model.reconstruct(batch["char_ids"], batch["lengths"], vocab)

    print("\n  Sample Reconstruction:")
    for orig, rec in zip(valid_words, reconstructed):
        status = "OK" if orig.lower() == rec.lower() else "NO"
        print(f"    {status} '{orig}' -> '{rec}'")
    print()
