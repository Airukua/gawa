# Training

This page describes the training workflow and checkpoints.

## Train from CLI

```bash
gawa-train --config configs/gawa_small.yaml
```

## What Happens During Training

- Words are loaded from `data.train_path` in the config.
- A `CharVocab` is built on the fly.
- The model is trained to reconstruct words.
- Checkpoints are saved under `training.checkpoint.dir`.

## Checkpoints

Checkpoints contain:

- Model weights
- Optimizer and scheduler state
- Training step and epoch
- Embedded config dictionary

The default best checkpoint path is typically:

- `checkpoints/gawa_small/best.pt`

## Resume Training

Set `training.checkpoint.resume_path` in your YAML config to resume.
