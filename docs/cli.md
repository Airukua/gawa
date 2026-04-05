# Command-Line Interface

GAWA provides several CLI entrypoints after installation.

## Training

```bash
gawa-train --config configs/gawa_small.yaml
```

## Data Preparation

```bash
gawa-prepare --input data/raw.txt --output data/processed/train.txt --lower
```

## Encoding

```bash
gawa-encode \
  --checkpoint checkpoints/gawa_small/best.pt \
  --words "makan,memakan,makanan"
```

Output options:

- `--output` write to `.jsonl` or `.npy`
- `--format` force `jsonl` or `npy`
- `--batch-size` override batch size
- `--device` override device

## Evaluation

```bash
gawa-evaluate --config configs/gawa_small.yaml --checkpoint checkpoints/gawa_small/best.pt
```
