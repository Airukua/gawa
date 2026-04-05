# Getting Started

This page walks you through installation and a minimal end-to-end run.

## Installation

### Install via GitHub (pip)

```bash
pip install git+https://github.com/AiRukua/gawa.git
```

### Local Development Install

```bash
git clone https://github.com/AiRukua/gawa.git
cd gawa
pip install -e .
```

### Optional Dev Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Run (CLI)

1. Prepare a word list from raw text:

```bash
gawa-prepare --input data/raw.txt --output data/processed/train.txt --lower
```

2. Train using a config:

```bash
gawa-train --config configs/gawa_small.yaml
```

3. Encode words using a trained checkpoint:

```bash
gawa-encode \
  --checkpoint checkpoints/gawa_small/best.pt \
  --words "makan,memakan,makanan"
```

## Quick Run (Python)

```python
from gawa import encode_words, load_config, train_from_config

cfg = load_config("configs/gawa_small.yaml")
train_from_config(cfg)

kept_words, embeddings = encode_words(
    checkpoint_path="checkpoints/gawa_small/best.pt",
    words=["makan", "memakan", "makanan"],
)
print(embeddings.shape)
```
