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

For very large files, process in batches:

```bash
gawa-prepare --input data/raw.txt --output data/processed/train.txt --lower --batch-lines 50000
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

## Quick Run (Python - Training)

```python
from gawa import load_config, train_from_config

cfg = load_config("configs/gawa_small.yaml")
train_from_config(cfg)
```

## Quick Run (Python - Pretrained)

```python
from gawa import GAWAModel

model = GAWAModel.from_pretrained("AiRukua/gawa")

kept_words, embs = model.encode_words(["makan", "memakan", "makanan"])
kept_words, recs = model.decode_words(["makan", "memakan", "makanan"])
```
