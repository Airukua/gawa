# GAWA — Gaussian-Weighted Abstraction for Word Architecture

> A morphological character-level encoder/decoder with Gaussian Positional Encoding,  
> designed as a front-end module for large language models.

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-AiRukua-yellow)](https://huggingface.co/AiRukua)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Abdul%20Wahid%20Rukua-blue)](https://id.linkedin.com/in/abdul-wahid-rukua)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

---

## Overview

**GAWA** is a word-level morphological autoencoder that encodes any word — including unseen or morphologically complex words — into a dense embedding vector (`eword`) using character-level representations weighted by a **Gaussian positional prior**.

Unlike subword tokenizers (BPE, WordPiece, SentencePiece), GAWA treats each word as a sequence of characters and compresses it into a single fixed-size vector. This makes it:

- **Language-agnostic**: Works on any character-based language without a pretrained vocabulary
- **Morphology-aware**: Positional weighting captures prefix/suffix importance naturally
- **Compact**: The output sequence length equals the number of words, not subword tokens

GAWA is designed to plug in as the **front-end morphological module** of a Global Transformer, replacing the tokenizer entirely.

---

## Architecture

```
Input Word (characters)
        │
        ├──► Char Embedding  (trainable)
        │
        ├──► Gaussian Positional Encoding  (fixed, non-trainable)
        │         μ_j = j,   σ_j = √j
        │
        └──► Concat → Fusion MLP
                          │
                    Weighted Pooling
                    (Gaussian Prior + Learnable Δ)
                          │
                    Output Projection
                          │
                       EWORD Vector  ──────────────────────────┐
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │    GAWA Decoder     │
                                                    │  Init GRU Hidden    │
                                                    │  Char Emb + Concat  │
                                                    │  GRU Cell           │
                                                    │  Cross-Attention    │
                                                    │  Residual + Logits  │
                                                    └─────────────────────┘
```

See the full diagram: **GAWA Architecture** (above image).

---

## Key Components

### 1. Gaussian Positional Encoding (Fixed)
Each character position `i` is encoded using `dim` Gaussian basis functions:

$$\text{GPE}(i, j) = \exp\left(-\frac{(i - \mu_j)^2}{2\sigma_j^2}\right), \quad \mu_j = j, \quad \sigma_j = \sqrt{j}$$

This is **non-trainable** — it provides a stable, smooth spatial prior over character positions without needing learned position embeddings.

### 2. Gaussian Position Prior (Weighted Pooling)
Instead of mean pooling, GAWA uses a position-importance prior inspired by psycholinguistic findings that the beginning and end of a word carry more information:

$$\sigma_i = d - (d - s_0) \cdot e^{-r \cdot i}$$

$$w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}$$

Default hyperparameters: `d=1.617`, `s0=0.5`, `r=1.105`

A small learnable MLP further adjusts these weights (`lambda_adjust=0.3`), allowing the model to fine-tune positional importance during training.

### 3. Encoder → EWORD Vector
The fused character representations are pooled via the weighted sum to produce a single `eword` vector of dimension `768` (compatible with standard transformer hidden sizes).

### 4. Decoder (GRU + Cross-Attention)
The decoder reconstructs the original word character-by-character using:
- GRU initialized from the `eword` vector
- Direct `eword` context concatenated at each step
- Cross-attention over the `eword` as key/value
- Residual connection before the output projection

---

## Installation

```bash
git clone https://github.com/AiRukua/gawa
cd gawa
pip install torch
```

No additional dependencies beyond PyTorch.

---

## Quick Start

### Training

```python
from gawa import train_gawa

words = ["makan", "memakan", "makanan", "dimakan", ...]  # your word list

model, vocab = train_gawa(
    words=words,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    eword_dim=768,
    save_path="gawa_checkpoint.pt",
)
```

### Encoding a Sentence

```python
from gawa import encode_sentence

sentence = "saya sedang belajar membuat model bahasa"
ewords = encode_sentence(sentence, model, vocab, device="cpu")

# ewords.shape → (num_words, 768)
# Ready to feed into a Global Transformer!
```

### Reconstruct (Sanity Check)

```python
reconstructed = model.reconstruct(char_ids, lengths, vocab)
# ['makan', 'memakan', 'makanan', ...]
```

---

## Model Dimensions

| Parameter         | Default | Description                          |
|-------------------|---------|--------------------------------------|
| `char_emb_dim`    | 64      | Character embedding size             |
| `pos_enc_dim`     | 64      | Gaussian PE dimension                |
| `hidden_dim`      | 256     | Fusion MLP & GRU hidden size         |
| `eword_dim`       | 768     | Output word embedding dimension      |
| `max_word_len`    | 32      | Maximum word length in characters    |
| `lambda_adjust`   | 0.3     | Weight of learnable position delta   |

---

## Why GAWA?

| Feature                        | BPE / WordPiece | GAWA           |
|-------------------------------|-----------------|----------------|
| Handles unseen words           | ✗ (UNK/fallback) | ✓ (char-based)|
| Morphology-aware               | Partial          | ✓ Explicit     |
| Sequence length                | Longer (subwords)| Shorter (words)|
| Language-specific vocab needed | ✓               | ✗              |
| Trainable end-to-end           | ✓               | ✓              |
| Positional character weighting | ✗               | ✓ Gaussian     |

---

## Recommended Training Data

For best results, train GAWA on a large, diverse word list:

- **Minimum**: ~10,000 unique words
- **Recommended**: 500,000 – 1,000,000 unique words
- **Sources**: Language dictionary, Wikipedia word dump, Common Crawl vocabulary

The model learns word reconstruction, so any large vocabulary list works — no labels required.

---

## Integration with Global Transformer

GAWA outputs an `eword` sequence that maps directly to transformer input:

```
Sentence: "saya belajar NLP"
           ↓       ↓      ↓
         eword₁  eword₂  eword₃     shape: (3, 768)
                   ↓
           Global Transformer
```

Because sequence length = number of words (not subword tokens), the attention window is used far more efficiently, especially for morphologically rich languages like Indonesian, Arabic, Finnish, or Turkish.


## Author

**Abdul Wahid Rukua**  
🤗 [huggingface.co/AiRukua](https://huggingface.co/AiRukua)  
💼 [linkedin.com/in/abdul-wahid-rukua](https://id.linkedin.com/in/abdul-wahid-rukua)

---

## License

MIT License. See `LICENSE` for details.