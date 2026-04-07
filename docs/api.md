# Python API

This page lists the primary public entrypoints.

## `train_from_config`

```python
from gawa import load_config, train_from_config

cfg = load_config("configs/gawa_small.yaml")
train_from_config(cfg)
```

## `encode_words`

```python
from gawa import encode_words

kept_words, embeddings = encode_words(
    checkpoint_path="checkpoints/gawa_small/best.pt",
    words=["makan", "memakan", "makanan"],
)
```

## `GAWAModel`

```python
from gawa import GAWAModel, CharVocab
import torch

vocab = CharVocab()
model = GAWAModel(vocab_size=vocab.vocab_size)
char_ids = torch.zeros(2, 8, dtype=torch.long)
lengths = torch.tensor([5, 3])
logits, eword = model(char_ids, lengths)
```

## `GAWAModel.from_pretrained`

```python
from gawa import GAWAModel

model = GAWAModel.from_pretrained("AiRukua/gawa")
kept_words, embs = model.encode_words(["makan", "memakan", "makanan"])
kept_words, recs = model.decode_words(["makan", "memakan", "makanan"])
```

## Notes

- The `gawa` package re-exports `encode_words`, `load_config`, `train_from_config`.
- Use `CharVocab` to encode raw words when building custom pipelines.
- `GAWAModel.encode_words` / `decode_words` are available only when the model
  is loaded via `from_pretrained()`.
