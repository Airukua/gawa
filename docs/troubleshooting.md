# Troubleshooting

## Training Fails with Length Mismatch

Error: `data.max_word_len must match model.max_word_len`

Fix: ensure both values are identical in your YAML config.

## Encoding Skips Words

Some words may be skipped if their length is outside the model range.

Fix: increase `data.max_word_len` and `model.max_word_len`, then retrain.

## CUDA Not Available

If training falls back to CPU, verify:

- CUDA toolkit is installed
- PyTorch build matches your CUDA version
- `device` is set to `cuda` in the config

## Slow Training

Tips:

- Use a GPU if available
- Increase `batch_size` if memory allows
- Reduce `max_word_len` if your language permits
