# Architecture

GAWA is a character-level autoencoder that produces one vector per word.

## Encoder

- Character embeddings
- Fixed Gaussian positional encoding
- Fusion MLP
- Gaussian-weighted pooling with a learnable adjustment

## Decoder

- GRU initialized from `eword`
- Cross-attention over `eword`
- Residual connection and output projection

## Output

The encoder output `eword` is a fixed-size embedding that can be fed into downstream transformer models.
