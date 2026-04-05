# Data Preparation

GAWA trains on a **word list** file: one word per line. You can generate it from raw text using the provided helper.

## From Raw Text

```bash
gawa-prepare --input data/raw.txt --output data/processed/train.txt --lower
```

### Options

- `--min-len` minimum token length
- `--max-len` maximum token length
- `--dedupe` remove duplicates while preserving order
- `--allow-redup` keep reduplicated hyphenated tokens (example: `go-go`)
- `--batch-lines` process input in batches of N lines (lower memory)

## Programmatic Usage

```python
from data.prepare import prepare_file

prepare_file(
    input_path="data/raw.txt",
    output_path="data/processed/train.txt",
    lower=True,
    min_len=1,
    max_len=64,
    dedupe=True,
    batch_lines=50000,
)
```

## Notes

- Tokens are extracted using a conservative ASCII regex for letters and hyphens.
- If `--allow-redup` is not set, hyphenated tokens are dropped.
