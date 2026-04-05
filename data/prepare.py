"""
Text cleaning helpers for word-list datasets.

Extracts alphabetic tokens from raw text and writes a one-word-per-line file
used by the training pipeline.

Example::
    >>> from data.prepare import prepare_words
    >>> prepare_words("Hello WORLD 123!", lower=True)
    ['hello', 'world']
"""

from __future__ import annotations

import re
from pathlib import Path


# ASCII word tokens with optional single-hyphen segments.
_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def _is_reduplication(token: str) -> bool:
    parts = token.split("-")
    if len(parts) <= 1:
        return True
    first = parts[0]
    return all(p == first for p in parts[1:])


def _iter_words(text: str) -> list[str]:
    """Return candidate word tokens extracted from raw text."""
    return _WORD_RE.findall(text)


def prepare_words(
    text: str,
    *,
    lower: bool = False,
    min_len: int = 1,
    max_len: int = 64,
    dedupe: bool = False,
    allow_redup: bool = False,
    allow_empty: bool = False,
) -> list[str]:
    """Clean raw text into a list of valid word tokens.

    Pipeline:
    1. Extract tokens with a conservative regex (ASCII letters + hyphens).
    2. Optionally lowercase.
    3. Filter by hyphen policy (drop or keep reduplications).
    4. Filter by length bounds.
    5. Optionally deduplicate while preserving order.

    Args:
        text: Raw input text. Must be a non-empty string.
        lower: Convert tokens to lowercase.
        min_len: Minimum token length to keep (inclusive).
        max_len: Maximum token length to keep (inclusive).
        dedupe: Remove duplicate tokens while preserving first occurrence.
        allow_redup: If True, allow hyphenated reduplications like
            ``go-go`` or ``bye-bye`` and filter out other hyphenated forms.
            If False, drop all tokens containing hyphens.
        allow_empty: If True, return an empty list instead of raising
            when no tokens remain.

    Returns:
        A list of cleaned word tokens.

    Raises:
        ValueError: If no tokens remain after filtering.
    """
    words = _iter_words(text)
    if lower:
        words = [w.lower() for w in words]

    if not allow_redup:
        # Keep only plain alphabetic tokens.
        words = [w for w in words if "-" not in w]
    else:
        # Keep only repeated hyphen patterns: "go-go", "bye-bye", ...
        words = [w for w in words if _is_reduplication(w)]

    words = [w for w in words if min_len <= len(w) <= max_len]

    if dedupe:
        seen = set()
        deduped = []
        for w in words:
            if w not in seen:
                deduped.append(w)
                seen.add(w)
        words = deduped

    if not words:
        if allow_empty:
            return []
        raise ValueError("No words after filtering; check your input text.")

    return words


def write_word_list(words: list[str], output_path: str | Path) -> None:
    """Write tokens to a one-word-per-line file.

    Args:
        words: List of tokens to write.
        output_path: Destination file path. Parent directories are created.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(words) + "\n", encoding="utf-8")


def prepare_file(
    input_path: str | Path,
    output_path: str | Path = "data/processed/train.txt",
    *,
    lower: bool = False,
    min_len: int = 1,
    max_len: int = 64,
    dedupe: bool = False,
    allow_redup: bool = False,
    batch_lines: int | None = None,
) -> list[str]:
    """Read a text file, clean it, and write a word list to disk.

    Args:
        input_path: Path to a UTF-8 text file (raw source text).
        output_path: Output path for the one-word-per-line dataset.
        lower: Convert tokens to lowercase.
        min_len: Minimum token length to keep (inclusive).
        max_len: Maximum token length to keep (inclusive).
        dedupe: Remove duplicate tokens while preserving order.
        allow_redup: Allow reduplicated hyphenated tokens (``go-go``).
        batch_lines: If set, process the input file in batches of N lines
            to reduce peak memory usage. When None, the entire file is read
            into memory at once.

    Returns:
        The cleaned list of tokens that were written to disk.
    """
    if batch_lines is None or batch_lines <= 0:
        text = Path(input_path).read_text(encoding="utf-8")
    words = prepare_words(
        text,
        lower=lower,
        min_len=min_len,
        max_len=max_len,
        dedupe=dedupe,
        allow_redup=allow_redup,
        allow_empty=False,
    )
    write_word_list(words, output_path)
    return words

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] | None = set() if dedupe else None
    collected: list[str] = []

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        buffer: list[str] = []
        for line in src:
            buffer.append(line)
            if len(buffer) >= batch_lines:
                _flush_batch(
                    buffer,
                    dst,
                    collected,
                    seen,
                    lower=lower,
                    min_len=min_len,
                    max_len=max_len,
                    allow_redup=allow_redup,
                )
                buffer = []
        if buffer:
            _flush_batch(
                buffer,
                dst,
                collected,
                seen,
                lower=lower,
                min_len=min_len,
                max_len=max_len,
                allow_redup=allow_redup,
            )

    if not collected:
        raise ValueError("No words after filtering; check your input text.")
    return collected


def _flush_batch(
    buffer: list[str],
    dst,
    collected: list[str],
    seen: set[str] | None,
    *,
    lower: bool,
    min_len: int,
    max_len: int,
    allow_redup: bool,
) -> None:
    text = "".join(buffer)
    words = prepare_words(
        text,
        lower=lower,
        min_len=min_len,
        max_len=max_len,
        dedupe=False,
        allow_redup=allow_redup,
        allow_empty=True,
    )

    if seen is not None:
        deduped: list[str] = []
        for w in words:
            if w not in seen:
                deduped.append(w)
                seen.add(w)
        words = deduped

    if words:
        dst.write("\n".join(words) + "\n")
        collected.extend(words)
