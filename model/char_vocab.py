"""Character-level vocabulary for sequence encoding and decoding.

Provides a compact, deterministic mapping between characters and integer
indices, with four reserved control tokens: ``<PAD>``, ``<BOS>``, ``<EOS>``,
and ``<UNK>``.

Typical usage::

    vocab = CharVocab()
    indices = vocab.encode("Hello")          # [33, 8, 12, 12, 15]
    text    = vocab.decode([vocab.BOS] + indices + [vocab.EOS])  # "Hello"
"""

from __future__ import annotations

__all__ = ["CharVocab"]

from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters supported out-of-the-box (order determines index assignment).
_ALPHABET: Final[str] = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,!?'-_"
)

# Reserved control tokens and their fixed indices.
_SPECIAL_TOKENS: Final[dict[str, int]] = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}

_FIRST_CHAR_IDX: Final[int] = len(_SPECIAL_TOKENS)  # 4


# ---------------------------------------------------------------------------
# CharVocab
# ---------------------------------------------------------------------------


class CharVocab:
    """Deterministic character vocabulary with four reserved control tokens.

    The index space is laid out as follows::

        0  →  <PAD>   (padding / ignored position)
        1  →  <BOS>   (begin of sequence)
        2  →  <EOS>   (end of sequence; terminates decoding)
        3  →  <UNK>   (out-of-vocabulary character)
        4+ →  printable characters (see ``_ALPHABET``)

    The vocabulary is **fixed at construction time** and contains no mutable
    state after ``__init__`` returns.  All public attributes are read-only
    by convention (prefixed with an underscore for internal dicts; exposed
    through properties).

    Attributes:
        PAD: Index reserved for padding tokens.
        BOS: Index reserved for the begin-of-sequence token.
        EOS: Index reserved for the end-of-sequence token.
        UNK: Index reserved for unknown / out-of-vocabulary characters.
        vocab_size: Total number of entries, including control tokens.

    Example::

        >>> vocab = CharVocab()
        >>> ids = vocab.encode("Hello123!")
        >>> vocab.decode([vocab.BOS] + ids + [vocab.EOS, vocab.PAD])
        'Hello123!'
    """

    # Control-token indices are class-level constants – identical for every
    # instance and useful for type-checked access without instantiation.
    PAD: Final[int] = _SPECIAL_TOKENS["<PAD>"]
    BOS: Final[int] = _SPECIAL_TOKENS["<BOS>"]
    EOS: Final[int] = _SPECIAL_TOKENS["<EOS>"]
    UNK: Final[int] = _SPECIAL_TOKENS["<UNK>"]

    def __init__(self) -> None:
        # Build char → index mapping.
        self._char2idx: dict[str, int] = dict(_SPECIAL_TOKENS)
        for offset, char in enumerate(_dedupe(iter(_ALPHABET))):
            self._char2idx[char] = _FIRST_CHAR_IDX + offset

        # Inverse mapping; built once and never mutated.
        self._idx2char: dict[int, str] = {v: k for k, v in self._char2idx.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens, including the four control tokens."""
        return len(self._char2idx)

    @property
    def char2idx(self) -> dict[str, int]:
        """Read-only view of the character-to-index mapping."""
        return self._char2idx

    @property
    def idx2char(self) -> dict[int, str]:
        """Read-only view of the index-to-character mapping."""
        return self._idx2char

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of vocabulary indices.

        Characters absent from the vocabulary are mapped to ``UNK`` (index 3).
        Control tokens (``<PAD>``, ``<BOS>``, ``<EOS>``, ``<UNK>``) embedded
        literally in ``text`` are treated as individual characters and will
        also resolve to ``UNK`` unless their string representation happens to
        appear in the alphabet (it does not by default).

        Args:
            text: The input string to encode. Must be a non-None ``str``.

        Returns:
            A list of non-negative integers of length ``len(text)``.

        Raises:
            TypeError:  If ``text`` is not a ``str``.
            ValueError: If ``text`` is ``None`` (guards against accidental
                ``Optional[str]`` misuse before type checkers catch it).

        Example::

            >>> vocab = CharVocab()
            >>> vocab.encode("words")
            [26, 18, 21, 7, 22]
        """
        _require_str(text, name="text")
        return [self._char2idx.get(ch, self.UNK) for ch in text]

    def decode(self, indices: list[int]) -> str:
        """Decode a list of vocabulary indices back into a string.

        Decoding stops as soon as an ``EOS`` token is encountered. ``PAD``,
        ``BOS``, and ``UNK`` tokens are silently skipped (not included in the
        output).

        Args:
            indices: A sequence of non-negative integer vocabulary indices.
                May include control-token indices.

        Returns:
            The decoded string, excluding all control tokens.

        Raises:
            TypeError: If ``indices`` is not iterable or contains non-integers.

        Example::

            >>> vocab = CharVocab()
            >>> vocab.decode([vocab.BOS, 26, 18, 21, 7, 22, vocab.EOS, vocab.PAD])
            'words'
        """
        _require_iterable(indices, name="indices")

        _SKIP: frozenset[int] = frozenset({self.PAD, self.BOS, self.UNK})
        chars: list[str] = []

        for idx in indices:
            if not isinstance(idx, int):
                raise TypeError(
                    f"All indices must be int, got {type(idx).__name__!r} "
                    f"at position {len(chars)}"
                )
            if idx == self.EOS:
                break
            if idx not in _SKIP:
                # Fall back to "?" for indices outside the known range rather
                # than raising, to tolerate model outputs with rare OOB values.
                chars.append(self._idx2char.get(idx, "?"))

        return "".join(chars)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the vocabulary size (mirrors ``vocab_size``)."""
        return self.vocab_size

    def __contains__(self, item: object) -> bool:
        """Support ``'a' in vocab`` and ``4 in vocab`` membership tests."""
        if isinstance(item, str):
            return item in self._char2idx
        if isinstance(item, int):
            return item in self._idx2char
        return False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"vocab_size={self.vocab_size}, "
            f"PAD={self.PAD}, BOS={self.BOS}, "
            f"EOS={self.EOS}, UNK={self.UNK})"
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _dedupe(chars: object) -> list[str]:
    """Return a deduplicated list preserving first-occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for ch in chars:  # type: ignore[union-attr]
        if ch not in seen:
            out.append(ch)
            seen.add(ch)
    return out


def _require_str(value: object, *, name: str) -> None:
    """Raise ``ValueError`` / ``TypeError`` if *value* is not a plain ``str``."""
    if value is None:
        raise ValueError(f"{name} must be a str, got None")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__!r}")


def _require_iterable(value: object, *, name: str) -> None:
    """Raise ``TypeError`` if *value* is not iterable."""
    if not hasattr(value, "__iter__"):
        raise TypeError(
            f"{name} must be iterable, got {type(value).__name__!r}"
        )
