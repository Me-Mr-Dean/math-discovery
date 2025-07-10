"""Simple helper utilities for validating numeric sequences."""

from __future__ import annotations

from typing import Sequence


def is_non_empty_numeric_sequence(seq: Sequence) -> bool:
    """Return ``True`` if ``seq`` contains numeric values and is not empty."""

    if not seq:
        return False
    return all(isinstance(x, (int, float)) for x in seq)


def has_unique_elements(seq: Sequence) -> bool:
    """Return ``True`` if ``seq`` has no repeated values."""

    return len(seq) == len(set(seq))


def is_increasing(seq: Sequence[float]) -> bool:
    """Return ``True`` if ``seq`` is strictly increasing."""

    if not seq:
        return False
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i + 1]:
            return False
    return True

