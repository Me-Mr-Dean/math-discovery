"""Utility functions for prefix-suffix matrix generation."""

from typing import Iterable, List

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def generate_prefix_suffix_matrix(
    numbers: Iterable[int], prefix_digits: int, suffix_digits: int
) -> "pd.DataFrame":
    """Create a prefix-suffix matrix from a collection of numbers.

    Each number is split into ``prefix_digits`` leading digits and
    ``suffix_digits`` trailing digits. The matrix contains a 1 when the
    combination ``prefix`` + ``suffix`` exists in ``numbers`` and 0 otherwise.

    Parameters
    ----------
    numbers:
        Iterable of integers to place in the matrix.
    prefix_digits:
        How many leading digits to use for the prefix (row index).
    suffix_digits:
        How many trailing digits to use for the suffix (column label).

    Returns
    -------
    pandas.DataFrame
        Binary matrix indexed by prefixes with suffix columns.
    """

    if pd is None:
        raise ImportError("pandas is required for generate_prefix_suffix_matrix")

    if prefix_digits <= 0 or suffix_digits <= 0:
        raise ValueError("prefix_digits and suffix_digits must be positive")

    prefixes: set[int] = set()
    suffixes: set[int] = set()
    entries: List[tuple[int, int]] = []

    for number in numbers:
        s = str(number)
        if len(s) < prefix_digits + suffix_digits:
            continue
        prefix = int(s[:prefix_digits])
        suffix = int(s[-suffix_digits:])
        prefixes.add(prefix)
        suffixes.add(suffix)
        entries.append((prefix, suffix))

    prefix_list = sorted(prefixes)
    suffix_list = sorted(suffixes)
    df = pd.DataFrame(
        0, index=prefix_list, columns=[str(s) for s in suffix_list]
    )
    df.index.name = f"prefix_{prefix_digits}d"
    df.columns.name = f"suffix_{suffix_digits}d"

    for prefix, suffix in entries:
        df.loc[prefix, str(suffix)] = 1

    return df
