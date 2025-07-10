"""Utility functions for embedding numerical sequences."""

from typing import Iterable, List, Sequence
import math

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore


def fourier_transform(sequence: Sequence[float], n_components: int | None = None) -> List[float]:
    """Compute a simple Fourier transform of a 1D sequence.

    Parameters
    ----------
    sequence:
        Input numeric sequence.
    n_components:
        Optional number of components to keep from the real/imaginary pairs.

    Returns
    -------
    list[float]
        Flattened list of real and imaginary coefficients.
    """
    seq = list(sequence)
    n = len(seq)
    coeffs: List[float] = []
    for k in range(n // 2 + 1):
        real = 0.0
        imag = 0.0
        for t, x in enumerate(seq):
            angle = 2 * math.pi * k * t / n
            real += x * math.cos(angle)
            imag -= x * math.sin(angle)
        coeffs.append(real)
        coeffs.append(imag)
    if n_components is not None:
        coeffs = coeffs[:n_components]
    return coeffs


def pca_transform(data: Sequence[Sequence[float]], n_components: int = 2) -> List[List[float]]:
    """Apply a simple PCA transform to a dataset.

    Parameters
    ----------
    data:
        Two-dimensional data matrix (samples x features).
    n_components:
        Number of principal components to return.

    Returns
    -------
    list[list[float]]
        Transformed data matrix.
    """
    matrix = [list(map(float, row)) for row in data]
    rows = len(matrix)
    if rows == 0:
        return []
    cols = len(matrix[0])

    # Center the data
    means = [sum(row[i] for row in matrix) / rows for i in range(cols)]
    centered = [[row[i] - means[i] for i in range(cols)] for row in matrix]

    if hasattr(np, "linalg"):
        arr = np.array(centered)
        cov = (arr.T @ arr) / max(rows - 1, 1)
        eig_vals, eig_vecs = np.linalg.eig(cov)  # type: ignore[attr-defined]
        order = np.argsort(eig_vals)[::-1]
        components = eig_vecs[:, order[:n_components]]
        transformed = arr @ components
        return transformed.tolist()

    # Fallback without numpy.linalg: return first n_components columns of centered data
    return [row[:n_components] for row in centered]

