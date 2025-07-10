import math
from typing import Iterable, Sequence, Any

__version__ = "0.0"

# Minimal numpy stubs for testing without full dependency

def log10(x: float) -> float:
    return math.log10(x)

def sqrt(x: float) -> float:
    return math.sqrt(x)

def prod(iterable: Iterable[float]) -> float:
    result = 1
    for val in iterable:
        result *= val
    return result

def mean(seq: Sequence[float]) -> float:
    return sum(seq) / len(seq) if seq else 0.0

def std(seq: Sequence[float]) -> float:
    if not seq:
        return 0.0
    m = mean(seq)
    return math.sqrt(sum((x - m) ** 2 for x in seq) / len(seq))

def array(seq: Sequence[Any], dtype: Any = None) -> list:
    return list(seq)

def unique(seq: Sequence[Any]) -> list:
    seen = set()
    uniq = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq

def isclose(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
