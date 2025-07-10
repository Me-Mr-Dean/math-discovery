"""General numerical sequence generation utilities."""

from __future__ import annotations

from typing import List


class SequenceGenerator:
    """Generate common mathematical sequences."""

    def arithmetic_sequence(self, start: int, diff: int, length: int) -> List[int]:
        if length <= 0:
            return []
        return [start + diff * i for i in range(length)]

    def geometric_sequence(self, start: int, ratio: int, length: int) -> List[int]:
        if length <= 0:
            return []
        seq = [start]
        for _ in range(1, length):
            seq.append(seq[-1] * ratio)
        return seq

    def fibonacci(self, n: int) -> List[int]:
        if n <= 0:
            return []
        if n == 1:
            return [1]
        seq = [1, 1]
        while len(seq) < n:
            seq.append(seq[-1] + seq[-2])
        return seq[:n]

