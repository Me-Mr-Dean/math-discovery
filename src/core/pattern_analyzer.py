"""Basic utilities for analysing simple numeric patterns."""

from __future__ import annotations

from typing import Optional, Sequence


class PatternAnalyzer:
    """Detect arithmetic or geometric patterns in sequences."""

    def common_difference(self, sequence: Sequence[float]) -> Optional[float]:
        if len(sequence) < 2:
            return None
        diffs = [sequence[i + 1] - sequence[i] for i in range(len(sequence) - 1)]
        return diffs[0] if len(set(diffs)) == 1 else None

    def common_ratio(self, sequence: Sequence[float]) -> Optional[float]:
        if len(sequence) < 2:
            return None
        ratios = []
        for i in range(len(sequence) - 1):
            if sequence[i] == 0:
                return None
            ratios.append(sequence[i + 1] / sequence[i])
        return ratios[0] if len(set(ratios)) == 1 else None

    def is_arithmetic(self, sequence: Sequence[float]) -> bool:
        return self.common_difference(sequence) is not None

    def is_geometric(self, sequence: Sequence[float]) -> bool:
        return self.common_ratio(sequence) is not None

