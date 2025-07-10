"""Utility classes for extracting features from integer sequences."""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..utils.math_utils import generate_mathematical_features


class FeatureExtractor:
    """Extract mathematical features from numeric sequences."""

    def extract_features(self, sequence: Sequence[int]) -> List[Dict[str, object]]:
        """Return feature dictionaries for each number in ``sequence``."""

        history: List[int] = []
        features: List[Dict[str, object]] = []
        for number in sequence:
            feat = generate_mathematical_features(number, previous_numbers=history)
            features.append(feat)
            history.append(number)
        return features

    def modular_features(self, sequence: Sequence[int], modulus: int) -> List[int]:
        """Return the modulo ``modulus`` of each element in ``sequence``."""

        if modulus <= 0:
            raise ValueError("modulus must be positive")
        return [n % modulus for n in sequence]

