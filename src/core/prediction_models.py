"""Lightweight prediction models used in the tests."""

from __future__ import annotations

import math
from typing import Sequence


class SimpleLogisticModel:
    """Minimal logistic regression using gradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 1000) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights: list[float] | None = None
        self.bias: float = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        X_arr = [list(map(float, row)) for row in X]
        y_arr = [float(v) for v in y]

        samples = len(X_arr)
        features = len(X_arr[0]) if samples else 0
        self.weights = [0.0 for _ in range(features)]
        self.bias = 0.0

        for _ in range(self.epochs):
            preds = []
            for row in X_arr:
                linear = sum(w * x for w, x in zip(self.weights, row)) + self.bias
                preds.append(1.0 / (1.0 + math.exp(-linear)))

            for j in range(features):
                dw = sum((preds[i] - y_arr[i]) * X_arr[i][j] for i in range(samples)) / samples
                self.weights[j] -= self.lr * dw

            db = sum(preds[i] - y_arr[i] for i in range(samples)) / samples
            self.bias -= self.lr * db

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        if self.weights is None:
            raise ValueError("Model is not fitted")
        X_arr = [list(map(float, row)) for row in X]
        preds = []
        for row in X_arr:
            linear = sum(w * x for w, x in zip(self.weights, row)) + self.bias
            preds.append(1.0 / (1.0 + math.exp(-linear)))
        return preds

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        proba = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in proba]

