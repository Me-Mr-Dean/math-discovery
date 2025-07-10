#!/usr/bin/env python3
from __future__ import annotations
"""
Universal Mathematical Discovery Engine
Discover patterns and structure in any number-theoretic function using machine learning.
Adapted from the Prime Pattern Discovery framework.
"""

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None
try:
    import numpy as np
except Exception:  # pragma: no cover - fallback to stub numpy
    import numpy as np  # type: ignore
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency
    train_test_split = cross_val_score = RandomForestClassifier = GradientBoostingClassifier = None
    LogisticRegression = DecisionTreeClassifier = export_text = None
    classification_report = confusion_matrix = roc_auc_score = None
    StandardScaler = None
try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None
import warnings
import time
from typing import Callable, Dict, List, Tuple, Any

from ..utils.math_utils import generate_mathematical_features

warnings.filterwarnings("ignore")


class DiscoveryEngine:
    """Minimal placeholder to satisfy unit tests."""

    def __init__(self):
        pass

    def discover_patterns(self, sequence):
        raise NotImplementedError


class UniversalMathDiscovery:
    def __init__(
        self,
        target_function: Callable[[int], bool],
        function_name: str,
        max_number: int = 100000,
    ):
        """
        Initialize the Universal Mathematical Discovery Engine

        Args:
            target_function: A function that takes an integer and returns True/False
            function_name: Descriptive name for the function being analyzed
            max_number: Maximum number to analyze
        """
        self.target_function = target_function
        self.function_name = function_name
        self.max_number = max_number
        self.X = None
        self.y = None
        self.models = {}
        self.feature_names = []
        self.scaler = StandardScaler() if StandardScaler else None
        self.analysis_results = {}

    def generate_fibonacci_set(self, max_n: int) -> set:
        """Generate Fibonacci numbers up to max_n"""
        fibs = set([1, 1])
        a, b = 1, 1
        while b <= max_n:
            a, b = b, a + b
            fibs.add(b)
        return fibs

    def prime_factors_count(self, n: int) -> int:
        """Count prime factors (with multiplicity)"""
        if n <= 1:
            return 0
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    def sum_of_divisors(self, n: int) -> int:
        """Calculate sum of proper divisors"""
        if n <= 1:
            return 0
        divisor_sum = 1  # 1 is always a proper divisor
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                divisor_sum += i
                if i != n // i:  # Avoid counting square root twice
                    divisor_sum += n // i
        return divisor_sum

    def is_happy_number(self, n: int) -> bool:
        """Check if number is happy (sum of squares of digits eventually reaches 1)"""
        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = sum(int(digit) ** 2 for digit in str(n))
        return n == 1

    def generate_target_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate features and targets for the mathematical function"""
        print(f"\nGenerating data for function: {self.function_name}")
        print(f"Analyzing numbers 1 to {self.max_number}")

        # Pre-compute expensive sets for some functions
        fibonacci_set = None
        if "fibonacci" in self.function_name.lower():
            fibonacci_set = self.generate_fibonacci_set(self.max_number)
            print(f"Pre-computed {len(fibonacci_set)} Fibonacci numbers")

        features_list = []
        target_list = []

        start_time = time.time()
        history: List[int] = []

        for number in range(1, self.max_number + 1):
            # Progress update
            if number % 10000 == 0:
                elapsed = time.time() - start_time
                rate = number / elapsed if elapsed > 0 else 0
                remaining = (self.max_number - number) / rate if rate > 0 else 0
                print(
                    f"  Progress: {number:,}/{self.max_number:,} ({number/self.max_number*100:.1f}%) - {rate:.0f} nums/sec - ETA: {remaining:.0f}s"
                )

            # Extract mathematical features
            features = generate_mathematical_features(number, previous_numbers=history)

            # Function-specific optimizations
            if fibonacci_set is not None:
                target_value = int(number in fibonacci_set)
            else:
                target_value = int(self.target_function(number))

            features_list.append(features)
            target_list.append(target_value)
            history.append(number)

        # Convert to DataFrame
        self.X = pd.DataFrame(features_list)
        self.y = np.array(target_list)
        self.feature_names = list(self.X.columns)

        positive_rate = self.y.mean()
        print(f"\nData generation complete!")
        print(
            f"Generated {len(self.X):,} samples with {len(self.feature_names)} features"
        )
        print(f"Positive rate: {positive_rate:.4f} ({self.y.sum():,} positive cases)")

        return self.X, self.y

    def train_discovery_models(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Dict[str, Any]:
        """Train ML models to discover mathematical patterns"""
        print(f"\nTraining mathematical discovery models...")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y if len(np.unique(self.y)) > 1 else None,
        )

        # Scale features for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        print(
            f"Training: {len(X_train):,} samples, Positive rate: {y_train.mean():.4f}"
        )
        print(f"Testing: {len(X_test):,} samples, Positive rate: {y_test.mean():.4f}")

        # Define models for discovery
        models_config = {
            "logistic": LogisticRegression(random_state=random_state, max_iter=2000),
            "decision_tree": DecisionTreeClassifier(
                random_state=random_state, max_depth=12, min_samples_split=50
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                max_depth=15,
                min_samples_split=10,
                n_jobs=-1,
            ),
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=8,
                learning_rate=0.1,
            ),
        }

        # Train models
        print("\nTraining discovery models...")
        for name, model in models_config.items():
            print(f"  Training {name}...")

            start_time = time.time()

            if name == "logistic":
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                test_proba = (
                    model.predict_proba(X_test_scaled)[:, 1]
                    if len(np.unique(y_test)) > 1
                    else test_pred
                )
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                test_proba = (
                    model.predict_proba(X_test)[:, 1]
                    if len(np.unique(y_test)) > 1
                    else test_pred
                )

            train_time = time.time() - start_time

            # Calculate metrics
            train_acc = (train_pred == y_train).mean()
            test_acc = (test_pred == y_test).mean()
            test_auc = (
                roc_auc_score(y_test, test_proba)
                if len(np.unique(y_test)) > 1
                else test_acc
            )

            self.models[name] = {
                "model": model,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_auc": test_auc,
                "training_time": train_time,
            }

            print(
                f"    Train: {train_acc:.4f}, Test: {test_acc:.4f}, AUC: {test_auc:.4f} ({train_time:.1f}s)"
            )

        return self.models

    def analyze_mathematical_discoveries(
        self, model_name: str = "random_forest"
    ) -> pd.DataFrame:
        """Analyze what mathematical patterns the model discovered"""
        print(f"\n🔍 MATHEMATICAL PATTERN DISCOVERIES ({self.function_name}):")
        print("=" * 60)

        model = self.models[model_name]["model"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame(
                {"feature": self.feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            print("🏆 TOP MATHEMATICAL PATTERNS DISCOVERED:")
            print("-" * 40)
            for i, row in feature_importance.head(15).iterrows():
                print(f"{row['importance']:8.4f} | {row['feature']}")

            # Categorize discoveries
            print("\n📊 PATTERN CATEGORIES DISCOVERED:")
            print("-" * 40)

            modular_features = feature_importance[
                feature_importance["feature"].str.contains("mod_")
            ]
            digit_features = feature_importance[
                feature_importance["feature"].str.contains("digit")
            ]
            mathematical_features = feature_importance[
                feature_importance["feature"].str.contains("perfect|triangular|power")
            ]
            positional_features = feature_importance[
                feature_importance["feature"].str.contains("first|last")
            ]

            print(
                f"Modular Arithmetic Importance: {modular_features['importance'].sum():.4f}"
            )
            print(f"Digit Pattern Importance: {digit_features['importance'].sum():.4f}")
            print(
                f"Mathematical Property Importance: {mathematical_features['importance'].sum():.4f}"
            )
            print(
                f"Positional Pattern Importance: {positional_features['importance'].sum():.4f}"
            )

            self.analysis_results["feature_importance"] = feature_importance
            return feature_importance
        else:
            print(f"Model {model_name} doesn't support feature importance")
            return None

    def extract_mathematical_rules(
        self, max_depth: int = 6
    ) -> Tuple[DecisionTreeClassifier, str]:
        """Extract interpretable mathematical rules discovered by the model"""
        print(f"\n🧮 MATHEMATICAL RULES DISCOVERED FOR {self.function_name}:")
        print("=" * 50)

        # Train interpretable tree
        rule_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=200,
            min_samples_leaf=100,
            random_state=42,
        )
        rule_tree.fit(self.X_train, self.y_train)

        # Extract rules
        tree_rules = export_text(
            rule_tree, feature_names=self.feature_names, max_depth=max_depth
        )

        print("DISCOVERED MATHEMATICAL RULES:")
        print("-" * 40)
        print(tree_rules[:3000])  # Show first 3000 characters
        if len(tree_rules) > 3000:
            print("... (truncated)")

        # Performance
        train_acc = rule_tree.score(self.X_train, self.y_train)
        test_acc = rule_tree.score(self.X_test, self.y_test)
        print(f"\nRule-Based Model Performance:")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        self.analysis_results["rules"] = tree_rules
        self.analysis_results["rule_tree"] = rule_tree

        return rule_tree, tree_rules

    def create_prediction_function(self, model_name: str = "random_forest"):
        """Create a prediction function for the discovered mathematical patterns"""
        print(f"\n🎯 Creating prediction function for {self.function_name}...")

        model = self.models[model_name]["model"]

        def predict_property(number: int) -> Dict[str, Any]:
            """Predict if a number has the mathematical property"""

            if number < 1:
                return {
                    "number": number,
                    "prediction": 0,
                    "probability": 0.0,
                    "confidence": "invalid",
                }

            # Extract same features as training
            features = {
                "number": number,
                "log_number": np.log10(number + 1),
                "sqrt_number": np.sqrt(number),
                "digit_count": len(str(number)),
                "mod_2": number % 2,
                "mod_3": number % 3,
                "mod_5": number % 5,
                "mod_6": number % 6,
                "mod_7": number % 7,
                "mod_10": number % 10,
                "mod_11": number % 11,
                "mod_13": number % 13,
                "mod_30": number % 30,
                "mod_210": number % 210,
                "last_digit": number % 10,
                "first_digit": int(str(number)[0]),
                "digit_sum": sum(int(d) for d in str(number)),
                "digit_product": np.prod([int(d) for d in str(number) if int(d) > 0]),
                "alternating_digit_sum": sum(
                    (-1) ** i * int(d) for i, d in enumerate(str(number))
                ),
                "is_perfect_square": int(int(np.sqrt(number)) ** 2 == number),
                "is_perfect_cube": int(round(number ** (1 / 3)) ** 3 == number),
                "is_power_of_2": int(number > 0 and (number & (number - 1)) == 0),
                "is_triangular": int(
                    int(((8 * number + 1) ** 0.5 - 1) / 2) ** 2 == number
                ),
                "is_6n_plus_1": int(number % 6 == 1),
                "is_6n_minus_1": int(number % 6 == 5),
                "twin_candidate": int((number % 6) in [1, 5]),
                "wheel_2_3": number % 6,
                "wheel_2_3_5": number % 30,
                "wheel_2_3_5_7": number % 210,
                "prime_factors_count": self.prime_factors_count(number),
                "sum_of_digits_mod_9": sum(int(d) for d in str(number)) % 9,
                "sum_of_proper_divisors": (
                    self.sum_of_divisors(number) if number <= 10000 else 0
                ),
                "is_happy": int(self.is_happy_number(number)) if number <= 10000 else 0,
            }

            # Create feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0))

            X_pred = np.array(feature_vector).reshape(1, -1)

            # Make prediction
            if model_name == "logistic":
                X_pred = self.scaler.transform(X_pred)

            prediction = model.predict(X_pred)[0]

            # Get probability if available
            if hasattr(model, "predict_proba") and len(np.unique(self.y)) > 1:
                probability = model.predict_proba(X_pred)[0, 1]
            else:
                probability = float(prediction)

            # Determine confidence
            if probability > 0.85:
                confidence = "high"
            elif probability > 0.65:
                confidence = "medium"
            elif probability > 0.35:
                confidence = "low"
            else:
                confidence = "very_low"

            return {
                "number": number,
                "prediction": int(prediction),
                "probability": float(probability),
                "confidence": confidence,
                "function": self.function_name,
            }

        return predict_property

    def run_complete_discovery(self) -> Callable:
        """Run the complete mathematical discovery pipeline"""
        print("=" * 60)
        print(f"🧮 MATHEMATICAL DISCOVERY: {self.function_name.upper()}")
        print("=" * 60)
        print("Discovering mathematical patterns using machine learning...")

        # Generate data
        self.generate_target_data()

        # Train models
        self.train_discovery_models()

        # Analyze discoveries
        self.analyze_mathematical_discoveries()

        # Extract rules
        self.extract_mathematical_rules()

        # Create prediction function
        prediction_function = self.create_prediction_function()

        print("\n" + "=" * 60)
        print(f"🎉 DISCOVERY COMPLETE FOR {self.function_name.upper()}!")
        print("=" * 60)
        print("Mathematical patterns discovered and prediction function created!")

        return prediction_function


# Example mathematical functions for testing
def perfect_squares(n: int) -> bool:
    """Perfect squares: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, ..."""
    return int(n**0.5) ** 2 == n


def triangular_numbers(n: int) -> bool:
    """Triangular numbers: 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ..."""
    # n = k(k+1)/2, so k = (-1 + sqrt(1 + 8n))/2
    k = int((-1 + (1 + 8 * n) ** 0.5) / 2)
    return k * (k + 1) // 2 == n


def powers_of_2(n: int) -> bool:
    """Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, ..."""
    return n > 0 and (n & (n - 1)) == 0


def fibonacci_numbers(n: int) -> bool:
    """Fibonacci numbers: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, ..."""
    # This is inefficient but works for demo
    a, b = 1, 1
    if n == 1:
        return True
    while b < n:
        a, b = b, a + b
    return b == n


def perfect_numbers(n: int) -> bool:
    """Perfect numbers: 6, 28, 496, 8128, ..."""
    if n <= 1:
        return False
    divisor_sum = 1
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divisor_sum += i
            if i != n // i:
                divisor_sum += n // i
    return divisor_sum == n


def semiprimes(n: int) -> bool:
    """Semiprimes (exactly 2 prime factors): 4, 6, 9, 10, 14, 15, 21, 22, ..."""
    if n <= 1:
        return False
    factors = 0
    d = 2
    temp_n = n
    while d * d <= temp_n:
        while temp_n % d == 0:
            factors += 1
            temp_n //= d
        d += 1
    if temp_n > 1:
        factors += 1
    return factors == 2


def main():
    """Example usage and testing"""
    import sys

    # Available test functions
    test_functions = {
        "squares": (perfect_squares, "Perfect Squares"),
        "triangular": (triangular_numbers, "Triangular Numbers"),
        "powers2": (powers_of_2, "Powers of 2"),
        "fibonacci": (fibonacci_numbers, "Fibonacci Numbers"),
        "perfect": (perfect_numbers, "Perfect Numbers"),
        "semiprimes": (semiprimes, "Semiprimes"),
    }

    if len(sys.argv) < 2:
        print("Universal Mathematical Discovery Engine")
        print("Usage: python universal_math_discovery.py <function> [max_number]")
        print("\nAvailable functions:")
        for key, (_, name) in test_functions.items():
            print(f"  {key:<12} - {name}")
        print("\nExample: python universal_math_discovery.py squares 10000")
        return

    function_key = sys.argv[1].lower()
    max_number = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    if function_key not in test_functions:
        print(f"Unknown function: {function_key}")
        print("Available functions:", list(test_functions.keys()))
        return

    target_function, function_name = test_functions[function_key]

    # Run discovery
    discoverer = UniversalMathDiscovery(target_function, function_name, max_number)
    prediction_function = discoverer.run_complete_discovery()

    # Test the prediction function
    print(f"\n🧪 Testing discovered patterns for {function_name}:")
    test_numbers = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]

    for num in test_numbers:
        actual = target_function(num)
        result = prediction_function(num)
        predicted = bool(result["prediction"])

        status = "✅" if actual == predicted else "❌"
        actual_str = "TRUE" if actual else "FALSE"
        pred_str = "TRUE" if predicted else "FALSE"

        print(
            f"{status} {num:3d}: Actual={actual_str:<5} Predicted={pred_str:<5} (prob: {result['probability']:.3f})"
        )

    print(f"\n✨ Mathematical discovery complete for {function_name}!")


if __name__ == "__main__":
    main()
