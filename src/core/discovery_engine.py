#!/usr/bin/env python3
"""
Universal Mathematical Discovery Engine
=====================================

A pure machine learning approach to discovering mathematical patterns.
No hard-coded mathematical knowledge - pure pattern extraction from data.

This is the core discovery engine that all other components depend on.

Author: Mathematical Pattern Discovery Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Callable, List, Dict, Any, Tuple, Optional
from pathlib import Path
import time

warnings.filterwarnings("ignore")

# Try to import our utilities, with fallbacks
try:
    from ..utils.math_utils import generate_mathematical_features
    from ..utils.embedding_utils import fourier_transform, pca_transform
except ImportError:
    try:
        from utils.math_utils import generate_mathematical_features
        from utils.embedding_utils import fourier_transform, pca_transform
    except ImportError:
        print("⚠️  Utils not available - using basic features only")

        def generate_mathematical_features(n, **kwargs):
            """Fallback feature generator"""
            return {
                "number": n,
                "mod_2": n % 2,
                "mod_3": n % 3,
                "mod_5": n % 5,
                "mod_7": n % 7,
                "mod_10": n % 10,
                "last_digit": n % 10,
                "digit_sum": sum(int(d) for d in str(n)),
                "is_prime_candidate": int(n % 6 in [1, 5]) if n > 3 else 0,
            }

        def fourier_transform(sequence, n_components=None):
            """Fallback fourier transform"""
            return list(sequence)[:8] if sequence else [0] * 8

        def pca_transform(data, n_components=2):
            """Fallback PCA"""
            return [
                [row[i] if i < len(row) else 0 for i in range(n_components)]
                for row in data
            ]


def powers_of_2(n: int) -> bool:
    """Example target function - powers of 2"""
    return n > 0 and (n & (n - 1)) == 0


class UniversalMathDiscovery:
    """
    Universal Mathematical Discovery Engine

    Pure machine learning approach to mathematical pattern discovery.
    No hard-coded mathematical knowledge required.
    """

    def __init__(
        self,
        target_function: Callable[[int], bool],
        function_name: str = "Mathematical Function",
        max_number: int = 100000,
        embedding: Optional[str] = None,
        embedding_components: Optional[int] = None,
    ):
        """
        Initialize the discovery engine.

        Args:
            target_function: Function to discover patterns for
            function_name: Descriptive name for the function
            max_number: Maximum number to test
            embedding: Optional embedding type ('fourier', 'pca')
            embedding_components: Number of embedding components
        """
        self.target_function = target_function
        self.function_name = function_name
        self.max_number = max_number
        self.embedding = embedding
        self.embedding_components = embedding_components or 8

        # Data containers
        self.X = None
        self.y = None
        self.feature_names = []
        self.models = {}
        self.scaler = StandardScaler()

        # Training data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def generate_target_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate feature data for the target function"""
        print(f"🔢 Generating data for: {self.function_name}")
        print(f"   Testing numbers 1 to {self.max_number:,}")

        features_list = []
        target_list = []
        history = []

        start_time = time.time()

        for n in range(1, self.max_number + 1):
            # Check if n satisfies the target function
            is_target = self.target_function(n)

            # Generate mathematical features
            features = generate_mathematical_features(
                n,
                previous_numbers=history[-5:] if history else None,
                window_size=5,
                digit_tensor=(self.embedding is not None),
            )

            features_list.append(features)
            target_list.append(1 if is_target else 0)

            if is_target:
                history.append(n)

            # Progress update
            if n % 10000 == 0:
                elapsed = time.time() - start_time
                rate = n / elapsed if elapsed > 0 else 0
                found = sum(target_list)
                density = found / n
                print(
                    f"   Progress: {n:,}/{self.max_number:,} ({100*n/self.max_number:.1f}%) "
                    f"- Found: {found:,} ({density:.4f}) - Rate: {rate:.0f}/sec"
                )

        # Convert to DataFrame
        self.X = pd.DataFrame(features_list)
        self.y = np.array(target_list)
        self.feature_names = list(self.X.columns)

        # Add embeddings if requested
        if self.embedding:
            self._add_embeddings()

        total_time = time.time() - start_time
        positive_count = sum(target_list)

        print(f"✅ Generated {len(self.X)} samples in {total_time:.1f}s")
        print(f"   Features: {len(self.feature_names)}")
        print(
            f"   Positive examples: {positive_count:,} ({positive_count/len(self.X):.4f})"
        )

        return self.X, self.y

    def _add_embeddings(self):
        """Add embedding features to the dataset"""
        print(f"🌀 Adding {self.embedding} embeddings...")

        if self.embedding == "fourier":
            # Extract digit tensors for Fourier transform
            digit_tensors = []
            for _, row in self.X.iterrows():
                if "digit_tensor" in row:
                    digit_tensors.append(row["digit_tensor"])
                else:
                    # Fallback: use digits of the number
                    digits = [int(d) for d in str(row["number"])]
                    digit_tensors.append(digits[:6] + [0] * max(0, 6 - len(digits)))

            # Apply Fourier transform
            fourier_features = [
                fourier_transform(tensor, self.embedding_components)
                for tensor in digit_tensors
            ]

            # Add to dataframe
            for i in range(self.embedding_components):
                self.X[f"fourier_{i}"] = [
                    f[i] if i < len(f) else 0 for f in fourier_features
                ]

        elif self.embedding == "pca":
            # Extract digit tensors for PCA
            digit_tensors = []
            for _, row in self.X.iterrows():
                if "digit_tensor" in row:
                    digit_tensors.append(row["digit_tensor"])
                else:
                    digits = [int(d) for d in str(row["number"])]
                    digit_tensors.append(digits[:6] + [0] * max(0, 6 - len(digits)))

            # Apply PCA
            pca_features = pca_transform(digit_tensors, self.embedding_components)

            # Add to dataframe
            for i in range(self.embedding_components):
                self.X[f"pca_{i}"] = [f[i] for f in pca_features]

        # Update feature names
        self.feature_names = list(self.X.columns)
        print(f"   Added {self.embedding} features, total: {len(self.feature_names)}")

    def train_discovery_models(self) -> Dict[str, Dict]:
        """Train multiple models for pattern discovery"""
        print(f"\n🤖 Training discovery models...")

        if self.X is None or self.y is None:
            print("   Generating data first...")
            self.generate_target_data()

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        print(f"   Training: {len(self.X_train):,} samples")
        print(f"   Testing: {len(self.X_test):,} samples")
        print(f"   Features: {len(self.feature_names)}")

        # Define models
        models_config = {
            "logistic": LogisticRegression(random_state=42, max_iter=2000),
            "decision_tree": DecisionTreeClassifier(
                random_state=42, max_depth=12, min_samples_split=50
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=15,
                min_samples_split=10,
                n_jobs=-1,
            ),
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=100, random_state=42, max_depth=8, learning_rate=0.1
            ),
        }

        # Train models
        for name, model in models_config.items():
            print(f"   Training {name}...")

            if name == "logistic":
                model.fit(X_train_scaled, self.y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                test_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
                test_proba = model.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
            train_acc = (train_pred == self.y_train).mean()
            test_acc = (test_pred == self.y_test).mean()
            test_auc = roc_auc_score(self.y_test, test_proba)

            self.models[name] = {
                "model": model,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_auc": test_auc,
            }

            print(
                f"     Train: {train_acc:.4f}, Test: {test_acc:.4f}, AUC: {test_auc:.4f}"
            )

        return self.models

    def analyze_mathematical_discoveries(
        self, model_name: str = "random_forest"
    ) -> Optional[pd.DataFrame]:
        """Analyze mathematical patterns discovered by the model"""
        print(f"\n🔍 Analyzing mathematical discoveries...")

        if model_name not in self.models:
            print(
                f"   Model {model_name} not found. Available: {list(self.models.keys())}"
            )
            return None

        model_info = self.models[model_name]
        model = model_info["model"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame(
                {"feature": self.feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            print(f"🏆 Top Mathematical Patterns Discovered:")
            print("-" * 50)
            for _, row in feature_importance.head(15).iterrows():
                print(f"  {row['importance']:8.4f} | {row['feature']}")

            # Categorize discoveries
            print(f"\n📊 Pattern Categories:")
            print("-" * 25)

            categories = {
                "Modular Arithmetic": feature_importance[
                    feature_importance["feature"].str.contains("mod_", na=False)
                ],
                "Digit Patterns": feature_importance[
                    feature_importance["feature"].str.contains("digit", na=False)
                ],
                "Prime-Related": feature_importance[
                    feature_importance["feature"].str.contains(
                        "prime|6n|twin", na=False
                    )
                ],
                "Embeddings": feature_importance[
                    feature_importance["feature"].str.contains("fourier|pca", na=False)
                ],
                "Sequence": feature_importance[
                    feature_importance["feature"].str.contains(
                        "diff|ratio|mean|std", na=False
                    )
                ],
            }

            for category, features in categories.items():
                if len(features) > 0:
                    importance_sum = features["importance"].sum()
                    print(
                        f"  {category}: {importance_sum:.4f} ({len(features)} features)"
                    )

            return feature_importance
        else:
            print(f"   Model {model_name} doesn't support feature importance")
            return None

    def extract_mathematical_rules(self, max_depth: int = 6) -> Tuple[Any, str]:
        """Extract interpretable mathematical rules"""
        print(f"\n📜 Extracting mathematical rules...")

        # Train interpretable tree
        rule_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=200,
            min_samples_leaf=100,
            random_state=42,
        )
        rule_tree.fit(self.X_train, self.y_train)

        # Extract rules as text
        tree_rules = export_text(rule_tree, feature_names=self.feature_names)

        # Performance
        train_acc = rule_tree.score(self.X_train, self.y_train)
        test_acc = rule_tree.score(self.X_test, self.y_test)

        print(f"Rule Performance:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

        print(f"\nDiscovered Rules (excerpt):")
        print("-" * 30)
        print(tree_rules[:1000] + "..." if len(tree_rules) > 1000 else tree_rules)

        return rule_tree, tree_rules

    def create_prediction_function(self, model_name: str = "random_forest"):
        """Create a prediction function using the trained model"""
        print(f"\n🎯 Creating prediction function using {model_name}...")

        if model_name not in self.models:
            print(f"   Model {model_name} not found")
            return None

        model_info = self.models[model_name]
        model = model_info["model"]

        def predict_number(n: int) -> Dict[str, Any]:
            """Predict if a number matches the discovered pattern"""
            try:
                # Generate features for the number
                features = generate_mathematical_features(n)

                # Create feature vector matching training data
                feature_vector = []
                for feature_name in self.feature_names:
                    if feature_name in features:
                        feature_vector.append(features[feature_name])
                    elif feature_name.startswith(("fourier_", "pca_")):
                        # Handle embedding features
                        feature_vector.append(0.0)  # Default for missing embeddings
                    else:
                        feature_vector.append(0.0)  # Default for missing features

                X_pred = np.array(feature_vector).reshape(1, -1)

                # Scale if using logistic regression
                if model_name == "logistic":
                    X_pred = self.scaler.transform(X_pred)

                # Make prediction
                prediction = model.predict(X_pred)[0]
                probability = model.predict_proba(X_pred)[0, 1]

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
                    "number": n,
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "confidence": confidence,
                    "method": f"ml_discovery_{model_name}",
                }

            except Exception as e:
                return {
                    "number": n,
                    "prediction": 0,
                    "probability": 0.0,
                    "confidence": "error",
                    "error": str(e),
                }

        return predict_number

    def run_complete_discovery(self) -> Callable[[int], Dict[str, Any]]:
        """Run the complete discovery pipeline"""
        print("🚀 RUNNING COMPLETE MATHEMATICAL DISCOVERY")
        print("=" * 60)
        print(f"Function: {self.function_name}")
        print(f"Scope: 1 to {self.max_number:,}")
        if self.embedding:
            print(f"Embedding: {self.embedding}")

        # Step 1: Generate data
        self.generate_target_data()

        # Step 2: Train models
        self.train_discovery_models()

        # Step 3: Analyze discoveries
        self.analyze_mathematical_discoveries()

        # Step 4: Extract rules
        self.extract_mathematical_rules()

        # Step 5: Create prediction function
        prediction_function = self.create_prediction_function()

        print("\n" + "=" * 60)
        print("🎉 DISCOVERY COMPLETE!")
        print("=" * 60)
        print("Mathematical patterns discovered and prediction function ready!")

        return prediction_function


# Utility functions for examples
def is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square"""
    root = int(n**0.5)
    return root * root == n


def is_fibonacci(n: int) -> bool:
    """Check if n is a Fibonacci number"""

    # A number is Fibonacci if one or both of (5*n^2 + 4) or (5*n^2 - 4) is a perfect square
    def is_perfect_square_check(x):
        if x < 0:
            return False
        root = int(x**0.5)
        return root * root == x

    return is_perfect_square_check(5 * n * n + 4) or is_perfect_square_check(
        5 * n * n - 4
    )


def main():
    """Example usage of the discovery engine"""
    print("🧮 Universal Mathematical Discovery Engine - Example")
    print("=" * 60)

    # Example 1: Discover patterns in perfect squares
    print("\nExample 1: Perfect Squares Discovery")
    print("-" * 40)

    discoverer = UniversalMathDiscovery(
        target_function=is_perfect_square,
        function_name="Perfect Squares",
        max_number=1000,
    )

    prediction_fn = discoverer.run_complete_discovery()

    # Test predictions
    print("\n🧪 Testing discovered patterns:")
    test_numbers = [16, 17, 25, 26, 36, 37, 49, 50, 64, 65]
    for num in test_numbers:
        result = prediction_fn(num)
        actual = is_perfect_square(num)
        status = "✅" if result["prediction"] == actual else "❌"
        print(
            f"  {status} {num:3d}: Predicted={result['prediction']}, "
            f"Actual={actual}, Prob={result['probability']:.3f}"
        )


if __name__ == "__main__":
    main()
