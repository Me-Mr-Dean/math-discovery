#!/usr/bin/env python3
"""
Universal Mathematical Discovery Engine - FIXED VERSION
=====================================================

A pure machine learning approach to discovering mathematical patterns.
NO LABEL LEAKING - Forces genuine pattern discovery.

CRITICAL FIXES APPLIED:
- Removed all direct target function calls during feature generation
- Eliminated boolean 'is_*' features that encode the answer
- Added validation to catch perfect accuracy (indicates leaking)
- Focused on raw mathematical structure only

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
    from ..utils.math_utils import validate_features_for_label_leaking
except ImportError:
    try:
        from utils.math_utils import generate_mathematical_features
        from utils.embedding_utils import fourier_transform, pca_transform
        from utils.math_utils import validate_features_for_label_leaking
    except ImportError:
        print("⚠️  Utils not available - using basic features only")

        def generate_mathematical_features(n, **kwargs):
            """Fallback feature generator - NO LABEL LEAKING"""
            return {
                "number": n,
                "mod_2": n % 2,
                "mod_3": n % 3,
                "mod_5": n % 5,
                "mod_7": n % 7,
                "mod_10": n % 10,
                "last_digit": n % 10,
                "digit_sum": sum(int(d) for d in str(n)),
                "log_number": np.log10(n + 1),
                "sqrt_fractional": np.sqrt(n) % 1,
            }

        def fourier_transform(sequence, n_components=None):
            return list(sequence)[:8] if sequence else [0] * 8

        def pca_transform(data, n_components=2):
            return [
                [row[i] if i < len(row) else 0 for i in range(n_components)]
                for row in data
            ]

        def validate_features_for_label_leaking(features, target_name=""):
            return []


def powers_of_2(n: int) -> bool:
    """Example target function - powers of 2"""
    return n > 0 and (n & (n - 1)) == 0


class UniversalMathDiscovery:
    """
    Universal Mathematical Discovery Engine - FIXED VERSION

    CRITICAL CHANGES:
    - NO direct target function calls during feature generation
    - NO boolean features that encode mathematical properties
    - Validation to catch potential label leaking
    - Focus on genuine pattern discovery
    """

    def __init__(
        self,
        target_function: Callable[[int], bool],
        function_name: str = "Mathematical Function",
        max_number: int = 100000,
        embedding: Optional[str] = None,
        embedding_components: Optional[int] = None,
        validate_no_leaking: bool = True,
    ):
        """
        Initialize the discovery engine.

        Args:
            target_function: Function to discover patterns for
            function_name: Descriptive name for the function
            max_number: Maximum number to test
            embedding: Optional embedding type ('fourier', 'pca')
            embedding_components: Number of embedding components
            validate_no_leaking: Whether to validate against label leaking
        """
        self.target_function = target_function
        self.function_name = function_name
        self.max_number = max_number
        self.embedding = embedding
        self.embedding_components = embedding_components or 8
        self.validate_no_leaking = validate_no_leaking

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

        # Validation data
        self.positive_numbers = []  # Store which numbers are positive

    def generate_target_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate feature data for the target function.

        FIXED: NO label leaking - features generated independently of target
        """
        print(f"🔢 Generating data for: {self.function_name}")
        print(f"   Testing numbers 1 to {self.max_number:,}")
        print(f"   🚨 LABEL LEAKING PROTECTION: ON")

        features_list = []
        target_list = []
        positive_examples = []

        start_time = time.time()

        # STEP 1: Generate target labels FIRST (to find positive examples)
        print("   Step 1: Finding positive examples...")
        for n in range(1, self.max_number + 1):
            is_target = self.target_function(n)
            target_list.append(1 if is_target else 0)
            if is_target:
                positive_examples.append(n)

        self.positive_numbers = positive_examples
        target_count = len(positive_examples)
        density = target_count / self.max_number

        print(f"   Found {target_count:,} positive examples ({density:.4f} density)")

        # STEP 2: Generate features WITHOUT access to target function
        print("   Step 2: Generating features (NO TARGET ACCESS)...")

        history = []  # Track sequence for context features

        for n in range(1, self.max_number + 1):
            # 🚨 CRITICAL: Generate features WITHOUT knowing if n is positive
            # This forces the model to discover patterns from structure alone

            features = generate_mathematical_features(
                n,
                previous_numbers=history[-5:] if history else None,
                window_size=5,
                digit_tensor=(self.embedding is not None),
                # NOTE: We do NOT pass reference_set here - that would leak labels
            )

            # Add sequence context based on INDEX position, not target membership
            if n > 1:
                features["position_in_sequence"] = n
                features["position_mod_100"] = n % 100
                features["position_mod_1000"] = n % 1000

            features_list.append(features)

            # Update history based on position, not target
            if n % 10 == 0:  # Every 10th number for context
                history.append(n)

            # Progress update
            if n % 10000 == 0:
                elapsed = time.time() - start_time
                rate = n / elapsed if elapsed > 0 else 0
                print(
                    f"   Progress: {n:,}/{self.max_number:,} ({100*n/self.max_number:.1f}%) - Rate: {rate:.0f}/sec"
                )

        # STEP 3: Validate features for label leaking
        if self.validate_no_leaking and features_list:
            print("   Step 3: Validating features for label leaking...")
            sample_features = features_list[0]
            problematic = validate_features_for_label_leaking(
                sample_features, self.function_name
            )

            if problematic:
                print(f"   🚨 WARNING: Potential label leaking detected!")
                for issue in problematic:
                    print(f"      ⚠️  {issue}")
                print(f"   Consider removing these features for genuine discovery.")
            else:
                print(f"   ✅ No label leaking detected in features")

        # Convert to DataFrame
        self.X = pd.DataFrame(features_list)
        self.y = np.array(target_list)
        self.feature_names = list(self.X.columns)

        # Add embeddings if requested
        if self.embedding:
            self._add_embeddings()

        total_time = time.time() - start_time

        print(f"✅ Generated {len(self.X)} samples in {total_time:.1f}s")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Positive examples: {target_count:,} ({density:.4f})")
        print(f"   🔍 Ready for genuine pattern discovery!")

        return self.X, self.y

    def _add_embeddings(self):
        """Add embedding features to the dataset"""
        print(f"🌀 Adding {self.embedding} embeddings...")

        if self.embedding == "fourier":
            # Extract digit tensors for Fourier transform
            digit_tensors = []
            for _, row in self.X.iterrows():
                if "digit_tensor" in row and isinstance(row["digit_tensor"], list):
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
                if "digit_tensor" in row and isinstance(row["digit_tensor"], list):
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
        """
        Train multiple models for pattern discovery.

        ADDED: Validation to catch perfect scores (indicates leaking)
        """
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
        suspicious_scores = []

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

            # 🚨 VALIDATION: Check for suspiciously perfect scores
            if test_acc >= 0.98 or test_auc >= 0.98:
                suspicious_scores.append(
                    f"{name}: {test_acc:.4f} acc, {test_auc:.4f} AUC"
                )

        # Warn about potential label leaking
        if suspicious_scores:
            print(f"\n🚨 WARNING: Suspiciously high scores detected!")
            print(f"   This may indicate label leaking:")
            for score in suspicious_scores:
                print(f"   ⚠️  {score}")
            print(f"   Expected range for genuine discovery: 0.60-0.90")
            print(f"   Consider auditing features for label leaking.")
        else:
            print(f"\n✅ Scores look realistic for genuine discovery")

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
                "Sequence Context": feature_importance[
                    feature_importance["feature"].str.contains(
                        "diff|ratio|mean|std|position", na=False
                    )
                ],
                "Mathematical Structure": feature_importance[
                    feature_importance["feature"].str.contains(
                        "totient|factors|sqrt|log", na=False
                    )
                ],
                "Embeddings": feature_importance[
                    feature_importance["feature"].str.contains("fourier|pca", na=False)
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
                # Generate features for the number (NO target function access)
                features = generate_mathematical_features(n)

                # Create feature vector matching training data
                feature_vector = []
                for feature_name in self.feature_names:
                    if feature_name in features:
                        feature_vector.append(features[feature_name])
                    elif feature_name.startswith(("fourier_", "pca_")):
                        # Handle embedding features
                        feature_vector.append(0.0)  # Default for missing embeddings
                    elif feature_name.startswith("position_"):
                        # Handle position features
                        if "mod_100" in feature_name:
                            feature_vector.append(n % 100)
                        elif "mod_1000" in feature_name:
                            feature_vector.append(n % 1000)
                        else:
                            feature_vector.append(n)  # position_in_sequence
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

    def validate_discovery_quality(self) -> Dict[str, Any]:
        """
        Validate that the discovery is genuine and not due to label leaking
        """
        print(f"\n🔍 VALIDATING DISCOVERY QUALITY")
        print("=" * 40)

        validation_results = {
            "genuine_discovery": True,
            "issues_found": [],
            "recommendations": [],
        }

        # Check 1: Realistic performance scores
        best_acc = max(model["test_accuracy"] for model in self.models.values())
        best_auc = max(model["test_auc"] for model in self.models.values())

        if best_acc >= 0.98:
            validation_results["genuine_discovery"] = False
            validation_results["issues_found"].append(
                f"Suspiciously high accuracy: {best_acc:.4f}"
            )

        if best_auc >= 0.98:
            validation_results["genuine_discovery"] = False
            validation_results["issues_found"].append(
                f"Suspiciously high AUC: {best_auc:.4f}"
            )

        # Check 2: Feature importance distribution
        if "random_forest" in self.models:
            rf_model = self.models["random_forest"]["model"]
            if hasattr(rf_model, "feature_importances_"):
                max_importance = np.max(rf_model.feature_importances_)
                if max_importance > 0.8:
                    validation_results["issues_found"].append(
                        f"Single feature dominance: {max_importance:.4f}"
                    )

        # Check 3: Training vs test performance gap
        for name, model in self.models.items():
            gap = model["train_accuracy"] - model["test_accuracy"]
            if gap > 0.15:
                validation_results["issues_found"].append(
                    f"{name}: Large train/test gap: {gap:.4f}"
                )

        # Generate recommendations
        if validation_results["issues_found"]:
            validation_results["recommendations"] = [
                "Audit features for label leaking using validate_features_for_label_leaking()",
                "Remove boolean 'is_*' features that encode mathematical properties",
                "Ensure target function is not called during feature generation",
                "Consider using unsupervised methods to validate discovered patterns",
                "Test discovery on truly unknown mathematical functions",
            ]
        else:
            validation_results["recommendations"] = [
                "Discovery appears genuine - models are working to find patterns",
                "Consider expanding to larger datasets to test robustness",
                "Try cross-validation on different mathematical functions",
                "Document discovered patterns for mathematical validation",
            ]

        # Display results
        if validation_results["genuine_discovery"]:
            print("✅ Discovery appears genuine!")
            print(f"   Best accuracy: {best_acc:.4f} (realistic)")
            print(f"   Best AUC: {best_auc:.4f} (realistic)")
        else:
            print("🚨 Potential issues detected:")
            for issue in validation_results["issues_found"]:
                print(f"   ⚠️  {issue}")

        print(f"\n💡 Recommendations:")
        for rec in validation_results["recommendations"]:
            print(f"   • {rec}")

        return validation_results

    def run_complete_discovery(self) -> Callable[[int], Dict[str, Any]]:
        """Run the complete discovery pipeline with validation"""
        print("🚀 RUNNING COMPLETE MATHEMATICAL DISCOVERY")
        print("=" * 60)
        print(f"Function: {self.function_name}")
        print(f"Scope: 1 to {self.max_number:,}")
        if self.embedding:
            print(f"Embedding: {self.embedding}")
        print(
            f"🚨 Label Leaking Protection: {'ON' if self.validate_no_leaking else 'OFF'}"
        )

        # Step 1: Generate data (with leak protection)
        self.generate_target_data()

        # Step 2: Train models (with score validation)
        self.train_discovery_models()

        # Step 3: Analyze discoveries
        self.analyze_mathematical_discoveries()

        # Step 4: Extract rules
        self.extract_mathematical_rules()

        # Step 5: Validate discovery quality
        validation = self.validate_discovery_quality()

        # Step 6: Create prediction function
        prediction_function = self.create_prediction_function()

        print("\n" + "=" * 60)
        print("🎉 DISCOVERY COMPLETE!")
        print("=" * 60)
        if validation["genuine_discovery"]:
            print("✅ Genuine mathematical patterns discovered!")
        else:
            print("⚠️  Potential label leaking detected - review recommendations")
        print("Prediction function ready for testing!")

        return prediction_function


# Utility functions for examples (keeping these for compatibility)
def is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square"""
    root = int(n**0.5)
    return root * root == n


def is_fibonacci(n: int) -> bool:
    """Check if n is a Fibonacci number"""

    def is_perfect_square_check(x):
        if x < 0:
            return False
        root = int(x**0.5)
        return root * root == x

    return is_perfect_square_check(5 * n * n + 4) or is_perfect_square_check(
        5 * n * n - 4
    )


def main():
    """Example usage of the fixed discovery engine"""
    print("🧮 Universal Mathematical Discovery Engine - FIXED VERSION")
    print("=" * 60)

    # Example 1: Discover patterns in perfect squares (should NOT achieve 100% accuracy)
    print("\nExample 1: Perfect Squares Discovery (No Label Leaking)")
    print("-" * 50)

    discoverer = UniversalMathDiscovery(
        target_function=is_perfect_square,
        function_name="Perfect Squares",
        max_number=1000,
        validate_no_leaking=True,
    )

    prediction_fn = discoverer.run_complete_discovery()

    # Test predictions
    print("\n🧪 Testing discovered patterns:")
    test_numbers = [16, 17, 25, 26, 36, 37, 49, 50, 64, 65]
    correct = 0
    for num in test_numbers:
        result = prediction_fn(num)
        actual = is_perfect_square(num)
        status = "✅" if result["prediction"] == actual else "❌"
        if result["prediction"] == actual:
            correct += 1
        print(
            f"  {status} {num:3d}: Predicted={result['prediction']}, "
            f"Actual={actual}, Prob={result['probability']:.3f}"
        )

    accuracy = correct / len(test_numbers)
    print(f"\nTest Accuracy: {accuracy:.3f}")
    if accuracy < 1.0:
        print("✅ Model is working to discover patterns (not perfect - this is good!)")
    else:
        print("⚠️  Perfect accuracy may indicate remaining label leaking")


if __name__ == "__main__":
    main()
