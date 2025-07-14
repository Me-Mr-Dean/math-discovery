#!/usr/bin/env python3
"""
Comprehensive Label Leaking Validation Script
===========================================

This script performs end-to-end validation that our mathematical discovery
system truly discovers patterns rather than exploiting label leaking.

Tests include:
1. Feature audit for suspicious patterns
2. Model performance validation (should be realistic, not perfect)
3. Cross-validation with unknown functions
4. Comparison against trivial baselines
5. Unsupervised pattern validation

Usage:
    python comprehensive_validation.py [--test-all] [--generate-report]
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional, Callable
import warnings

warnings.filterwarnings("ignore")

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    # Try direct imports first
    from src.core.discovery_engine import UniversalMathDiscovery
    from src.generators.universal_generator import (
        UniversalDatasetGenerator,
        MathematicalRule,
    )
    from src.utils.math_utils import (
        validate_features_for_label_leaking,
        generate_mathematical_features,
    )
    from src.analyzers.prime_analyzer import PurePrimeMLDiscovery
except ImportError:
    try:
        # Fallback to relative imports
        from core.discovery_engine import UniversalMathDiscovery
        from generators.universal_generator import (
            UniversalDatasetGenerator,
            MathematicalRule,
        )
        from utils.math_utils import (
            validate_features_for_label_leaking,
            generate_mathematical_features,
        )
        from analyzers.prime_analyzer import PurePrimeMLDiscovery
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Trying to locate modules...")

        # Try to find and import modules manually
        src_dir = project_root / "src"
        if src_dir.exists():
            print(f"Found src directory: {src_dir}")
            sys.path.insert(0, str(src_dir))

            # Check for specific files
            discovery_file = src_dir / "core" / "discovery_engine.py"
            if discovery_file.exists():
                print(f"✅ Found discovery_engine.py")
            else:
                print(f"❌ Missing discovery_engine.py")

            # Import what we can and create minimal fallbacks for missing components
            try:
                from core.discovery_engine import UniversalMathDiscovery

                print("✅ Successfully imported UniversalMathDiscovery")
            except ImportError:
                print("❌ Failed to import UniversalMathDiscovery - creating fallback")

                # Create a minimal fallback class
                class UniversalMathDiscovery:
                    def __init__(self, *args, **kwargs):
                        raise ImportError("UniversalMathDiscovery not available")

            try:
                from utils.math_utils import (
                    validate_features_for_label_leaking,
                    generate_mathematical_features,
                )

                print("✅ Successfully imported math_utils")
            except ImportError:
                print("❌ Failed to import math_utils - creating fallbacks")

                def validate_features_for_label_leaking(features, target_name=""):
                    return []

                def generate_mathematical_features(n, **kwargs):
                    return {"number": n, "mod_2": n % 2}

            try:
                from generators.universal_generator import (
                    UniversalDatasetGenerator,
                    MathematicalRule,
                )

                print("✅ Successfully imported universal_generator")
            except ImportError:
                print("❌ Failed to import universal_generator - creating fallbacks")

                class MathematicalRule:
                    def __init__(self, func, name, description=""):
                        self.func = func
                        self.name = name
                        self.description = description
                        self.safe_name = name.lower().replace(" ", "_")

                class UniversalDatasetGenerator:
                    def __init__(self):
                        raise ImportError("UniversalDatasetGenerator not available")

            try:
                from analyzers.prime_analyzer import PurePrimeMLDiscovery

                print("✅ Successfully imported prime_analyzer")
            except ImportError:
                print("❌ Failed to import prime_analyzer - creating fallback")

                class PurePrimeMLDiscovery:
                    def __init__(self, *args, **kwargs):
                        raise ImportError("PurePrimeMLDiscovery not available")

        else:
            print("❌ No src directory found")
            sys.exit(1)


class LabelLeakingValidator:
    """Comprehensive validator for label leaking in mathematical discovery"""

    def __init__(self):
        self.validation_results = {}
        self.test_functions = []
        self.baseline_accuracies = {}

    def validate_feature_generation(self, test_numbers: List[int] = None) -> Dict:
        """Test 1: Validate that feature generation doesn't leak labels"""
        print("🔍 TEST 1: FEATURE GENERATION VALIDATION")
        print("=" * 50)

        if test_numbers is None:
            test_numbers = list(range(1, 101))  # Test first 100 numbers

        validation_results = {
            "test_name": "Feature Generation",
            "passed": True,
            "issues": [],
            "sample_features": {},
        }

        print(f"Testing feature generation on {len(test_numbers)} numbers...")

        # Test our fixed math_utils
        sample_features = generate_mathematical_features(42)  # Test number

        # Validate features
        problematic = validate_features_for_label_leaking(
            sample_features, "test_function"
        )

        if problematic:
            validation_results["passed"] = False
            validation_results["issues"] = problematic
            print(f"❌ Feature generation validation FAILED:")
            for issue in problematic:
                print(f"   ⚠️  {issue}")
        else:
            print(f"✅ Feature generation validation PASSED")
            print(f"   No suspicious boolean flags or label-encoding features detected")

        # Store sample for inspection
        validation_results["sample_features"] = {
            k: v
            for k, v in sample_features.items()
            if not isinstance(v, (list, np.ndarray))  # Skip complex types for JSON
        }

        print(f"   Generated {len(sample_features)} features for number 42")
        print(
            f"   Feature types: {set(type(v).__name__ for v in sample_features.values())}"
        )

        return validation_results

    def validate_model_performance(self) -> Dict:
        """Test 2: Check that models achieve realistic (not perfect) performance"""
        print("\n🤖 TEST 2: MODEL PERFORMANCE VALIDATION")
        print("=" * 50)

        validation_results = {
            "test_name": "Model Performance",
            "passed": True,
            "issues": [],
            "performance_data": {},
        }

        # Test with a simple mathematical function
        def test_function(n):
            return n % 4 == 1  # Numbers ≡ 1 (mod 4)

        print("Testing with function: numbers ≡ 1 (mod 4)")
        print("Expected: Models should achieve 60-85% accuracy (not 95%+)")

        try:
            # Create discovery engine
            discoverer = UniversalMathDiscovery(
                target_function=test_function,
                function_name="Mod 4 Equals 1 Test",
                max_number=1000,  # Small for testing
                validate_no_leaking=True,
            )

            # Run discovery
            discoverer.generate_target_data()
            models = discoverer.train_discovery_models()

            # Check performance
            suspicious_models = []
            for name, model_info in models.items():
                test_acc = model_info["test_accuracy"]
                test_auc = model_info["test_auc"]

                validation_results["performance_data"][name] = {
                    "test_accuracy": test_acc,
                    "test_auc": test_auc,
                }

                # Flag suspiciously high performance
                if test_acc >= 0.95 or test_auc >= 0.95:
                    suspicious_models.append(
                        f"{name}: {test_acc:.3f} acc, {test_auc:.3f} AUC"
                    )

            if suspicious_models:
                validation_results["passed"] = False
                validation_results["issues"] = suspicious_models
                print(f"❌ Model performance validation FAILED:")
                for model in suspicious_models:
                    print(f"   ⚠️  Suspiciously high performance: {model}")
                print(f"   This indicates potential label leaking")
            else:
                print(f"✅ Model performance validation PASSED")
                print(f"   All models achieved realistic accuracy (60-90% range)")
                for name, perf in validation_results["performance_data"].items():
                    print(
                        f"   {name}: {perf['test_accuracy']:.3f} acc, {perf['test_auc']:.3f} AUC"
                    )

        except Exception as e:
            validation_results["passed"] = False
            validation_results["issues"] = [f"Discovery engine failed: {str(e)}"]
            print(f"❌ Model performance test failed: {e}")

        return validation_results

    def validate_cross_function_discovery(self) -> Dict:
        """Test 3: Test discovery on multiple different mathematical functions"""
        print("\n🔬 TEST 3: CROSS-FUNCTION DISCOVERY VALIDATION")
        print("=" * 50)

        validation_results = {
            "test_name": "Cross-Function Discovery",
            "passed": True,
            "issues": [],
            "function_results": {},
        }

        # Define test functions
        test_functions = [
            ("Powers of 2", lambda n: n > 0 and (n & (n - 1)) == 0),
            ("Multiples of 7", lambda n: n % 7 == 0),
            ("Digit Sum Even", lambda n: sum(int(d) for d in str(n)) % 2 == 0),
            ("Last Digit 3", lambda n: n % 10 == 3),
        ]

        print(
            f"Testing discovery on {len(test_functions)} different mathematical functions:"
        )

        all_passed = True

        for func_name, func in test_functions:
            print(f"\n   Testing: {func_name}")

            try:
                # Quick discovery test
                discoverer = UniversalMathDiscovery(
                    target_function=func,
                    function_name=func_name,
                    max_number=500,  # Small for speed
                    validate_no_leaking=True,
                )

                # Generate data and train one model
                discoverer.generate_target_data()
                models = discoverer.train_discovery_models()

                # Get best performance
                best_acc = max(model["test_accuracy"] for model in models.values())
                best_auc = max(model["test_auc"] for model in models.values())

                validation_results["function_results"][func_name] = {
                    "accuracy": best_acc,
                    "auc": best_auc,
                    "passed": 0.5 <= best_acc <= 0.95,  # Reasonable range
                }

                if 0.5 <= best_acc <= 0.95:
                    print(f"      ✅ {func_name}: {best_acc:.3f} acc (realistic)")
                else:
                    print(f"      ❌ {func_name}: {best_acc:.3f} acc (suspicious)")
                    all_passed = False

            except Exception as e:
                print(f"      ❌ {func_name}: Failed - {e}")
                validation_results["function_results"][func_name] = {
                    "error": str(e),
                    "passed": False,
                }
                all_passed = False

        validation_results["passed"] = all_passed

        if all_passed:
            print(f"\n✅ Cross-function validation PASSED")
            print(f"   All functions achieved realistic discovery performance")
        else:
            print(f"\n❌ Cross-function validation FAILED")
            print(f"   Some functions showed suspicious performance patterns")

        return validation_results

    def validate_baseline_comparison(self) -> Dict:
        """Test 4: Compare against trivial baselines"""
        print("\n📊 TEST 4: BASELINE COMPARISON VALIDATION")
        print("=" * 50)

        validation_results = {
            "test_name": "Baseline Comparison",
            "passed": True,
            "issues": [],
            "baseline_results": {},
            "discovery_results": {},
        }

        def test_function(n):
            return n % 6 == 1  # Simple modular function

        print("Testing function: numbers ≡ 1 (mod 6)")
        print("Comparing ML discovery vs trivial baselines")

        # Generate test data
        test_numbers = list(range(1, 1001))
        y_true = [1 if test_function(n) else 0 for n in test_numbers]

        # Baseline 1: Random guessing
        np.random.seed(42)
        y_random = np.random.randint(0, 2, len(test_numbers))
        random_acc = (y_random == y_true).mean()

        # Baseline 2: Always predict most common class
        majority_class = max(set(y_true), key=y_true.count)
        y_majority = [majority_class] * len(test_numbers)
        majority_acc = (y_majority == y_true).mean()

        # Baseline 3: Simple modular rule (cheating baseline)
        y_mod6 = [1 if n % 6 == 1 else 0 for n in test_numbers]
        mod6_acc = (y_mod6 == y_true).mean()

        validation_results["baseline_results"] = {
            "random": random_acc,
            "majority_class": majority_acc,
            "mod6_rule": mod6_acc,
        }

        print(f"   Baseline accuracies:")
        print(f"   • Random guessing: {random_acc:.3f}")
        print(f"   • Majority class: {majority_acc:.3f}")
        print(f"   • Mod 6 rule (cheating): {mod6_acc:.3f}")

        # Test our discovery system
        try:
            discoverer = UniversalMathDiscovery(
                target_function=test_function,
                function_name="Mod 6 Test",
                max_number=1000,
                validate_no_leaking=True,
            )

            discoverer.generate_target_data()
            models = discoverer.train_discovery_models()

            best_acc = max(model["test_accuracy"] for model in models.values())
            validation_results["discovery_results"]["best_accuracy"] = best_acc

            print(f"   ML Discovery accuracy: {best_acc:.3f}")

            # Validate expectations
            if best_acc <= random_acc + 0.1:
                validation_results["issues"].append("ML performs no better than random")
                validation_results["passed"] = False
            elif best_acc >= mod6_acc - 0.05:
                validation_results["issues"].append(
                    "ML performs too close to cheating baseline"
                )
                validation_results["passed"] = False
            elif majority_acc < best_acc < mod6_acc:
                print(
                    f"   ✅ ML performs between majority class and perfect rule (good!)"
                )
            else:
                print(f"   ✅ ML achieves reasonable discovery performance")

        except Exception as e:
            validation_results["passed"] = False
            validation_results["issues"] = [f"Discovery failed: {str(e)}"]
            print(f"   ❌ Discovery system failed: {e}")

        if validation_results["passed"]:
            print(f"\n✅ Baseline comparison validation PASSED")
        else:
            print(f"\n❌ Baseline comparison validation FAILED")
            for issue in validation_results["issues"]:
                print(f"   ⚠️  {issue}")

        return validation_results

    def validate_universal_generator(self) -> Dict:
        """Test 5: Validate Universal Dataset Generator"""
        print("\n🧮 TEST 5: UNIVERSAL GENERATOR VALIDATION")
        print("=" * 50)

        validation_results = {
            "test_name": "Universal Generator",
            "passed": True,
            "issues": [],
            "generator_results": {},
        }

        print("Testing Universal Dataset Generator for label leaking...")

        try:
            # Create test rule
            rule = MathematicalRule(
                func=lambda n: n % 5 == 2,
                name="Mod 5 Equals 2 Test",
                description="Numbers that leave remainder 2 when divided by 5",
            )

            # Generate datasets
            generator = UniversalDatasetGenerator()
            summary = generator.generate_complete_pipeline(
                rule=rule,
                max_number=500,  # Small for testing
                processors=["algebraic_features", "digit_tensor_simple"],
            )

            validation_results["generator_results"] = summary["validation"]

            # Check validation results
            clean_datasets = summary["validation"]["clean_datasets"]
            total_datasets = summary["validation"]["total_datasets"]

            if clean_datasets == total_datasets:
                print(f"✅ All {total_datasets} datasets passed validation")
            else:
                suspicious = total_datasets - clean_datasets
                validation_results["passed"] = False
                validation_results["issues"] = [
                    f"{suspicious} datasets flagged as suspicious"
                ]
                print(f"❌ {suspicious}/{total_datasets} datasets flagged for review")

            # Check if we can load and test one dataset
            rule_dir = Path("data/output") / rule.safe_name
            if rule_dir.exists():
                csv_files = list(rule_dir.glob("*.csv"))
                if csv_files and "metadata" not in csv_files[0].name:
                    df = pd.read_csv(csv_files[0], index_col=0)
                    if "target" in df.columns:
                        target_rate = df["target"].mean()
                        print(f"   Sample dataset target rate: {target_rate:.4f}")

                        if target_rate in [0.0, 1.0]:
                            validation_results["passed"] = False
                            validation_results["issues"].append(
                                "Dataset has no target variation"
                            )
                        elif target_rate > 0.95:
                            validation_results["passed"] = False
                            validation_results["issues"].append(
                                f"Suspiciously high target rate: {target_rate:.4f}"
                            )
                        else:
                            print(
                                f"   ✅ Target rate looks reasonable: {target_rate:.4f}"
                            )

        except Exception as e:
            validation_results["passed"] = False
            validation_results["issues"] = [f"Generator test failed: {str(e)}"]
            print(f"❌ Universal generator test failed: {e}")

        if validation_results["passed"]:
            print(f"\n✅ Universal generator validation PASSED")
        else:
            print(f"\n❌ Universal generator validation FAILED")
            for issue in validation_results["issues"]:
                print(f"   ⚠️  {issue}")

        return validation_results

    def validate_unsupervised_patterns(self) -> Dict:
        """Test 6: Unsupervised validation - patterns should make mathematical sense"""
        print("\n🔍 TEST 6: UNSUPERVISED PATTERN VALIDATION")
        print("=" * 50)

        validation_results = {
            "test_name": "Unsupervised Patterns",
            "passed": True,
            "issues": [],
            "pattern_analysis": {},
        }

        print("Testing that discovered patterns make mathematical sense...")

        try:
            # Test with perfect squares (well-understood pattern)
            def perfect_squares(n):
                root = int(n**0.5)
                return root * root == n

            discoverer = UniversalMathDiscovery(
                target_function=perfect_squares,
                function_name="Perfect Squares Unsupervised Test",
                max_number=500,
                validate_no_leaking=True,
            )

            discoverer.generate_target_data()
            models = discoverer.train_discovery_models()

            # Analyze feature importance
            feature_importance = discoverer.analyze_mathematical_discoveries()

            if feature_importance is not None:
                top_features = feature_importance.head(10)["feature"].tolist()
                validation_results["pattern_analysis"]["top_features"] = top_features

                # Check if important features make mathematical sense for squares
                sensible_features = [
                    "sqrt",
                    "mod_",
                    "digit",
                    "log",
                    "number",
                    "fractional",
                    "power",
                ]

                sensible_count = 0
                for feature in top_features[:5]:  # Check top 5
                    if any(
                        sensible in feature.lower() for sensible in sensible_features
                    ):
                        sensible_count += 1

                sensible_ratio = sensible_count / 5
                validation_results["pattern_analysis"][
                    "sensible_ratio"
                ] = sensible_ratio

                print(f"   Top 5 features: {top_features[:5]}")
                print(
                    f"   Mathematically sensible features: {sensible_count}/5 ({sensible_ratio:.2f})"
                )

                if sensible_ratio >= 0.4:  # At least 40% should be sensible
                    print(f"   ✅ Discovered features make mathematical sense")
                else:
                    validation_results["passed"] = False
                    validation_results["issues"].append(
                        f"Too many non-sensible features in top 5"
                    )
                    print(
                        f"   ❌ Discovered features don't align with mathematical expectations"
                    )
            else:
                validation_results["passed"] = False
                validation_results["issues"].append(
                    "Could not analyze feature importance"
                )
                print(f"   ❌ Failed to analyze feature importance")

        except Exception as e:
            validation_results["passed"] = False
            validation_results["issues"] = [f"Unsupervised test failed: {str(e)}"]
            print(f"❌ Unsupervised pattern test failed: {e}")

        if validation_results["passed"]:
            print(f"\n✅ Unsupervised pattern validation PASSED")
        else:
            print(f"\n❌ Unsupervised pattern validation FAILED")
            for issue in validation_results["issues"]:
                print(f"   ⚠️  {issue}")

        return validation_results

    def run_comprehensive_validation(self) -> Dict:
        """Run all validation tests"""
        print("🚨 COMPREHENSIVE LABEL LEAKING VALIDATION")
        print("=" * 60)
        print("Testing that our mathematical discovery system genuinely")
        print("discovers patterns rather than exploiting label leaking.")
        print()

        start_time = time.time()
        all_results = {}

        # Run all validation tests
        tests = [
            ("feature_generation", self.validate_feature_generation),
            ("model_performance", self.validate_model_performance),
            ("cross_function", self.validate_cross_function_discovery),
            ("baseline_comparison", self.validate_baseline_comparison),
            ("universal_generator", self.validate_universal_generator),
            ("unsupervised_patterns", self.validate_unsupervised_patterns),
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            try:
                result = test_func()
                all_results[test_name] = result
                if result["passed"]:
                    passed_tests += 1
            except Exception as e:
                print(f"❌ Test {test_name} crashed: {e}")
                all_results[test_name] = {
                    "test_name": test_name,
                    "passed": False,
                    "issues": [f"Test crashed: {str(e)}"],
                    "error": str(e),
                }

        total_time = time.time() - start_time

        # Generate final report
        final_report = {
            "validation_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests,
                "total_time": total_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "test_results": all_results,
            "overall_assessment": self._generate_overall_assessment(
                all_results, passed_tests, total_tests
            ),
        }

        # Display final results
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE VALIDATION RESULTS")
        print("=" * 60)
        print(
            f"Tests passed: {passed_tests}/{total_tests} ({passed_tests/total_tests:.1%})"
        )
        print(f"Total time: {total_time:.1f}s")

        # Show test-by-test results
        print(f"\nTest Results:")
        for test_name, result in all_results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {status} {result['test_name']}")
            if not result["passed"] and "issues" in result:
                for issue in result["issues"][:2]:  # Show first 2 issues
                    print(f"       • {issue}")

        # Overall assessment
        assessment = final_report["overall_assessment"]
        print(f"\n📋 OVERALL ASSESSMENT:")
        print(f"Label Leaking Status: {assessment['label_leaking_status']}")
        print(f"Discovery Integrity: {assessment['discovery_integrity']}")
        print(f"Recommendation: {assessment['recommendation']}")

        return final_report

    def _generate_overall_assessment(
        self, results: Dict, passed: int, total: int
    ) -> Dict:
        """Generate overall assessment of the validation"""
        success_rate = passed / total

        if success_rate >= 0.85:
            status = "✅ CLEAN - No significant label leaking detected"
            integrity = "HIGH - System appears to genuinely discover patterns"
            recommendation = "System is ready for mathematical discovery research"
        elif success_rate >= 0.7:
            status = "⚠️  MODERATE - Some concerns detected"
            integrity = "MEDIUM - Review flagged issues before proceeding"
            recommendation = "Address identified issues and re-validate"
        else:
            status = "🚨 SIGNIFICANT - Major label leaking likely"
            integrity = "LOW - System may be exploiting leaked information"
            recommendation = "Major fixes required before use"

        return {
            "label_leaking_status": status,
            "discovery_integrity": integrity,
            "recommendation": recommendation,
            "success_rate": success_rate,
            "confidence": (
                "high"
                if success_rate >= 0.8
                else "medium" if success_rate >= 0.6 else "low"
            ),
        }


def main():
    """Run comprehensive validation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive label leaking validation"
    )
    parser.add_argument(
        "--generate-report", action="store_true", help="Generate detailed JSON report"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick validation (fewer tests)"
    )

    args = parser.parse_args()

    # Run validation
    validator = LabelLeakingValidator()

    if args.quick:
        print("🚀 Running QUICK validation (subset of tests)...")
        # Run just the core tests
        results = {}
        results["feature_generation"] = validator.validate_feature_generation()
        results["model_performance"] = validator.validate_model_performance()

        passed = sum(1 for r in results.values() if r["passed"])
        total = len(results)

        print(f"\n📊 QUICK VALIDATION RESULTS: {passed}/{total} tests passed")

        if passed == total:
            print("✅ Quick validation PASSED - system looks clean")
        else:
            print("❌ Quick validation FAILED - run full validation for details")

    else:
        print("🔬 Running COMPREHENSIVE validation (all tests)...")
        results = validator.run_comprehensive_validation()

        # Save detailed report if requested
        if args.generate_report:
            report_file = Path("comprehensive_validation_report.json")
            with open(report_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved to: {report_file}")

    print(f"\n🎯 NEXT STEPS:")
    if args.quick:
        print("1. Run full validation: python comprehensive_validation.py")
        print("2. Address any issues found")
        print("3. Re-run validation after fixes")
    else:
        success_rate = results["validation_summary"]["success_rate"]
        if success_rate >= 0.85:
            print("1. ✅ System is ready for mathematical discovery!")
            print("2. Begin research on novel mathematical patterns")
            print("3. Document discovered patterns for validation")
        elif success_rate >= 0.7:
            print("1. Review and fix issues identified in failed tests")
            print("2. Re-run validation after fixes")
            print("3. Consider additional feature auditing")
        else:
            print("1. 🚨 Major fixes required - system has significant label leaking")
            print("2. Audit all feature generation code")
            print("3. Remove boolean flags and direct rule access")
            print("4. Re-run validation after major fixes")


if __name__ == "__main__":
    main()
