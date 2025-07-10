#!/usr/bin/env python3
"""
Universal Mathematical Dataset Generator
=========================================

A flexible system to convert any mathematical rule/function into multiple
ML-ready dataset formats for pattern discovery.

Features:
- Takes mathematical functions and generates raw number sets
- Converts to multiple ML representations (prefix-suffix, digit tensor, etc.)
- Organized folder structure for easy management
- Scalable up to millions of numbers
- No hard-coded mathematical knowledge - pure feature extraction

Author: Mathematical Pattern Discovery Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import warnings
from typing import Callable, List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import argparse

warnings.filterwarnings("ignore")

# Import our utilities
try:
    from utils.prefix_suffix_utils import generate_prefix_suffix_matrix
    from utils.math_utils import generate_mathematical_features
    from utils.embedding_utils import fourier_transform, pca_transform
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from utils.prefix_suffix_utils import generate_prefix_suffix_matrix
    from utils.math_utils import generate_mathematical_features
    from utils.embedding_utils import fourier_transform, pca_transform


class MathematicalRule:
    """Container for a mathematical rule/function with metadata"""

    def __init__(
        self,
        func: Callable[[int], bool],
        name: str,
        description: str = "",
        examples: List[int] = None,
    ):
        self.func = func
        self.name = name
        self.description = description
        self.examples = examples or []
        self.safe_name = self._make_safe_name(name)

    def _make_safe_name(self, name: str) -> str:
        """Convert name to filesystem-safe format"""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower())
        return safe.replace("__", "_").strip("_")

    def evaluate(self, n: int) -> bool:
        """Safely evaluate the function"""
        try:
            return bool(self.func(n))
        except:
            return False


class DatasetProcessor(ABC):
    """Abstract base class for dataset processing methods"""

    @abstractmethod
    def process(self, numbers: List[int], metadata: Dict) -> pd.DataFrame:
        """Process a list of numbers into an ML-ready dataset"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the processor name"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get the processor description"""
        pass


class PrefixSuffixProcessor(DatasetProcessor):
    """Converts numbers to prefix-suffix matrix format"""

    def __init__(self, prefix_digits: int = 2, suffix_digits: int = 1):
        self.prefix_digits = prefix_digits
        self.suffix_digits = suffix_digits

    def process(self, numbers: List[int], metadata: Dict) -> pd.DataFrame:
        """Create prefix-suffix matrix"""
        df = generate_prefix_suffix_matrix(
            numbers, self.prefix_digits, self.suffix_digits
        )
        return df

    def get_name(self) -> str:
        return f"prefix_{self.prefix_digits}d_suffix_{self.suffix_digits}d"

    def get_description(self) -> str:
        return f"Prefix-suffix matrix ({self.prefix_digits} prefix digits, {self.suffix_digits} suffix digits)"


class DigitTensorProcessor(DatasetProcessor):
    """Converts numbers to digit-based tensor features"""

    def __init__(self, max_digits: int = 6, include_embeddings: bool = True):
        self.max_digits = max_digits
        self.include_embeddings = include_embeddings

    def process(self, numbers: List[int], metadata: Dict) -> pd.DataFrame:
        """Create digit tensor dataset"""
        max_num = max(numbers) if numbers else 1000

        features_list = []
        target_list = []
        digit_tensors = []

        print(f"    Processing {max_num} numbers for digit tensor features...")

        for n in range(1, max_num + 1):
            # Basic digit features
            digits = [int(d) for d in str(n)]
            padded_digits = digits + [0] * (self.max_digits - len(digits))
            padded_digits = padded_digits[: self.max_digits]  # Truncate if too long

            features = {
                "number": n,
                "digit_count": len(digits),
                "digit_sum": sum(digits),
                "digit_product": (
                    np.prod([d for d in digits if d > 0])
                    if any(d > 0 for d in digits)
                    else 0
                ),
                "first_digit": digits[0] if digits else 0,
                "last_digit": digits[-1] if digits else 0,
                "is_palindrome": int(digits == digits[::-1]),
                "has_repeating_digits": int(len(set(digits)) < len(digits)),
                "alternating_sum": sum((-1) ** i * d for i, d in enumerate(digits)),
            }

            # Add individual digit positions
            for i in range(self.max_digits):
                features[f"digit_pos_{i}"] = padded_digits[i]

            features_list.append(features)
            target_list.append(1 if n in numbers else 0)
            digit_tensors.append(padded_digits)

        df = pd.DataFrame(features_list)
        df["target"] = target_list

        # Add embeddings if requested
        if self.include_embeddings:
            # Fourier transform of digit patterns
            fourier_features = [
                fourier_transform(tensor, 8) for tensor in digit_tensors
            ]
            for i in range(8):
                df[f"fourier_{i}"] = [
                    f[i] if i < len(f) else 0 for f in fourier_features
                ]

            # PCA of digit patterns
            pca_features = pca_transform(digit_tensors, 3)
            for i in range(3):
                df[f"pca_{i}"] = [f[i] for f in pca_features]

        return df

    def get_name(self) -> str:
        embed_str = "_with_embeddings" if self.include_embeddings else ""
        return f"digit_tensor_{self.max_digits}d{embed_str}"

    def get_description(self) -> str:
        embed_str = " with Fourier/PCA embeddings" if self.include_embeddings else ""
        return f"Digit tensor features (max {self.max_digits} digits){embed_str}"


class SequencePatternProcessor(DatasetProcessor):
    """Extracts sequential pattern features"""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def process(self, numbers: List[int], metadata: Dict) -> pd.DataFrame:
        """Create sequence pattern dataset"""
        sorted_numbers = sorted(numbers)
        max_num = max(numbers) if numbers else 1000

        features_list = []
        target_list = []

        print(
            f"    Processing sequence patterns with window size {self.window_size}..."
        )

        for n in range(1, max_num + 1):
            # Find position in sequence
            is_member = n in numbers

            # Gap analysis
            smaller = [x for x in sorted_numbers if x < n]
            larger = [x for x in sorted_numbers if x > n]

            prev_gap = n - max(smaller) if smaller else 0
            next_gap = min(larger) - n if larger else 0

            # Local density
            window_start = max(1, n - self.window_size)
            window_end = min(max_num, n + self.window_size)
            window_members = [
                x for x in sorted_numbers if window_start <= x <= window_end
            ]
            local_density = len(window_members) / (window_end - window_start + 1)

            # Differences and ratios for nearby members
            if is_member and len(smaller) >= 2:
                recent_diffs = [smaller[-1] - smaller[-2]]
                if len(smaller) >= 3:
                    recent_diffs.append(smaller[-2] - smaller[-3])
                avg_recent_diff = np.mean(recent_diffs)

                if smaller[-1] != 0:
                    growth_ratio = n / smaller[-1]
                else:
                    growth_ratio = 1
            else:
                avg_recent_diff = 0
                growth_ratio = 1

            features = {
                "number": n,
                "prev_gap": prev_gap,
                "next_gap": next_gap,
                "local_density": local_density,
                "avg_recent_diff": avg_recent_diff,
                "growth_ratio": growth_ratio,
                "members_before": len(smaller),
                "members_after": len(larger),
                "is_isolated": int(prev_gap > 10 and next_gap > 10),
                "in_cluster": int(prev_gap <= 2 or next_gap <= 2),
            }

            features_list.append(features)
            target_list.append(1 if is_member else 0)

        df = pd.DataFrame(features_list)
        df["target"] = target_list

        return df

    def get_name(self) -> str:
        return f"sequence_patterns_w{self.window_size}"

    def get_description(self) -> str:
        return f"Sequential pattern features (window size {self.window_size})"


class AlgebraicFeatureProcessor(DatasetProcessor):
    """Extracts algebraic and number-theoretic features"""

    def process(self, numbers: List[int], metadata: Dict) -> pd.DataFrame:
        """Create algebraic feature dataset"""
        max_num = max(numbers) if numbers else 1000

        features_list = []
        target_list = []

        print(f"    Processing algebraic features for {max_num} numbers...")

        for n in range(1, max_num + 1):
            # Use our existing comprehensive feature generator
            features = generate_mathematical_features(n)

            # Add set-specific features
            features["is_member"] = int(n in numbers)

            features_list.append(features)
            target_list.append(1 if n in numbers else 0)

        df = pd.DataFrame(features_list)
        df["target"] = target_list

        return df

    def get_name(self) -> str:
        return "algebraic_features"

    def get_description(self) -> str:
        return "Comprehensive algebraic and number-theoretic features"


class UniversalDatasetGenerator:
    """Main class for generating universal mathematical datasets"""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.output_dir = self.base_dir / "output"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Available processors
        self.processors = {
            "prefix_suffix_2_1": PrefixSuffixProcessor(2, 1),
            "prefix_suffix_3_2": PrefixSuffixProcessor(3, 2),
            "digit_tensor": DigitTensorProcessor(6, True),
            "digit_tensor_simple": DigitTensorProcessor(4, False),
            "sequence_patterns": SequencePatternProcessor(5),
            "sequence_patterns_wide": SequencePatternProcessor(10),
            "algebraic_features": AlgebraicFeatureProcessor(),
        }

    def generate_raw_dataset(
        self, rule: MathematicalRule, max_number: int = 100000, save_raw: bool = True
    ) -> Tuple[List[int], Dict]:
        """Generate raw number set from mathematical rule"""

        print(f"\n🔢 Generating raw dataset: {rule.name}")
        print(f"   Testing numbers 1 to {max_number:,}")

        numbers = []
        start_time = time.time()

        for n in range(1, max_number + 1):
            if rule.evaluate(n):
                numbers.append(n)

            # Progress update
            if n % 10000 == 0:
                elapsed = time.time() - start_time
                rate = n / elapsed if elapsed > 0 else 0
                remaining = (max_number - n) / rate if rate > 0 else 0
                found_rate = len(numbers) / n
                print(
                    f"   Progress: {n:,}/{max_number:,} ({100*n/max_number:.1f}%) "
                    f"- Found: {len(numbers):,} ({found_rate:.4f}) "
                    f"- Rate: {rate:.0f}/sec - ETA: {remaining:.0f}s"
                )

        total_time = time.time() - start_time

        # Create metadata
        metadata = {
            "rule_name": rule.name,
            "rule_description": rule.description,
            "safe_name": rule.safe_name,
            "max_number": max_number,
            "total_found": len(numbers),
            "density": len(numbers) / max_number,
            "generation_time": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "examples": rule.examples,
            "first_20": numbers[:20],
            "last_20": numbers[-20:] if len(numbers) >= 20 else numbers,
        }

        print(
            f"   ✅ Found {len(numbers):,} numbers ({len(numbers)/max_number:.4f} density)"
        )
        print(f"   ⏱️  Generation time: {total_time:.1f}s")

        if save_raw:
            # Save raw dataset
            rule_dir = self.raw_dir / rule.safe_name
            rule_dir.mkdir(exist_ok=True)

            # Save numbers
            numbers_file = rule_dir / f"numbers_up_to_{max_number}.csv"
            pd.DataFrame({"number": numbers}).to_csv(numbers_file, index=False)

            # Save metadata
            metadata_file = rule_dir / f"metadata_up_to_{max_number}.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"   💾 Saved raw dataset to: {rule_dir}")

        return numbers, metadata

    def process_to_ml_datasets(
        self, numbers: List[int], metadata: Dict, processors: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Convert raw numbers to ML-ready datasets using multiple processors"""

        if processors is None:
            processors = list(self.processors.keys())

        rule_name = metadata["safe_name"]
        max_number = metadata["max_number"]

        print(f"\n🤖 Processing to ML datasets: {rule_name}")
        print(f"   Using processors: {processors}")

        # Create output directory for this rule
        output_rule_dir = self.output_dir / rule_name
        output_rule_dir.mkdir(exist_ok=True)

        results = {}

        for proc_name in processors:
            if proc_name not in self.processors:
                print(f"   ⚠️  Unknown processor: {proc_name}")
                continue

            processor = self.processors[proc_name]

            print(f"\n   📊 Processing with: {processor.get_description()}")

            try:
                start_time = time.time()

                # Process the data
                df = processor.process(numbers, metadata)

                process_time = time.time() - start_time

                # Save the dataset
                output_file = (
                    output_rule_dir / f"{processor.get_name()}_up_to_{max_number}.csv"
                )
                df.to_csv(output_file, index=True)

                # Create processor metadata
                proc_metadata = {
                    "processor": proc_name,
                    "description": processor.get_description(),
                    "dataset_shape": list(df.shape),
                    "columns": list(df.columns),
                    "processing_time": process_time,
                    "output_file": str(output_file),
                }

                metadata_file = (
                    output_rule_dir
                    / f"{processor.get_name()}_metadata_up_to_{max_number}.json"
                )
                with open(metadata_file, "w") as f:
                    json.dump(proc_metadata, f, indent=2)

                results[proc_name] = df

                print(f"      ✅ Shape: {df.shape}, Time: {process_time:.1f}s")
                print(f"      💾 Saved to: {output_file}")

            except Exception as e:
                print(f"      ❌ Failed: {e}")
                continue

        print(f"\n   🎉 Generated {len(results)} ML datasets for {rule_name}")

        return results

    def generate_complete_pipeline(
        self,
        rule: MathematicalRule,
        max_number: int = 100000,
        processors: List[str] = None,
    ) -> Dict[str, Any]:
        """Complete pipeline: rule -> raw dataset -> ML datasets"""

        print("=" * 70)
        print(f"🧮 UNIVERSAL DATASET GENERATION PIPELINE")
        print("=" * 70)
        print(f"Rule: {rule.name}")
        print(f"Description: {rule.description}")
        print(f"Max number: {max_number:,}")
        print(f"Examples: {rule.examples}")

        # Step 1: Generate raw dataset
        numbers, metadata = self.generate_raw_dataset(rule, max_number)

        # Step 2: Process to ML datasets
        ml_datasets = self.process_to_ml_datasets(numbers, metadata, processors)

        # Create summary
        summary = {
            "rule": rule,
            "metadata": metadata,
            "raw_numbers": numbers,
            "ml_datasets": ml_datasets,
            "processors_used": list(ml_datasets.keys()),
            "total_datasets": len(ml_datasets),
        }

        print("\n" + "=" * 70)
        print(f"🎉 PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Raw dataset: {len(numbers):,} numbers")
        print(f"ML datasets: {len(ml_datasets)} different formats")
        print(f"Storage location: {self.output_dir / rule.safe_name}")

        return summary


# Example mathematical rules for testing
def create_example_rules() -> List[MathematicalRule]:
    """Create a set of example mathematical rules"""

    rules = []

    # Perfect squares
    rules.append(
        MathematicalRule(
            func=lambda n: int(n**0.5) ** 2 == n,
            name="Perfect Squares",
            description="Numbers that are perfect squares: n = k² for some integer k",
            examples=[1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        )
    )

    # Triangular numbers
    rules.append(
        MathematicalRule(
            func=lambda n: int((-1 + (1 + 8 * n) ** 0.5) / 2)
            * (int((-1 + (1 + 8 * n) ** 0.5) / 2) + 1)
            // 2
            == n,
            name="Triangular Numbers",
            description="Numbers of the form n = k(k+1)/2 for some integer k",
            examples=[1, 3, 6, 10, 15, 21, 28, 36, 45, 55],
        )
    )

    # Powers of 2
    rules.append(
        MathematicalRule(
            func=lambda n: n > 0 and (n & (n - 1)) == 0,
            name="Powers of 2",
            description="Numbers that are powers of 2: n = 2^k for some integer k",
            examples=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        )
    )

    # Fibonacci numbers (inefficient but works for demo)
    def is_fibonacci(n):
        a, b = 1, 1
        if n == 1:
            return True
        while b < n:
            a, b = b, a + b
        return b == n

    rules.append(
        MathematicalRule(
            func=is_fibonacci,
            name="Fibonacci Numbers",
            description="Numbers in the Fibonacci sequence: F(n) = F(n-1) + F(n-2)",
            examples=[1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        )
    )

    # Numbers with digit sum divisible by 3
    rules.append(
        MathematicalRule(
            func=lambda n: sum(int(d) for d in str(n)) % 3 == 0,
            name="Digit Sum Divisible by 3",
            description="Numbers whose digit sum is divisible by 3",
            examples=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        )
    )

    # Palindromic numbers
    rules.append(
        MathematicalRule(
            func=lambda n: str(n) == str(n)[::-1],
            name="Palindromic Numbers",
            description="Numbers that read the same forwards and backwards",
            examples=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
        )
    )

    return rules


def create_custom_rules():
    """Create additional custom mathematical rules for the CLI"""

    rules = []

    # Primes (simple trial division)
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    rules.append(
        MathematicalRule(
            func=is_prime,
            name="Prime Numbers",
            description="Numbers with exactly two positive divisors",
            examples=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        )
    )

    # Narcissistic numbers
    def is_narcissistic(n):
        digits = [int(d) for d in str(n)]
        power = len(digits)
        return n == sum(d**power for d in digits)

    rules.append(
        MathematicalRule(
            func=is_narcissistic,
            name="Narcissistic Numbers",
            description="Numbers equal to sum of digits raised to power of digit count",
            examples=[1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 371, 407],
        )
    )

    return rules


def list_all_rules():
    """List all available mathematical rules"""
    print("📋 AVAILABLE MATHEMATICAL RULES")
    print("=" * 50)

    # Get built-in rules
    builtin_rules = create_example_rules()
    custom_rules = create_custom_rules()

    all_rules = builtin_rules + custom_rules

    print(f"\n🏗️  Built-in Rules ({len(builtin_rules)}):")
    for i, rule in enumerate(builtin_rules, 1):
        print(f"  {i:2d}. {rule.name}")
        print(f"      {rule.description}")
        print(f"      Examples: {rule.examples[:8]}...")
        print()

    print(f"🎯 Additional Rules ({len(custom_rules)}):")
    start_idx = len(builtin_rules) + 1
    for i, rule in enumerate(custom_rules, start_idx):
        print(f"  {i:2d}. {rule.name}")
        print(f"      {rule.description}")
        print(f"      Examples: {rule.examples[:8]}...")
        print()

    print(f"Total: {len(all_rules)} mathematical rules available")
    return all_rules


def interactive_rule_creator():
    """Enhanced interactive mode to create custom rules with size controls"""
    print("\n🛠️  ENHANCED INTERACTIVE RULE CREATOR")
    print("=" * 50)
    print("Create a custom mathematical rule and specify dataset sizes.")
    print("Your function should take an integer n and return True/False.")
    print()

    print("Examples:")
    print("  lambda n: n % 3 == 0                    # Multiples of 3")
    print("  lambda n: str(n) == str(n)[::-1]        # Palindromes")
    print("  lambda n: sum(int(d)**2 for d in str(n)) == n  # Happy numbers")
    print("  lambda n: len(str(n)) == sum(int(d) for d in str(n))  # Special property")
    print()

    try:
        # Step 1: Get function code
        func_code = input("Enter your function (lambda n: ...): ").strip()
        if not func_code.startswith("lambda"):
            func_code = "lambda n: " + func_code

        # Test the function
        print("\n🧪 Testing your function...")
        func = eval(func_code)

        # Test on small numbers to verify it works
        test_results = []
        for i in range(1, 51):  # Test 1-50
            try:
                result = func(i)
                if result:
                    test_results.append(i)
            except:
                pass

        print(
            f"Test results (1-50): {test_results[:15]}{'...' if len(test_results) > 15 else ''}"
        )
        print(f"Found {len(test_results)} matching numbers in range 1-50")

        if not test_results:
            print("⚠️  No matches found in 1-50. Rule might be too restrictive.")
            print("Consider testing a broader range or adjusting your rule.")
            return None

        # Step 2: Get rule metadata
        print("\n📝 Rule Information:")
        name = input("Rule name: ").strip()
        description = input("Description: ").strip()

        # Step 3: Size configuration
        print("\n📊 Dataset Size Configuration:")
        print("You can specify different sizes for generation and processing.")
        print()

        # Raw dataset size
        print("🔢 Raw Dataset Generation:")
        print("  This determines how many numbers to test to find matches.")
        print("  Larger ranges find more examples but take longer.")
        print("  Recommended: 10,000 - 1,000,000")

        while True:
            try:
                max_number = input("Maximum number to test (default: 50,000): ").strip()
                if not max_number:
                    max_number = 50000
                else:
                    max_number = int(max_number)

                if max_number < 100:
                    print("⚠️  Too small. Minimum recommended: 100")
                    continue
                elif max_number > 10000000:
                    print(
                        "⚠️  Very large. This might take a long time. Continue? (y/N): ",
                        end="",
                    )
                    if input().strip().lower() != "y":
                        continue
                break
            except ValueError:
                print("❌ Please enter a valid number")

        # ML processing scope
        print(f"\n🤖 ML Dataset Processing:")
        print("  This determines the scope for generating ML features.")
        print("  You can process fewer numbers than you tested for speed.")
        print("  This affects the final ML dataset size, not the raw findings.")

        while True:
            try:
                ml_max_str = input(
                    f"ML processing scope (default: {min(max_number, 10000)}): "
                ).strip()
                if not ml_max_str:
                    ml_max_number = min(max_number, 10000)
                else:
                    ml_max_number = int(ml_max_str)

                if ml_max_number > max_number:
                    print(
                        f"⚠️  ML scope cannot exceed raw generation scope ({max_number})"
                    )
                    continue
                elif ml_max_number < 100:
                    print("⚠️  Too small for meaningful ML. Minimum recommended: 100")
                    continue
                break
            except ValueError:
                print("❌ Please enter a valid number")

        # Processor selection
        print(f"\n⚙️  Processor Selection:")
        print("  Choose which ML representations to generate:")

        available_processors = {
            "1": ("prefix_suffix_2_1", "Prefix-Suffix Matrix (2-1 digits)"),
            "2": ("prefix_suffix_3_2", "Prefix-Suffix Matrix (3-2 digits)"),
            "3": ("digit_tensor", "Digit Tensor with embeddings"),
            "4": ("digit_tensor_simple", "Simple Digit Tensor"),
            "5": ("sequence_patterns", "Sequence Pattern Analysis"),
            "6": ("sequence_patterns_wide", "Wide Sequence Patterns"),
            "7": ("algebraic_features", "Comprehensive Algebraic Features"),
            "a": ("all", "All processors (recommended)"),
        }

        for key, (proc_name, description) in available_processors.items():
            print(f"  {key}. {description}")

        print()
        processor_input = (
            input("Select processors (comma-separated, or 'a' for all): ")
            .strip()
            .lower()
        )

        if processor_input == "a" or processor_input == "all":
            selected_processors = None  # None means all
            processor_names = "All processors"
        else:
            selected_keys = [k.strip() for k in processor_input.split(",")]
            selected_processors = []
            processor_descriptions = []

            for key in selected_keys:
                if key in available_processors and key != "a":
                    proc_name, description = available_processors[key]
                    selected_processors.append(proc_name)
                    processor_descriptions.append(description)

            if not selected_processors:
                print("⚠️  No valid processors selected. Using all processors.")
                selected_processors = None
                processor_names = "All processors"
            else:
                processor_names = ", ".join(processor_descriptions)

        # Confirmation
        print(f"\n📋 CONFIGURATION SUMMARY:")
        print("=" * 40)
        print(f"Rule Name: {name}")
        print(f"Description: {description}")
        print(f"Function: {func_code}")
        print(f"Raw Generation Scope: 1 to {max_number:,}")
        print(f"ML Processing Scope: 1 to {ml_max_number:,}")
        print(f"Processors: {processor_names}")
        print()

        # Estimate time and size
        estimated_raw_time = max_number / 100000  # Rough estimate
        estimated_ml_time = ml_max_number / 10000  # Rough estimate

        print(f"⏱️  Estimated time:")
        print(f"  Raw generation: ~{estimated_raw_time:.1f} seconds")
        print(f"  ML processing: ~{estimated_ml_time:.1f} seconds")
        print()

        confirm = input("Proceed with generation? (Y/n): ").strip().lower()
        if confirm == "n":
            print("❌ Generation cancelled.")
            return None

        # Create rule with the specified configuration
        rule = MathematicalRule(
            func=func, name=name, description=description, examples=test_results[:10]
        )

        # Return rule and configuration
        return {
            "rule": rule,
            "max_number": max_number,
            "ml_max_number": ml_max_number,
            "processors": selected_processors,
        }

    except KeyboardInterrupt:
        print("\n❌ Cancelled by user.")
        return None
    except Exception as e:
        print(f"❌ Error creating rule: {e}")
        return None


def enhanced_generation_pipeline(rule_config):
    """Enhanced pipeline that handles different sizes for raw and ML generation"""
    rule = rule_config["rule"]
    max_number = rule_config["max_number"]
    ml_max_number = rule_config["ml_max_number"]
    processors = rule_config["processors"]

    print(f"\n🚀 ENHANCED GENERATION PIPELINE")
    print("=" * 50)
    print(f"Rule: {rule.name}")
    print(f"Raw generation: 1 to {max_number:,}")
    print(f"ML processing: 1 to {ml_max_number:,}")

    generator = UniversalDatasetGenerator()

    # Step 1: Generate raw dataset
    print(f"\n🔢 Step 1: Raw Dataset Generation")
    numbers, metadata = generator.generate_raw_dataset(rule, max_number)

    found_count = len(numbers)
    density = found_count / max_number

    print(f"✅ Found {found_count:,} matching numbers")
    print(f"📊 Density: {density:.4f} ({density*100:.2f}%)")

    if found_count == 0:
        print("❌ No matching numbers found. Try a different rule or larger range.")
        return None

    # Step 2: Adjust for ML processing scope
    if ml_max_number < max_number:
        print(f"\n🎯 Step 2: Adjusting for ML Processing Scope")

        # Filter numbers to only those within ML scope
        ml_numbers = [n for n in numbers if n <= ml_max_number]
        ml_found_count = len(ml_numbers)
        ml_density = ml_found_count / ml_max_number if ml_max_number > 0 else 0

        print(f"📊 Numbers in ML scope: {ml_found_count:,} out of {found_count:,}")
        print(f"📊 ML scope density: {ml_density:.4f} ({ml_density*100:.2f}%)")

        if ml_found_count == 0:
            print("⚠️  No numbers found in ML processing scope.")
            print("Consider increasing ML scope or adjusting the rule.")
            return None

        # Create adjusted metadata for ML processing
        ml_metadata = metadata.copy()
        ml_metadata["max_number"] = ml_max_number
        ml_metadata["total_found"] = ml_found_count
        ml_metadata["density"] = ml_density
        ml_metadata["original_max_number"] = max_number
        ml_metadata["original_total_found"] = found_count

        processing_numbers = ml_numbers
        processing_metadata = ml_metadata
    else:
        processing_numbers = numbers
        processing_metadata = metadata

    # Step 3: Generate ML datasets
    print(f"\n🤖 Step 3: ML Dataset Generation")
    ml_datasets = generator.process_to_ml_datasets(
        processing_numbers, processing_metadata, processors
    )

    # Step 4: Results summary
    print(f"\n📊 GENERATION COMPLETE!")
    print("=" * 40)
    print(f"✅ Raw dataset: {found_count:,} numbers found (testing 1-{max_number:,})")
    print(
        f"✅ ML datasets: {len(ml_datasets)} formats generated (scope 1-{ml_max_number:,})"
    )
    print(f"📁 Location: data/output/{rule.safe_name}/")

    # Show what was created
    output_dir = Path("data/output") / rule.safe_name
    if output_dir.exists():
        files = list(output_dir.glob("*.csv"))
        print(f"\n📂 Generated Files:")
        for file in files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  📄 {file.name} ({size_mb:.2f} MB)")

    return {
        "rule": rule,
        "raw_numbers": numbers,
        "ml_datasets": ml_datasets,
        "raw_metadata": metadata,
        "ml_metadata": processing_metadata,
        "processors_used": list(ml_datasets.keys()),
        "total_datasets": len(ml_datasets),
    }


def generate_rule_datasets_enhanced(rule_number, max_number, ml_max_number, processors):
    """Enhanced rule generation with separate raw and ML scopes"""

    # Get all rules
    builtin_rules = create_example_rules()
    custom_rules = create_custom_rules()
    all_rules = builtin_rules + custom_rules

    if rule_number < 1 or rule_number > len(all_rules):
        print(f"❌ Invalid rule number. Use 1-{len(all_rules)}")
        return False

    rule = all_rules[rule_number - 1]

    # Create config
    rule_config = {
        "rule": rule,
        "max_number": max_number,
        "ml_max_number": ml_max_number,
        "processors": processors,
    }

    # Run enhanced pipeline
    summary = enhanced_generation_pipeline(rule_config)

    if summary:
        print(f"\n✅ Successfully generated {summary['total_datasets']} datasets!")
        print(f"📁 Location: data/output/{rule.safe_name}/")
        return True
    else:
        return False


def main():
    """Main CLI interface with enhanced interactive mode"""
    parser = argparse.ArgumentParser(
        description="Generate ML-ready datasets from mathematical rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                           # List all available rules
  %(prog)s generate 1 --max 10000        # Generate perfect squares up to 10,000
  %(prog)s generate 5 --processors prefix_suffix_2_1,digit_tensor
  %(prog)s interactive                    # Enhanced interactive rule creation
  %(prog)s demo                          # Quick demo with perfect squares
        """,
    )

    parser.add_argument(
        "command",
        choices=["list", "generate", "interactive", "demo"],
        help="Command to execute",
    )
    parser.add_argument(
        "rule_number",
        nargs="?",
        type=int,
        help='Rule number to generate (use "list" to see options)',
    )
    parser.add_argument(
        "--max",
        "--max-number",
        type=int,
        default=10000,
        help="Maximum number to test (default: 10000)",
    )
    parser.add_argument(
        "--ml-max",
        "--ml-max-number",
        type=int,
        help="Maximum number for ML processing (default: same as --max)",
    )
    parser.add_argument(
        "--processors", type=str, help="Comma-separated list of processors to use"
    )

    args = parser.parse_args()

    # Parse processors
    processors = None
    if args.processors:
        processors = [p.strip() for p in args.processors.split(",")]

    print("🧮 UNIVERSAL DATASET GENERATOR")
    print("=" * 50)

    if args.command == "list":
        list_all_rules()

    elif args.command == "generate":
        if args.rule_number is None:
            print("❌ Please specify a rule number (use 'list' command to see options)")
            parser.print_help()
            return

        ml_max = args.ml_max if args.ml_max else args.max
        success = generate_rule_datasets_enhanced(
            args.rule_number, args.max, ml_max, processors
        )
        if not success:
            sys.exit(1)

    elif args.command == "interactive":
        rule_config = interactive_rule_creator()
        if rule_config:
            summary = enhanced_generation_pipeline(rule_config)
            if summary:
                print(
                    f"✅ Successfully generated {summary['total_datasets']} ML datasets!"
                )

    elif args.command == "demo":
        print("🎯 Running quick demo with Perfect Squares...")
        ml_max = min(args.max, 5000)
        success = generate_rule_datasets_enhanced(1, args.max, ml_max, processors)
        if success:
            print("\n🎉 Demo complete! Check data/output/perfect_squares/ for results.")


if __name__ == "__main__":
    main()
