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


def main():
    """Example usage and testing"""
    import sys

    if len(sys.argv) < 2:
        print("Universal Mathematical Dataset Generator")
        print("=" * 50)
        print("Usage: python universal_dataset_generator.py <command> [options]")
        print()
        print("Commands:")
        print("  demo              - Run demo with example rules")
        print("  list              - List available mathematical rules")
        print("  generate <rule>   - Generate datasets for specific rule")
        print()
        print("Options:")
        print("  --max-number N    - Maximum number to test (default: 10000)")
        print("  --processors P    - Comma-separated list of processors")
        print()
        print("Example:")
        print("  python universal_dataset_generator.py demo --max-number 50000")
        return

    command = sys.argv[1].lower()

    # Parse options
    max_number = 10000
    processors = None

    for i, arg in enumerate(sys.argv):
        if arg == "--max-number" and i + 1 < len(sys.argv):
            max_number = int(sys.argv[i + 1])
        elif arg == "--processors" and i + 1 < len(sys.argv):
            processors = sys.argv[i + 1].split(",")

    # Initialize generator
    generator = UniversalDatasetGenerator()

    if command == "demo":
        print("🎯 Running demo with Perfect Squares...")
        rules = create_example_rules()
        rule = rules[0]  # Perfect squares

        summary = generator.generate_complete_pipeline(
            rule=rule, max_number=max_number, processors=processors
        )

        print(f"\n📊 Demo Summary:")
        print(f"  Generated {summary['total_datasets']} ML datasets")
        print(f"  Processors used: {', '.join(summary['processors_used'])}")

    elif command == "list":
        print("📋 Available Mathematical Rules:")
        print("=" * 40)
        rules = create_example_rules()
        for i, rule in enumerate(rules, 1):
            print(f"{i}. {rule.name}")
            print(f"   {rule.description}")
            print(f"   Examples: {rule.examples[:5]}...")
            print()

    elif command == "generate":
        if len(sys.argv) < 3:
            print("Please specify a rule number (use 'list' command to see options)")
            return

        rule_num = int(sys.argv[2]) - 1
        rules = create_example_rules()

        if 0 <= rule_num < len(rules):
            rule = rules[rule_num]
            summary = generator.generate_complete_pipeline(
                rule=rule, max_number=max_number, processors=processors
            )
        else:
            print(f"Invalid rule number. Use 1-{len(rules)}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
