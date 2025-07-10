#!/usr/bin/env python3
"""
Universal Dataset Generator - Complete Usage Examples
====================================================

This file shows comprehensive examples of how to use the Universal Dataset Generator
to create ML-ready datasets from any mathematical rule or function.

Author: Mathematical Pattern Discovery Team
"""

import sys
from pathlib import Path
import pandas as pd

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from generators.universal_generator import (
        UniversalDatasetGenerator,
        MathematicalRule,
        PrefixSuffixProcessor,
        DigitTensorProcessor,
        SequencePatternProcessor,
        AlgebraicFeatureProcessor,
    )
    from core.discovery_engine import UniversalMathDiscovery
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root.")
    sys.exit(1)


def example_1_basic_usage():
    """Example 1: Basic usage with a simple mathematical rule"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage - Perfect Squares")
    print("=" * 60)

    # Step 1: Define your mathematical rule
    def is_perfect_square(n):
        root = int(n**0.5)
        return root * root == n

    rule = MathematicalRule(
        func=is_perfect_square,
        name="Perfect Squares",
        description="Numbers that are perfect squares: n = k² for some integer k",
        examples=[1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
    )

    # Step 2: Initialize the generator
    generator = UniversalDatasetGenerator()

    # Step 3: Generate all datasets (this creates raw + ML datasets)
    summary = generator.generate_complete_pipeline(
        rule=rule,
        max_number=1000,  # Test numbers 1-1000
        processors=["prefix_suffix_2_1", "digit_tensor", "algebraic_features"],
    )

    print(f"\n✅ Generated {summary['total_datasets']} ML-ready datasets!")
    print(f"📁 Location: data/output/{rule.safe_name}/")

    return summary


def example_2_custom_oeis_sequence():
    """Example 2: Working with OEIS sequences"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: OEIS Sequence - A000045 (Fibonacci)")
    print("=" * 60)

    # OEIS A000045: Fibonacci numbers
    # Using the property that n is Fibonacci iff 5n²±4 is a perfect square
    def is_fibonacci(n):
        def is_perfect_square(x):
            if x < 0:
                return False
            root = int(x**0.5)
            return root * root == x

        return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)

    rule = MathematicalRule(
        func=is_fibonacci,
        name="OEIS A000045 Fibonacci",
        description="Fibonacci sequence: F(n) = F(n-1) + F(n-2) with F(1)=F(2)=1",
        examples=[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377],
    )

    generator = UniversalDatasetGenerator()

    # Generate with specific processors that work well for sequences
    summary = generator.generate_complete_pipeline(
        rule=rule,
        max_number=5000,
        processors=["sequence_patterns", "digit_tensor_simple", "algebraic_features"],
    )

    return summary


def example_3_prime_numbers():
    """Example 3: Prime numbers with comprehensive analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Prime Numbers - Comprehensive Analysis")
    print("=" * 60)

    # Simple but effective primality test
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

    rule = MathematicalRule(
        func=is_prime,
        name="Prime Numbers",
        description="Numbers with exactly two positive divisors: 1 and themselves",
        examples=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
    )

    generator = UniversalDatasetGenerator()

    # Use ALL processors for comprehensive analysis
    summary = generator.generate_complete_pipeline(
        rule=rule,
        max_number=10000,
        processors=None,  # None = use all available processors
    )

    print(f"\n🔍 Prime Analysis Complete!")
    print(f"   Found: {len(summary['raw_numbers'])} primes up to 10,000")
    print(f"   Density: {len(summary['raw_numbers'])/10000:.4f}")

    return summary


def example_4_digit_based_rules():
    """Example 4: Digit-based mathematical rules"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Digit-Based Rules")
    print("=" * 60)

    # Rule 1: Numbers whose digit sum is a perfect square
    def digit_sum_perfect_square(n):
        digit_sum = sum(int(d) for d in str(n))
        root = int(digit_sum**0.5)
        return root * root == digit_sum

    rule1 = MathematicalRule(
        func=digit_sum_perfect_square,
        name="Digit Sum Perfect Square",
        description="Numbers whose digit sum is a perfect square",
        examples=[
            1,
            4,
            9,
            13,
            18,
            22,
            27,
            31,
            36,
            40,
        ],  # digit sums: 1,4,9,4,9,4,9,4,9,4
    )

    # Rule 2: Palindromic numbers
    rule2 = MathematicalRule(
        func=lambda n: str(n) == str(n)[::-1],
        name="Palindromic Numbers",
        description="Numbers that read the same forwards and backwards",
        examples=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66],
    )

    generator = UniversalDatasetGenerator()

    # Generate both rules with digit-focused processors
    for rule in [rule1, rule2]:
        print(f"\n🔢 Processing: {rule.name}")
        summary = generator.generate_complete_pipeline(
            rule=rule,
            max_number=2000,
            processors=["digit_tensor", "prefix_suffix_2_1", "algebraic_features"],
        )
        print(f"   Generated {summary['total_datasets']} datasets")


def example_5_comparative_study():
    """Example 5: Comparative study of related mathematical concepts"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Comparative Study - Powers Analysis")
    print("=" * 60)

    # Create multiple related rules for comparison
    rules = [
        MathematicalRule(
            lambda n: int(n**0.5) ** 2 == n,
            "Perfect Squares",
            "Numbers that are perfect squares: n = k²",
        ),
        MathematicalRule(
            lambda n: int(n ** (1 / 3)) ** 3 == n,
            "Perfect Cubes",
            "Numbers that are perfect cubes: n = k³",
        ),
        MathematicalRule(
            lambda n: n > 0 and (n & (n - 1)) == 0,
            "Powers of 2",
            "Numbers that are powers of 2: n = 2^k",
        ),
        MathematicalRule(
            lambda n: any(int(n ** (1 / k)) ** k == n for k in range(2, 7)),
            "Perfect Powers",
            "Numbers that are perfect powers: n = k^m for some k,m≥2",
        ),
    ]

    generator = UniversalDatasetGenerator()
    results = {}

    for rule in rules:
        print(f"\n📊 Analyzing: {rule.name}")
        summary = generator.generate_complete_pipeline(
            rule=rule,
            max_number=3000,
            processors=["algebraic_features", "sequence_patterns"],
        )

        results[rule.name] = {
            "count": len(summary["raw_numbers"]),
            "density": len(summary["raw_numbers"]) / 3000,
            "examples": summary["raw_numbers"][:15],
        }

    # Print comparison
    print(f"\n📈 COMPARISON RESULTS (up to 3000):")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name:<20}: {data['count']:4d} numbers ({data['density']:.4f} density)")
        print(f"{'':22} Examples: {data['examples']}")


def example_6_integration_with_discovery_engine():
    """Example 6: Using generated datasets with the discovery engine"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Integration with Discovery Engine")
    print("=" * 60)

    # Step 1: Generate a dataset
    rule = MathematicalRule(
        lambda n: n % 7 == 1,  # Numbers ≡ 1 (mod 7)
        name="Congruent 1 mod 7",
        description="Numbers that leave remainder 1 when divided by 7",
        examples=[1, 8, 15, 22, 29, 36, 43, 50, 57, 64],
    )

    generator = UniversalDatasetGenerator()
    summary = generator.generate_complete_pipeline(
        rule=rule, max_number=1000, processors=["algebraic_features"]
    )

    # Step 2: Load the generated dataset and use with discovery engine
    dataset_path = f"data/output/{rule.safe_name}/algebraic_features_up_to_1000.csv"

    try:
        # Check if file exists and load it
        if Path(dataset_path).exists():
            df = pd.read_csv(dataset_path, index_col=0)
            print(f"📊 Loaded dataset: {df.shape}")
            print(f"   Columns: {list(df.columns)[:10]}...")

            # Step 3: Use with discovery engine
            print(f"\n🤖 Running discovery engine on generated dataset...")

            # Create discovery engine using the same rule
            discovery_engine = UniversalMathDiscovery(
                target_function=rule.func, function_name=rule.name, max_number=1000
            )

            # Run discovery
            prediction_function = discovery_engine.run_complete_discovery()

            # Test predictions
            test_numbers = [1, 8, 15, 22, 29, 36, 43, 50, 100, 107, 114]
            print(f"\n🧪 Testing predictions:")
            for n in test_numbers:
                result = prediction_function(n)
                actual = rule.func(n)
                status = "✅" if result["prediction"] == actual else "❌"
                print(
                    f"   {status} {n:3d}: Predicted={result['prediction']}, "
                    f"Actual={actual}, Prob={result['probability']:.3f}"
                )

        else:
            print(f"❌ Dataset file not found: {dataset_path}")

    except Exception as e:
        print(f"❌ Error in discovery integration: {e}")


def example_7_custom_processor():
    """Example 7: Creating a custom processor"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Custom Processor")
    print("=" * 60)

    # Create a custom processor for modular arithmetic analysis
    class ModularArithmeticProcessor:
        def __init__(self, moduli=[2, 3, 5, 7, 11, 13]):
            self.moduli = moduli

        def process(self, numbers, metadata):
            max_num = max(numbers) if numbers else 1000

            features_list = []
            target_list = []

            for n in range(1, max_num + 1):
                features = {"number": n}

                # Add modular features
                for mod in self.moduli:
                    features[f"mod_{mod}"] = n % mod
                    features[f"is_zero_mod_{mod}"] = int(n % mod == 0)

                # Add combined modular features
                features["chinese_remainder"] = n % (self.moduli[0] * self.moduli[1])
                features["sum_of_residues"] = sum(n % mod for mod in self.moduli)

                features_list.append(features)
                target_list.append(1 if n in numbers else 0)

            df = pd.DataFrame(features_list)
            df["target"] = target_list
            return df

        def get_name(self):
            return f"modular_arithmetic_{len(self.moduli)}mods"

        def get_description(self):
            return f"Modular arithmetic features for moduli {self.moduli}"

    # Use the custom processor
    rule = MathematicalRule(
        lambda n: n % 6 == 1,  # Numbers ≡ 1 (mod 6)
        name="Congruent 1 mod 6",
        description="Numbers that leave remainder 1 when divided by 6",
        examples=[1, 7, 13, 19, 25, 31, 37, 43, 49, 55],
    )

    generator = UniversalDatasetGenerator()

    # Add our custom processor
    custom_processor = ModularArithmeticProcessor([2, 3, 5, 6, 7, 11])
    generator.processors["custom_modular"] = custom_processor

    # Generate using our custom processor
    summary = generator.generate_complete_pipeline(
        rule=rule, max_number=1000, processors=["custom_modular", "algebraic_features"]
    )

    print(f"✅ Used custom processor to generate {summary['total_datasets']} datasets!")


def run_all_examples():
    """Run all examples in sequence"""
    print("🚀 UNIVERSAL DATASET GENERATOR - COMPLETE EXAMPLES")
    print("=" * 70)
    print("Running all examples to demonstrate capabilities...")

    try:
        # Run each example
        examples = [
            ("Basic Usage", example_1_basic_usage),
            ("OEIS Sequence", example_2_custom_oeis_sequence),
            ("Prime Numbers", example_3_prime_numbers),
            ("Digit-Based Rules", example_4_digit_based_rules),
            ("Comparative Study", example_5_comparative_study),
            (
                "Discovery Engine Integration",
                example_6_integration_with_discovery_engine,
            ),
            ("Custom Processor", example_7_custom_processor),
        ]

        results = {}

        for name, example_func in examples:
            print(f"\n🎯 Running: {name}")
            try:
                result = example_func()
                results[name] = "✅ Success"
                print(f"   ✅ {name} completed successfully!")
            except Exception as e:
                results[name] = f"❌ Error: {e}"
                print(f"   ❌ {name} failed: {e}")

        # Summary
        print("\n" + "=" * 70)
        print("📊 EXAMPLES SUMMARY")
        print("=" * 70)

        success_count = 0
        for name, status in results.items():
            print(f"{name:<30}: {status}")
            if status.startswith("✅"):
                success_count += 1

        print(f"\n🎉 Completed {success_count}/{len(examples)} examples successfully!")

        if success_count == len(examples):
            print(
                "\n🚀 All examples completed! Your Universal Dataset Generator is working perfectly!"
            )
            print("\n📁 Check the following locations for generated datasets:")
            print("   data/raw/          - Raw number sets")
            print("   data/output/       - ML-ready datasets")

            print("\n🎯 Next steps:")
            print(
                "   1. Use the CLI: python scripts/generate_universal_datasets.py list"
            )
            print(
                "   2. Create custom rules: python scripts/generate_universal_datasets.py interactive"
            )
            print(
                "   3. Generate large datasets: python scripts/generate_universal_datasets.py generate 1 --max 100000"
            )
            print(
                "   4. Use with discovery engine: Load datasets and run pattern discovery"
            )

    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed with error: {e}")


def quick_demo():
    """Quick demo for immediate testing"""
    print("⚡ QUICK DEMO - Universal Dataset Generator")
    print("=" * 50)
    print("Generating a small dataset to verify everything works...")

    # Simple rule: even numbers
    rule = MathematicalRule(
        lambda n: n % 2 == 0,
        name="Even Numbers",
        description="Numbers divisible by 2",
        examples=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    )

    generator = UniversalDatasetGenerator()

    # Generate small dataset quickly
    summary = generator.generate_complete_pipeline(
        rule=rule,
        max_number=100,  # Very small for quick test
        processors=["prefix_suffix_2_1", "digit_tensor_simple"],
    )

    print(f"\n✅ Quick demo complete!")
    print(f"   Found: {len(summary['raw_numbers'])} even numbers in 1-100")
    print(f"   Generated: {summary['total_datasets']} ML datasets")
    print(f"   Location: data/output/{rule.safe_name}/")

    # Show what was created
    output_dir = Path("data/output") / rule.safe_name
    if output_dir.exists():
        files = list(output_dir.glob("*.csv"))
        print(f"\n📁 Created files:")
        for file in files:
            size_kb = file.stat().st_size / 1024
            print(f"   {file.name} ({size_kb:.1f} KB)")

    return summary


def main():
    """Main function - choose what to run"""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "all":
            run_all_examples()
        elif command == "demo":
            quick_demo()
        elif command in ["1", "basic"]:
            example_1_basic_usage()
        elif command in ["2", "oeis"]:
            example_2_custom_oeis_sequence()
        elif command in ["3", "primes"]:
            example_3_prime_numbers()
        elif command in ["4", "digits"]:
            example_4_digit_based_rules()
        elif command in ["5", "comparative"]:
            example_5_comparative_study()
        elif command in ["6", "integration"]:
            example_6_integration_with_discovery_engine()
        elif command in ["7", "custom"]:
            example_7_custom_processor()
        else:
            print(f"Unknown command: {command}")
    else:
        print("🧮 Universal Dataset Generator - Usage Examples")
        print("=" * 55)
        print("Choose what to run:")
        print()
        print(
            "  python examples/universal_dataset_examples.py demo        # Quick demo"
        )
        print(
            "  python examples/universal_dataset_examples.py all         # All examples"
        )
        print(
            "  python examples/universal_dataset_examples.py 1          # Basic usage"
        )
        print(
            "  python examples/universal_dataset_examples.py 2          # OEIS sequences"
        )
        print(
            "  python examples/universal_dataset_examples.py 3          # Prime numbers"
        )
        print(
            "  python examples/universal_dataset_examples.py 4          # Digit-based rules"
        )
        print(
            "  python examples/universal_dataset_examples.py 5          # Comparative study"
        )
        print(
            "  python examples/universal_dataset_examples.py 6          # Discovery integration"
        )
        print(
            "  python examples/universal_dataset_examples.py 7          # Custom processor"
        )
        print()
        print("For a quick test to verify everything works:")
        print("  python examples/universal_dataset_examples.py demo")


if __name__ == "__main__":
    main()
