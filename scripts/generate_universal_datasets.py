#!/usr/bin/env python3
"""
Generate Universal Datasets - CLI Interface
===========================================

Command-line interface for the Universal Dataset Generator.
Easily create ML-ready datasets from any mathematical rule.

Usage:
    python scripts/generate_universal_datasets.py <command> [options]

Author: Mathematical Pattern Discovery Team
"""

import sys
import argparse
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from generators.universal_generator import (
        UniversalDatasetGenerator,
        MathematicalRule,
        create_example_rules,
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(
        "Make sure you're running from the project root and have installed dependencies."
    )
    sys.exit(1)


def create_custom_rules():
    """Create additional custom mathematical rules"""

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

    # Composite numbers
    rules.append(
        MathematicalRule(
            func=lambda n: n > 1 and not is_prime(n),
            name="Composite Numbers",
            description="Numbers with more than two positive divisors",
            examples=[4, 6, 8, 9, 10, 12, 14, 15, 16, 18],
        )
    )

    # Highly composite numbers (approximation)
    def count_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1 if i * i == n else 2
        return count

    def is_highly_composite_approx(n):
        if n <= 1:
            return False
        div_n = count_divisors(n)
        return all(count_divisors(k) < div_n for k in range(max(1, n - 10), n))

    rules.append(
        MathematicalRule(
            func=is_highly_composite_approx,
            name="Highly Composite (Approx)",
            description="Numbers with more divisors than any smaller number (approximate)",
            examples=[1, 2, 4, 6, 12, 24, 36, 48, 60, 120],
        )
    )

    # Numbers with digit sum equal to number of digits
    rules.append(
        MathematicalRule(
            func=lambda n: sum(int(d) for d in str(n)) == len(str(n)),
            name="Digit Sum Equals Digit Count",
            description="Numbers where sum of digits equals number of digits",
            examples=[
                1,
                10,
                100,
                1000,
            ],  # 1=1, 1+0=2 (no), 1+0+0=3 (no), 1+0+0+0=4 (no)
        )
    )

    # Actually, let's fix that example
    rules[-1] = MathematicalRule(
        func=lambda n: sum(int(d) for d in str(n)) == len(str(n)),
        name="Digit Sum Equals Digit Count",
        description="Numbers where sum of digits equals number of digits",
        examples=[1],  # Only 1 works for small numbers: sum=1, count=1
    )

    # Narcissistic numbers (numbers equal to sum of digits raised to power of digit count)
    def is_narcissistic(n):
        digits = [int(d) for d in str(n)]
        power = len(digits)
        return n == sum(d**power for d in digits)

    rules.append(
        MathematicalRule(
            func=is_narcissistic,
            name="Narcissistic Numbers",
            description="Numbers equal to sum of digits raised to power of digit count",
            examples=[1, 2, 3, 4, 5, 6, 7, 8, 9, 153, 371, 407],  # 153 = 1³ + 5³ + 3³
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

    print(f"🎯 Custom Rules ({len(custom_rules)}):")
    start_idx = len(builtin_rules) + 1
    for i, rule in enumerate(custom_rules, start_idx):
        print(f"  {i:2d}. {rule.name}")
        print(f"      {rule.description}")
        print(f"      Examples: {rule.examples[:8]}...")
        print()

    print(f"Total: {len(all_rules)} mathematical rules available")
    return all_rules


def generate_rule_datasets(rule_number, max_number, processors):
    """Generate datasets for a specific rule"""

    # Get all rules
    builtin_rules = create_example_rules()
    custom_rules = create_custom_rules()
    all_rules = builtin_rules + custom_rules

    if rule_number < 1 or rule_number > len(all_rules):
        print(f"❌ Invalid rule number. Use 1-{len(all_rules)}")
        return False

    rule = all_rules[rule_number - 1]

    print(f"🎯 Generating datasets for: {rule.name}")
    print(f"   Description: {rule.description}")
    print(f"   Max number: {max_number:,}")
    if processors:
        print(f"   Processors: {', '.join(processors)}")

    # Initialize generator
    generator = UniversalDatasetGenerator()

    # Generate complete pipeline
    summary = generator.generate_complete_pipeline(
        rule=rule, max_number=max_number, processors=processors
    )

    print(f"\n✅ Successfully generated {summary['total_datasets']} datasets!")
    print(f"📁 Location: data/output/{rule.safe_name}/")

    return True


def interactive_rule_creator():
    """Interactive mode to create custom rules"""
    print("\n🛠️  INTERACTIVE RULE CREATOR")
    print("=" * 40)
    print("Create a custom mathematical rule using Python code.")
    print("Your function should take an integer n and return True/False.")
    print()

    print("Examples:")
    print("  lambda n: n % 3 == 0                    # Multiples of 3")
    print("  lambda n: str(n) == str(n)[::-1]        # Palindromes")
    print("  lambda n: sum(int(d)**2 for d in str(n)) == n  # Happy numbers")
    print()

    try:
        # Get function code
        func_code = input("Enter your function (lambda n: ...): ").strip()
        if not func_code.startswith("lambda"):
            func_code = "lambda n: " + func_code

        # Test the function
        func = eval(func_code)

        # Test on small numbers
        test_results = []
        for i in range(1, 21):
            try:
                result = func(i)
                if result:
                    test_results.append(i)
            except:
                pass

        print(f"\nTest results (1-20): {test_results}")

        if not test_results:
            print("⚠️  No matches found in 1-20. Rule might be too restrictive.")
            return None

        # Get metadata
        name = input("Rule name: ").strip()
        description = input("Description: ").strip()

        # Create rule
        rule = MathematicalRule(
            func=func, name=name, description=description, examples=test_results[:10]
        )

        print(f"\n✅ Created rule: {rule.name}")
        return rule

    except Exception as e:
        print(f"❌ Error creating rule: {e}")
        return None


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Generate ML-ready datasets from mathematical rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                           # List all available rules
  %(prog)s generate 1 --max 10000        # Generate perfect squares up to 10,000
  %(prog)s generate 5 --processors prefix_suffix_2_1,digit_tensor
  %(prog)s interactive                    # Create custom rule interactively
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

        success = generate_rule_datasets(args.rule_number, args.max, processors)
        if not success:
            sys.exit(1)

    elif args.command == "interactive":
        rule = interactive_rule_creator()
        if rule:
            print(f"\nGenerate datasets for '{rule.name}'? (y/N): ", end="")
            if input().strip().lower() == "y":
                generator = UniversalDatasetGenerator()
                summary = generator.generate_complete_pipeline(
                    rule, args.max, processors
                )
                print(f"✅ Generated {summary['total_datasets']} datasets!")

    elif args.command == "demo":
        print("🎯 Running quick demo with Perfect Squares...")
        success = generate_rule_datasets(1, min(args.max, 5000), processors)
        if success:
            print("\n🎉 Demo complete! Check data/output/perfect_squares/ for results.")


if __name__ == "__main__":
    main()
