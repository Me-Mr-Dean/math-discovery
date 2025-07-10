#!/usr/bin/env python3
"""
Quick Prime Analysis - Simple script to get started
"""

from src.analyzers.prime_analyzer import PurePrimeMLDiscovery


def quick_analysis(dataset_path):
    """Run a quick analysis on your prime dataset"""

    print("🔍 Quick Prime Pattern Analysis")
    print("=" * 50)

    # Initialize analyzer
    analyzer = PurePrimeMLDiscovery(dataset_path)

    # Run full analysis
    prime_function = analyzer.run_pure_discovery()

    print("\n🎯 Testing the extracted function:")
    print("-" * 30)

    # Test some interesting numbers
    test_cases = [
        127,  # Known prime
        129,  # Not prime (3 × 43)
        131,  # Known prime
        133,  # Not prime (7 × 19)
        1009,  # Known prime
        1001,  # Not prime (7 × 11 × 13)
        2017,  # Known prime
        2021,  # Not prime (43 × 47)
    ]

    for num in test_cases:
        result = prime_function(num)
        status = "✅ Prime" if result["prediction"] else "❌ Not Prime"
        confidence = result["confidence"].upper()
        probability = result["probability"]

        print(f"{num:4d}: {status} (prob: {probability:.3f}, confidence: {confidence})")

    print(f"\n🔧 Model Performance Summary:")
    print(f"   Features used: {len(analyzer.feature_names)}")
    print(f"   Training samples: {len(analyzer.X)}")
    print(f"   Prime ratio in data: {analyzer.y.mean():.3f}")

    for name, model_info in analyzer.models.items():
        acc = model_info["test_accuracy"]
        auc = model_info["test_auc"]
        print(f"   {name.replace('_', ' ').title()}: {acc:.3f} accuracy, {auc:.3f} AUC")

    print(f"\n🚀 Function ready! Use prime_function(number) to predict.")

    return prime_function


def batch_test(prime_function, start=100, end=200):
    """Test the function on a range of numbers"""
    print(f"\n🧪 Batch testing numbers {start}-{end}:")
    print("-" * 40)

    predictions = []
    for num in range(start, end + 1):
        result = prime_function(num)
        if result["prediction"] == 1:  # Predicted as prime
            predictions.append((num, result["probability"]))

    # Sort by probability (highest confidence first)
    predictions.sort(key=lambda x: x[1], reverse=True)

    print("Numbers predicted as PRIME (sorted by confidence):")
    for num, prob in predictions[:20]:  # Top 20
        print(f"  {num:3d} (confidence: {prob:.3f})")

    if len(predictions) > 20:
        print(f"  ... and {len(predictions) - 20} more")

    return predictions


if __name__ == "__main__":
    # Quick start - adjust the path to your dataset
    dataset_path = "../../data/raw/ml_dataset1_odd_endings.csv"  # Change this to your actual file

    try:
        # Run analysis
        prime_function = quick_analysis(dataset_path)

        # Batch test
        batch_test(prime_function, 100, 200)

        # Interactive mode
        print("\n💬 Interactive mode (type 'quit' to exit):")
        while True:
            try:
                user_input = input("Test number: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                number = int(user_input)
                result = prime_function(number)

                status = "✅ PRIME" if result["prediction"] else "❌ NOT PRIME"
                print(f"  {number} → {status} (prob: {result['probability']:.3f})")

            except ValueError:
                print("  Please enter a valid number")
            except KeyboardInterrupt:
                break

        print("\n✨ Analysis complete! Your function is ready to use.")

    except FileNotFoundError:
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please update the dataset_path variable with your actual file path")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(
            "Make sure you have the required packages: pip install pandas numpy scikit-learn matplotlib seaborn"
        )
