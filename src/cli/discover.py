#!/usr/bin/env python3
"""
Discovery Engine CLI - Command Line Interface
===========================================

Enhanced command-line interface for the Mathematical Pattern Discovery Engine
that works seamlessly with Universal Dataset Generator outputs.

Usage:
    python scripts/discover_patterns.py <command> [options]

Author: Mathematical Pattern Discovery Team
"""

import sys
import argparse
from pathlib import Path
import json

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from core.discovery_engine import UniversalMathDiscovery
    from utils.path_utils import find_data_file, get_data_directory
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(
        "Make sure you're running from the project root and have installed dependencies."
    )
    sys.exit(1)


def list_available_datasets():
    """List all available datasets for analysis"""
    print("📋 AVAILABLE DATASETS FOR PATTERN DISCOVERY")
    print("=" * 60)

    data_dir = get_data_directory()
    output_dir = data_dir.parent / "output"

    datasets_found = 0

    # Check Universal Generator outputs
    if output_dir.exists():
        print("\n🧮 Universal Generator Datasets:")
        print("-" * 40)

        for rule_dir in output_dir.iterdir():
            if rule_dir.is_dir():
                csv_files = list(rule_dir.glob("*.csv"))
                if csv_files:
                    datasets_found += len(csv_files)
                    print(f"\n  📁 {rule_dir.name.replace('_', ' ').title()}")

                    for csv_file in csv_files:
                        if "metadata" not in csv_file.name:
                            size_mb = csv_file.stat().st_size / (1024 * 1024)

                            # Determine dataset type
                            if "prefix" in csv_file.name and "suffix" in csv_file.name:
                                dataset_type = "Prefix-Suffix Matrix"
                            elif "digit_tensor" in csv_file.name:
                                dataset_type = "Digit Tensor"
                            elif "sequence_patterns" in csv_file.name:
                                dataset_type = "Sequence Patterns"
                            elif "algebraic_features" in csv_file.name:
                                dataset_type = "Algebraic Features"
                            else:
                                dataset_type = "Unknown Type"

                            print(
                                f"    • {dataset_type}: {csv_file.name} ({size_mb:.1f} MB)"
                            )

    # Check legacy datasets
    if data_dir.exists():
        legacy_files = [
            f
            for f in data_dir.glob("*.csv")
            if "ml_dataset" in f.name or "dataset" in f.name
        ]
        if legacy_files:
            datasets_found += len(legacy_files)
            print("\n📊 Legacy ML Datasets:")
            print("-" * 25)

            for csv_file in legacy_files:
                size_mb = csv_file.stat().st_size / (1024 * 1024)
                print(f"    • {csv_file.name} ({size_mb:.1f} MB)")

    if datasets_found == 0:
        print("\n❌ No datasets found!")
        print("\n💡 Generate datasets first:")
        print("   python src/generators/universal_generator.py interactive")
        print("   python src/generators/prime_generator.py ml 10000")
    else:
        print(f"\n✅ Found {datasets_found} datasets available for analysis")
        print("\n🚀 Start discovery:")
        print("   python scripts/discover_patterns.py interactive")
        print("   python scripts/discover_patterns.py analyze <dataset_path>")


def analyze_single_dataset(dataset_path: str, config: dict):
    """Analyze a single dataset file"""
    print(f"🔬 ANALYZING DATASET: {dataset_path}")
    print("=" * 60)

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        # Try to find it in data directories
        found_file = find_data_file(dataset_file.name)
        if found_file:
            dataset_file = found_file
        else:
            print(f"❌ Dataset not found: {dataset_path}")
            return False

    try:
        import pandas as pd

        # Load dataset
        print("📊 Loading dataset...")
        df = pd.read_csv(
            dataset_file,
            index_col=0 if dataset_file.stat().st_size < 100 * 1024 * 1024 else None,
        )

        print(f"📈 Dataset shape: {df.shape}")
        print(
            f"📝 Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
        )

        # Determine analysis type
        if "target" in df.columns:
            return _analyze_ml_dataset(df, dataset_file, config)
        else:
            return _analyze_matrix_dataset(df, dataset_file, config)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


def _analyze_ml_dataset(df, dataset_file, config):
    """Analyze ML-ready dataset"""
    print("\n🤖 ML Dataset Analysis Mode")
    print("-" * 30)

    # Extract features and target
    target_col = "target"
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    positive_rate = y.mean()
    print(f"📊 Positive rate: {positive_rate:.4f} ({y.sum():,}/{len(y):,})")

    if positive_rate == 0 or positive_rate == 1:
        print("⚠️  No variation in target - cannot perform ML analysis")
        return False

    # Limit sample size if configured
    max_samples = config.get("max_samples", 50000)
    if len(df) > max_samples:
        print(f"🎯 Sampling {max_samples:,} rows from {len(df):,} total")
        df_sample = df.sample(n=max_samples, random_state=42)
        X = df_sample[feature_cols]
        y = df_sample[target_col]

    # Create synthetic target function (placeholder)
    def synthetic_target(n):
        return n % 2 == 0  # Placeholder - in practice we'd need the original rule

    # Initialize discovery engine
    print("🔧 Initializing discovery engine...")

    try:
        discoverer = UniversalMathDiscovery(
            target_function=synthetic_target,
            function_name=f"Analysis of {dataset_file.stem}",
            max_number=min(len(X), 10000),  # Reasonable limit
            embedding=config.get("embedding"),
        )

        # Override with our data
        discoverer.X = X
        discoverer.y = y.values
        discoverer.feature_names = feature_cols

        # Train models
        print("🤖 Training discovery models...")
        models = discoverer.train_discovery_models()

        # Analyze patterns
        print("🔍 Analyzing discovered patterns...")
        feature_importance = discoverer.analyze_mathematical_discoveries()

        # Extract rules if not in quick mode
        if not config.get("quick_mode", False):
            print("📜 Extracting mathematical rules...")
            rule_tree, rules = discoverer.extract_mathematical_rules()

        print("\n🏆 DISCOVERY RESULTS:")
        print("=" * 30)

        # Show model performance
        print("Model Performance:")
        for name, info in models.items():
            print(
                f"  {name}: Test Acc = {info['test_accuracy']:.3f}, AUC = {info['test_auc']:.3f}"
            )

        # Show top features
        if feature_importance is not None:
            print("\nTop Mathematical Features:")
            for _, row in feature_importance.head(10).iterrows():
                print(f"  {row['importance']:8.4f} | {row['feature']}")

        # Save results if configured
        if config.get("save_results", True):
            output_file = Path(f"discovery_results_{dataset_file.stem}.json")
            results = {
                "dataset": str(dataset_file),
                "shape": df.shape,
                "positive_rate": positive_rate,
                "model_performance": {
                    name: {
                        "train_acc": info["train_accuracy"],
                        "test_acc": info["test_accuracy"],
                        "test_auc": info["test_auc"],
                    }
                    for name, info in models.items()
                },
                "top_features": (
                    feature_importance.head(15).to_dict("records")
                    if feature_importance is not None
                    else []
                ),
            }

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")

        return True

    except Exception as e:
        print(f"❌ Discovery analysis failed: {e}")
        return False


def _analyze_matrix_dataset(df, dataset_file, config):
    """Analyze prefix-suffix matrix dataset"""
    print("\n🧮 Matrix Dataset Analysis Mode")
    print("-" * 30)

    # Basic statistics
    total_cells = df.shape[0] * df.shape[1]
    positive_cells = (df == 1).sum().sum()
    positive_rate = positive_cells / total_cells

    print(f"📊 Matrix dimensions: {df.shape[0]} × {df.shape[1]}")
    print(f"📈 Positive rate: {positive_rate:.4f} ({positive_cells:,}/{total_cells:,})")

    # Analyze patterns
    print("\n🔍 Pattern Analysis:")

    # Row patterns (prefixes)
    row_sums = df.sum(axis=1)
    best_rows = row_sums.nlargest(10)
    print(f"\nBest Prefixes (top 10):")
    for prefix, count in best_rows.items():
        print(f"  Prefix {prefix}: {count} matches")

    # Column patterns (suffixes)
    col_sums = df.sum(axis=0)
    best_cols = col_sums.nlargest(10)
    print(f"\nBest Suffixes (top 10):")
    for suffix, count in best_cols.items():
        print(f"  Suffix {suffix}: {count} matches")

    # Statistical insights
    print(f"\nStatistical Insights:")
    print(f"  Average matches per prefix: {row_sums.mean():.2f}")
    print(f"  Average matches per suffix: {col_sums.mean():.2f}")
    print(f"  Most productive prefix: {row_sums.idxmax()} ({row_sums.max()} matches)")
    print(f"  Most productive suffix: {col_sums.idxmax()} ({col_sums.max()} matches)")

    # Save results if configured
    if config.get("save_results", True):
        output_file = Path(f"matrix_analysis_{dataset_file.stem}.json")
        results = {
            "dataset": str(dataset_file),
            "matrix_shape": df.shape,
            "total_cells": int(total_cells),
            "positive_cells": int(positive_cells),
            "positive_rate": positive_rate,
            "best_prefixes": best_rows.head(10).to_dict(),
            "best_suffixes": best_cols.head(10).to_dict(),
            "row_stats": row_sums.describe().to_dict(),
            "col_stats": col_sums.describe().to_dict(),
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")

    return True


def analyze_folder(folder_path: str, config: dict):
    """Analyze all datasets in a folder"""
    print(f"📁 ANALYZING FOLDER: {folder_path}")
    print("=" * 60)

    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return False

    # Find all CSV files
    csv_files = list(folder.glob("*.csv"))
    # Filter out metadata files
    dataset_files = [f for f in csv_files if "metadata" not in f.name]

    if not dataset_files:
        print("❌ No dataset files found in folder")
        return False

    print(f"📊 Found {len(dataset_files)} datasets to analyze")

    results = []
    for i, dataset_file in enumerate(dataset_files, 1):
        print(f"\n[{i}/{len(dataset_files)}] Analyzing: {dataset_file.name}")
        print("-" * 50)

        try:
            success = analyze_single_dataset(str(dataset_file), config)
            results.append({"file": dataset_file.name, "success": success})
        except Exception as e:
            print(f"❌ Failed to analyze {dataset_file.name}: {e}")
            results.append(
                {"file": dataset_file.name, "success": False, "error": str(e)}
            )

    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n📊 FOLDER ANALYSIS SUMMARY")
    print("=" * 30)
    print(f"✅ Successfully analyzed: {successful}/{len(dataset_files)} datasets")

    if successful < len(dataset_files):
        failed = [r for r in results if not r["success"]]
        print(f"❌ Failed analyses:")
        for failure in failed:
            print(f"  • {failure['file']}")

    return successful > 0


def run_interactive_mode():
    """Run the interactive discovery engine"""
    print("🚀 Starting Interactive Discovery Mode...")
    print("=" * 50)

    try:
        # Import and run the interactive engine
        sys.path.insert(0, str(Path(__file__).parent))
        from interactive_discovery_engine import InteractiveDiscoveryEngine

        engine = InteractiveDiscoveryEngine()
        engine.run_interactive_session()

    except ImportError:
        print(
            "❌ Interactive engine not found. Please ensure interactive_discovery_engine.py is available."
        )
        print("💡 Fallback: Use the command-line options instead:")
        print("   python scripts/discover_patterns.py list")
        print("   python scripts/discover_patterns.py analyze <dataset_path>")
    except Exception as e:
        print(f"❌ Interactive mode failed: {e}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Mathematical Pattern Discovery Engine - Enhanced CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                              # List all available datasets
  %(prog)s interactive                       # Full interactive mode (recommended)
  %(prog)s analyze data/output/primes/algebraic_features_up_to_10000.csv
  %(prog)s folder data/output/perfect_squares/  # Analyze all datasets in folder
  %(prog)s analyze dataset.csv --quick      # Quick analysis mode
  %(prog)s analyze dataset.csv --embedding fourier --max-samples 100000
        """,
    )

    parser.add_argument(
        "command",
        choices=["list", "analyze", "folder", "interactive"],
        help="Command to execute",
    )
    parser.add_argument("path", nargs="?", help="Path to dataset file or folder")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Maximum samples to use for analysis (default: 50000)",
    )
    parser.add_argument(
        "--embedding", choices=["fourier", "pca"], help="Use embeddings for analysis"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick analysis mode (faster, less thorough)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to file"
    )

    args = parser.parse_args()

    # Build configuration
    config = {
        "max_samples": args.max_samples,
        "embedding": args.embedding,
        "quick_mode": args.quick,
        "save_results": not args.no_save,
    }

    print("🧮 MATHEMATICAL PATTERN DISCOVERY ENGINE")
    print("=" * 50)

    if args.command == "list":
        list_available_datasets()

    elif args.command == "interactive":
        run_interactive_mode()

    elif args.command == "analyze":
        if args.path is None:
            print("❌ Please specify a dataset file to analyze")
            parser.print_help()
            return

        success = analyze_single_dataset(args.path, config)
        if not success:
            sys.exit(1)

    elif args.command == "folder":
        if args.path is None:
            print("❌ Please specify a folder to analyze")
            parser.print_help()
            return

        success = analyze_folder(args.path, config)
        if not success:
            sys.exit(1)

    print("\n✨ Discovery analysis complete!")


if __name__ == "__main__":
    main()
