#!/usr/bin/env python3
"""
Prefix-Suffix Prime Correlation Analysis
Analyze if certain prefix + suffix combinations correlate with primality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class PrefixSuffixPrimeAnalyzer:
    def __init__(self, dataset_path):
        """Initialize the prefix-suffix analyzer"""
        self.dataset_path = dataset_path
        self.df = None
        self.analysis_results = {}

    def load_data(self):
        """Load and prepare the matrix data"""
        print("Loading prefix-suffix matrix data...")
        self.df = pd.read_csv(self.dataset_path, index_col=0)

        # Remove metadata columns
        metadata_columns = ["range_start", "range_end", "prime_count", "prime_density"]
        data_columns = [col for col in self.df.columns if col not in metadata_columns]
        self.df = self.df[data_columns]

        print(f"Matrix shape: {self.df.shape}")
        print(f"Prefixes (rows): {self.df.index.min()} to {self.df.index.max()}")
        print(f"Suffixes (columns): {data_columns}")

        return self.df

    def analyze_suffix_performance(self):
        """Analyze how well each suffix performs across all prefixes"""
        print("\n🔍 ANALYZING SUFFIX PERFORMANCE:")
        print("=" * 50)

        suffix_stats = {}

        for suffix in self.df.columns:
            suffix_data = self.df[suffix].dropna()
            suffix_data = suffix_data[suffix_data != ""]

            if len(suffix_data) > 0:
                prime_rate = suffix_data.mean()
                total_numbers = len(suffix_data)
                prime_count = suffix_data.sum()

                suffix_stats[suffix] = {
                    "suffix": int(suffix),
                    "prime_rate": prime_rate,
                    "prime_count": int(prime_count),
                    "total_count": total_numbers,
                    "last_digit": int(suffix) % 10,
                }

        # Convert to DataFrame and sort
        suffix_df = pd.DataFrame(suffix_stats).T
        suffix_df = suffix_df.sort_values("prime_rate", ascending=False)

        print("SUFFIX PERFORMANCE RANKING:")
        print("-" * 40)
        print("Suffix | Prime Rate | Count | Last Digit")
        print("-" * 40)

        for _, row in suffix_df.iterrows():
            print(
                f"{int(row['suffix']):6d} | {row['prime_rate']:9.4f} | {int(row['prime_count']):5d} | {int(row['last_digit']):10d}"
            )

        self.analysis_results["suffix_performance"] = suffix_df
        return suffix_df

    def analyze_prefix_performance(self):
        """Analyze how well each prefix performs across all suffixes"""
        print("\n🔍 ANALYZING PREFIX PERFORMANCE:")
        print("=" * 50)

        prefix_stats = {}

        for prefix in self.df.index:
            prefix_data = self.df.loc[prefix].dropna()
            prefix_data = prefix_data[prefix_data != ""]

            if len(prefix_data) > 0:
                prime_rate = prefix_data.mean()
                total_numbers = len(prefix_data)
                prime_count = prefix_data.sum()

                prefix_stats[prefix] = {
                    "prefix": prefix,
                    "prime_rate": prime_rate,
                    "prime_count": int(prime_count),
                    "total_count": total_numbers,
                    "prefix_mod_6": prefix % 6,
                    "prefix_mod_10": prefix % 10,
                }

        # Convert to DataFrame and analyze patterns
        prefix_df = pd.DataFrame(prefix_stats).T

        # Group by prefix properties
        print("PREFIX PERFORMANCE BY MOD 6:")
        print("-" * 30)
        mod6_groups = prefix_df.groupby("prefix_mod_6")["prime_rate"].agg(
            ["mean", "std", "count"]
        )
        print(mod6_groups)

        print("\nPREFIX PERFORMANCE BY MOD 10:")
        print("-" * 30)
        mod10_groups = prefix_df.groupby("prefix_mod_10")["prime_rate"].agg(
            ["mean", "std", "count"]
        )
        print(mod10_groups)

        # Show best and worst prefixes
        top_prefixes = prefix_df.nlargest(10, "prime_rate")
        bottom_prefixes = prefix_df.nsmallest(10, "prime_rate")

        print("\nTOP 10 BEST PREFIXES:")
        print("-" * 25)
        for _, row in top_prefixes.iterrows():
            print(f"Prefix {int(row['prefix']):5d}: {row['prime_rate']:.4f} prime rate")

        print("\nTOP 10 WORST PREFIXES:")
        print("-" * 25)
        for _, row in bottom_prefixes.iterrows():
            print(f"Prefix {int(row['prefix']):5d}: {row['prime_rate']:.4f} prime rate")

        self.analysis_results["prefix_performance"] = prefix_df
        return prefix_df

    def analyze_prefix_suffix_interactions(self):
        """Analyze specific prefix-suffix combinations"""
        print("\n🔍 ANALYZING PREFIX-SUFFIX INTERACTIONS:")
        print("=" * 50)

        interactions = []

        for prefix in self.df.index:
            for suffix in self.df.columns:
                value = self.df.loc[prefix, suffix]

                if pd.notna(value) and value != "":
                    # Calculate the actual number - simple concatenation
                    # Row 42 + Column "7" = 427 (not 42917!)
                    suffix_int = int(suffix)
                    actual_number = int(str(prefix) + str(suffix_int))

                    interactions.append(
                        {
                            "prefix": prefix,
                            "suffix": suffix_int,
                            "number": actual_number,
                            "is_prime": int(value),
                            "prefix_mod_6": prefix % 6,
                            "suffix_mod_6": suffix_int % 6,
                            "prefix_last_digit": prefix % 10,
                            "suffix_last_digit": suffix_int % 10,
                        }
                    )

        interaction_df = pd.DataFrame(interactions)

        # Analyze interaction patterns
        print("PREFIX MOD 6 × SUFFIX MOD 6 INTERACTION:")
        print("-" * 45)

        interaction_pivot = interaction_df.pivot_table(
            values="is_prime",
            index="prefix_mod_6",
            columns="suffix_mod_6",
            aggfunc="mean",
        )
        print(interaction_pivot.round(4))

        print("\nPREFIX LAST DIGIT × SUFFIX LAST DIGIT:")
        print("-" * 40)

        digit_interaction = interaction_df.pivot_table(
            values="is_prime",
            index="prefix_last_digit",
            columns="suffix_last_digit",
            aggfunc="mean",
        )
        print(digit_interaction.round(4))

        self.analysis_results["interactions"] = interaction_df
        return interaction_df

    def find_best_combinations(self, min_samples=50):
        """Find the best prefix-suffix combinations"""
        print(f"\n🏆 BEST PREFIX-SUFFIX COMBINATIONS (min {min_samples} samples):")
        print("=" * 60)

        if "interactions" not in self.analysis_results:
            self.analyze_prefix_suffix_interactions()

        interaction_df = self.analysis_results["interactions"]

        # Group by prefix-suffix pairs
        combination_stats = (
            interaction_df.groupby(["prefix", "suffix"])
            .agg({"is_prime": ["mean", "sum", "count"], "number": "first"})
            .round(4)
        )

        combination_stats.columns = [
            "prime_rate",
            "prime_count",
            "total_count",
            "example_number",
        ]
        combination_stats = combination_stats[
            combination_stats["total_count"] >= min_samples
        ]
        combination_stats = combination_stats.sort_values("prime_rate", ascending=False)

        print("TOP PREFIX-SUFFIX COMBINATIONS:")
        print("-" * 50)
        print("Prefix | Suffix | Prime Rate | Count | Example Number")
        print("-" * 50)

        for (prefix, suffix), row in combination_stats.head(20).iterrows():
            print(
                f"{int(prefix):6d} | {int(suffix):6d} | {row['prime_rate']:9.4f} | {int(row['total_count']):5d} | {int(row['example_number']):12d}"
            )

        print("\nWORST PREFIX-SUFFIX COMBINATIONS:")
        print("-" * 50)

        for (prefix, suffix), row in combination_stats.tail(10).iterrows():
            print(
                f"{int(prefix):6d} | {int(suffix):6d} | {row['prime_rate']:9.4f} | {int(row['total_count']):5d} | {int(row['example_number']):12d}"
            )

        return combination_stats

    def train_prefix_suffix_model(self):
        """Train a model to predict primality based on prefix-suffix features"""
        print("\n🤖 TRAINING PREFIX-SUFFIX CORRELATION MODEL:")
        print("=" * 50)

        if "interactions" not in self.analysis_results:
            self.analyze_prefix_suffix_interactions()

        interaction_df = self.analysis_results["interactions"]

        # Create features focused on prefix-suffix relationships
        features = interaction_df[
            [
                "prefix",
                "suffix",
                "prefix_mod_6",
                "suffix_mod_6",
                "prefix_last_digit",
                "suffix_last_digit",
            ]
        ].copy()

        # Add interaction features
        features["prefix_suffix_mod6_product"] = (
            features["prefix_mod_6"] * features["suffix_mod_6"]
        )
        features["prefix_suffix_digit_sum"] = (
            features["prefix_last_digit"] + features["suffix_last_digit"]
        )
        features["prefix_suffix_digit_product"] = (
            features["prefix_last_digit"] * features["suffix_last_digit"]
        )

        target = interaction_df["is_prime"]

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Analyze feature importance
        feature_importance = pd.DataFrame(
            {"feature": features.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("PREFIX-SUFFIX FEATURE IMPORTANCE:")
        print("-" * 35)
        for _, row in feature_importance.iterrows():
            print(f"{row['importance']:8.4f} | {row['feature']}")

        # Performance
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        print(f"\nModel Performance:")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        return model, feature_importance

    def run_complete_analysis(self):
        """Run the complete prefix-suffix analysis"""
        print("🔬 PREFIX-SUFFIX PRIME CORRELATION ANALYSIS")
        print("=" * 60)
        print("Analyzing if certain prefix + suffix combinations")
        print("are more likely to produce prime numbers...")

        # Load data
        self.load_data()

        # Run all analyses
        self.analyze_suffix_performance()
        self.analyze_prefix_performance()
        self.analyze_prefix_suffix_interactions()
        self.find_best_combinations()
        self.train_prefix_suffix_model()

        print("\n" + "=" * 60)
        print("🎉 PREFIX-SUFFIX ANALYSIS COMPLETE!")
        print("=" * 60)

        return self.analysis_results


def main():
    """Main analysis function"""
    import sys

    if len(sys.argv) < 2:
        print("Prefix-Suffix Prime Correlation Analysis")
        print("Usage: python prefix_suffix_analysis.py <dataset_path>")
        print(
            "Example: python prefix_suffix_analysis.py output/dataset1_odd_endings_sample.csv"
        )
        return

    dataset_path = sys.argv[1]

    analyzer = PrefixSuffixPrimeAnalyzer(dataset_path)
    results = analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
