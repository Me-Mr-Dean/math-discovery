#!/usr/bin/env python3
"""
Prime CSV Generator - Multiple Dataset Formats
Reads 1m.csv containing prime numbers and generates four different CSV formats
based on different mathematical insights about prime number patterns.
"""

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None
try:
    import numpy as np
except Exception:  # pragma: no cover
    import numpy as np  # type: ignore
import os
import sys
from pathlib import Path
import time


class PrimeGenerator:
    """Minimal stub used for unit tests."""

    def __init__(self):
        pass

    def generate_primes(self, n: int):
        raise NotImplementedError

class PrimeCSVGenerator:
    def __init__(self, input_file="../../data/raw/1m.csv", output_dir="output"):
        """
        Initialize the Prime CSV Generator

        Args:
            input_file (str): Path to the input CSV file containing prime data
            output_dir (str): Directory where output CSV files will be saved
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.prime_set = set()
        self.max_prime = 0
        self.total_primes = 0

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)

    def load_primes(self):
        """Load prime numbers from the input CSV file"""
        print(f"Loading primes from {self.input_file}...")

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file '{self.input_file}' not found!")

        try:
            # Read the CSV file
            df = pd.read_csv(self.input_file)
            print(f"Loaded CSV with columns: {list(df.columns)}")

            # Extract prime numbers from the 'Num' column
            primes = df["Num"].tolist()
            self.prime_set = set(primes)
            self.max_prime = max(primes)
            self.total_primes = len(primes)

            print(f"Successfully loaded {self.total_primes} prime numbers")
            print(f"Range: 2 to {self.max_prime}")

        except Exception as e:
            raise Exception(f"Error loading primes: {str(e)}")

    def generate_column_sets(self, max_ending=50):
        """
        Generate the four different column sets

        Args:
            max_ending (int): Maximum ending digit to consider

        Returns:
            dict: Dictionary of column sets
        """
        # 1. Classic odd endings (excluding 5 since primes > 5 can't end in 5)
        odd_endings = [1, 3, 7, 9]

        # 2. All single digits
        all_digits = list(range(10))  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

        # 3. Extended prime-friendly endings: all numbers ending in 1,3,7,9
        prime_friendly_endings = []
        for num in range(1, max_ending + 1):
            last_digit = num % 10
            if last_digit in [1, 3, 7, 9]:
                prime_friendly_endings.append(num)

        # 4. All odd numbers (excluding even numbers completely)
        odd_numbers = [
            i for i in range(1, max_ending + 1, 2)
        ]  # 1, 3, 5, 7, 9, 11, 13, ...

        return {
            "odd_endings": odd_endings,
            "all_digits": all_digits,
            "prime_endings": prime_friendly_endings,
            "odd_numbers": odd_numbers,
        }

    def generate_prime_matrix(
        self, max_number, endings, dataset_name, include_metadata=False
    ):
        """
        Generate a matrix where rows=tens, columns=endings

        Args:
            max_number (int): Maximum number to include
            endings (list): List of ending digits/numbers to include as columns
            dataset_name (str): Name of the dataset for logging
            include_metadata (bool): Include additional metadata columns

        Returns:
            pandas.DataFrame: Prime matrix with proper row/column labels
        """
        # Always use proper concatenation for consistency
        # Numbers are formed by concatenating row_value with ending
        # For example: row 429 + ending 17 = 42917
        # Start from row 1 to avoid single digits

        base = 10  # Starting base for row calculation
        max_rows = max_number // base
        start_row = 1  # Skip row 0 (single digits 0-9)

        # Ensure we don't exceed the max_number when creating the last row
        if max_rows * base > max_number:
            max_rows -= 1

        print(f"Generating {dataset_name}: {max_rows} rows x {len(endings)} columns")
        print(
            f"  Base: {base}, Endings: {endings[:10]}{'...' if len(endings) > 10 else ''}"
        )
        print(
            f"  Number formation: row_value concatenated with ending (e.g., 429 + 17 = 42917)"
        )

        # Create the data matrix
        data = []
        row_labels = []

        # FIX 1: Use start_row in the range to exclude row 0
        for row_val in range(start_row, max_rows + 1):
            row_data = []
            row_labels.append(row_val)

            for ending in endings:
                # Calculate proper base multiplier based on ending digit count
                ending_str = str(ending)
                ending_digits = len(ending_str)
                base_multiplier = 10**ending_digits

                # Form number by concatenating row_val with ending
                number = row_val * base_multiplier + ending

                if number <= max_number:
                    # 1 if prime, 0 if not prime
                    is_prime = 1 if number in self.prime_set else 0
                    row_data.append(is_prime)
                else:
                    # Empty for numbers beyond max_number
                    row_data.append("")

            if include_metadata:
                # Add metadata
                range_start = row_val * base
                range_end = row_val * base + (base - 1)
                valid_numbers = [
                    row_val * base + ending
                    for ending in endings
                    if row_val * base + ending <= max_number
                ]
                prime_count = sum(1 for num in valid_numbers if num in self.prime_set)
                prime_density = prime_count / len(valid_numbers) if valid_numbers else 0

                row_data.extend(
                    [range_start, range_end, prime_count, round(prime_density, 4)]
                )

            data.append(row_data)

            # Progress update
            if row_val % 1000 == 0 and row_val > 0:
                print(f"    Progress: {row_val}/{max_rows} rows processed")

        # Create column labels
        col_labels = [str(ending) for ending in endings]
        if include_metadata:
            col_labels.extend(
                ["range_start", "range_end", "prime_count", "prime_density"]
            )

        # Create DataFrame with proper row labels (starting from 1)
        df = pd.DataFrame(data, columns=col_labels, index=row_labels)
        df.index.name = "tens"  # Always tens regardless of ending size

        print(f"  Generated matrix: {len(df)} rows x {len(df.columns)} columns")
        print(f"  Row range: {df.index[0]} to {df.index[-1]}")

        return df

    def save_matrix_to_csv(self, df, filename, description):
        """Save a matrix to CSV file"""
        output_path = self.output_dir / filename
        print(f"  Saving {description} to {output_path}...")

        # Save with row index included
        df.to_csv(output_path, index=True)

        # Get file info
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(
            f"    Saved: {len(df)} rows x {len(df.columns)} columns, {file_size:.1f} MB"
        )

        return df

    def generate_dataset_1_odd_endings(self, sample_size=None, include_metadata=False):
        """Dataset 1: Odd endings only (1, 3, 7, 9)"""
        max_num = sample_size if sample_size else self.max_prime
        endings = [1, 3, 7, 9]

        print(f"\n=== Dataset 1: Odd Endings (1, 3, 7, 9) ===")
        print("Mathematical insight: Primes > 2 must end in 1, 3, 7, or 9")

        df = self.generate_prime_matrix(
            max_num, endings, "Odd Endings", include_metadata
        )

        filename = f"dataset1_odd_endings{'_sample' if sample_size else ''}.csv"
        if include_metadata:
            filename = f"../../data/raw/ml_dataset1_odd_endings{'_sample' if sample_size else ''}.csv"

        self.save_matrix_to_csv(df, filename, "Dataset 1 - Odd Endings")
        return df

    def generate_dataset_2_all_digits(self, sample_size=None, include_metadata=False):
        """Dataset 2: All single digits (0-9)"""
        max_num = sample_size if sample_size else self.max_prime
        endings = list(range(10))  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

        print(f"\n=== Dataset 2: All Digits (0-9) ===")
        print("Complete coverage: All possible single-digit endings")

        df = self.generate_prime_matrix(
            max_num, endings, "All Digits", include_metadata
        )

        filename = f"dataset2_all_digits{'_sample' if sample_size else ''}.csv"
        if include_metadata:
            filename = f"../../data/raw/ml_dataset2_all_digits{'_sample' if sample_size else ''}.csv"

        self.save_matrix_to_csv(df, filename, "Dataset 2 - All Digits")
        return df

    def generate_dataset_3_prime_endings(
        self, sample_size=None, include_metadata=False, max_ending=100
    ):
        """Dataset 3: Extended prime-friendly endings (all numbers ending in 1,3,7,9)"""
        max_num = sample_size if sample_size else self.max_prime

        # Generate all numbers up to max_ending that end in 1, 3, 7, or 9
        # This includes: 1,3,7,9, 11,13,17,19, 21,23,27,29, 31,33,37,39, etc.
        prime_friendly_endings = []

        for num in range(1, max_ending + 1):
            last_digit = num % 10
            if last_digit in [1, 3, 7, 9]:
                prime_friendly_endings.append(num)

        print(
            f"\n=== Dataset 3: Extended Prime-Friendly Endings (up to {max_ending}) ==="
        )
        print(f"Mathematical insight: All numbers ending in 1, 3, 7, or 9")
        print(f"Pattern: 1,3,7,9, 11,13,17,19, 21,23,27,29, 31,33,37,39, ...")
        print(
            f"Endings: {prime_friendly_endings[:20]}{'...' if len(prime_friendly_endings) > 20 else ''}"
        )
        print(f"Total columns: {len(prime_friendly_endings)}")

        df = self.generate_prime_matrix(
            max_num,
            prime_friendly_endings,
            "Extended Prime-Friendly Endings",
            include_metadata,
        )

        filename = f"dataset3_prime_endings{'_sample' if sample_size else ''}.csv"
        if include_metadata:
            filename = (
                f"../../data/raw/ml_dataset3_prime_endings{'_sample' if sample_size else ''}.csv"
            )

        self.save_matrix_to_csv(
            df, filename, "Dataset 3 - Extended Prime-Friendly Endings"
        )
        return df

    def show_sample_data(self, df, title, max_rows=5):
        """Display sample data from a DataFrame"""
        print(f"\n{title} (first {max_rows} rows, starting from row 1):")
        print("-" * 60)

        # Show just the first few columns if there are many
        display_df = df.iloc[:max_rows, : min(10, len(df.columns))]
        print(display_df.to_string())

        if len(df.columns) > 10:
            print(f"... and {len(df.columns) - 10} more columns")

        # Show examples starting from row 1
        if 1 in df.index:
            print(f"\nExample interpretations (row 1 = 10s):")
            if "1" in df.columns:
                print(f"  Row 1, Column '1' = number 11, value = {df.loc[1, '1']}")
            if "3" in df.columns:
                print(f"  Row 1, Column '3' = number 13, value = {df.loc[1, '3']}")
            if "17" in df.columns:
                print(f"  Row 1, Column '17' = number 117, value = {df.loc[1, '17']}")
            elif "7" in df.columns:
                print(f"  Row 1, Column '7' = number 17, value = {df.loc[1, '7']}")

    def generate_all_datasets(
        self, sample_size=None, include_metadata=False, max_ending=50
    ):
        """Generate all three datasets"""
        print("Prime CSV Generator - Three Dataset Generation")
        print("=" * 60)

        if sample_size:
            print(f"Generating sample datasets for numbers 0-{sample_size}")
        else:
            print(f"Generating full datasets up to {self.max_prime}")

        print(f"Max ending digit/number: {max_ending}")
        print("\nDataset Concepts:")
        print("1. Odd Endings (1,3,7,9) - Classic prime theory")
        print("2. All Digits (0-9) - Complete coverage")
        print(
            "3. Extended Prime-Friendly (1,3,7,9,11,13,17,19,21,23,27,29,...) - All numbers ending in 1,3,7,9"
        )

        # Generate all three datasets
        datasets = {}

        datasets["odd_endings"] = self.generate_dataset_1_odd_endings(
            sample_size, include_metadata
        )
        datasets["all_digits"] = self.generate_dataset_2_all_digits(
            sample_size, include_metadata
        )
        datasets["prime_endings"] = self.generate_dataset_3_prime_endings(
            sample_size, include_metadata, max_ending
        )

        # Show samples
        if sample_size or max_ending <= 20:
            for name, df in datasets.items():
                self.show_sample_data(df, f"Dataset: {name.replace('_', ' ').title()}")

        print(f"\n✅ All datasets generated successfully!")
        print(f"📁 Files saved in: {self.output_dir.absolute()}")

        return datasets

    def show_statistics(self):
        """Display file statistics"""
        print("\n" + "=" * 50)
        print("PRIME DATASET STATISTICS")
        print("=" * 50)
        print(f"Total prime entries: {self.total_primes:,}")
        print(f"Maximum prime: {self.max_prime:,}")

        print(f"\nDataset Dimensions (estimated, excluding row 0):")
        max_tens = self.max_prime // 10

        print(f"  Dataset 1 (Odd endings): {max_tens:,} rows x 4 columns")
        print(f"  Dataset 2 (All digits): {max_tens:,} rows x 10 columns")
        print(
            f"  Dataset 3 (Extended prime-friendly ≤{max_ending//10*4}): {max_tens:,} rows x ~{max_ending//10*4} columns"
        )

        print(f"\nMathematical Insights:")
        print(f"  - Starting from row 1 (10s) to avoid special single-digit cases")
        print(f"  - Dataset 1: Focuses on known prime ending patterns")
        print(f"  - Dataset 2: Complete baseline for comparison")
        print(f"  - Dataset 3: Tests extended prime-friendly ending patterns")
        print("=" * 50)


def main():
    """Main function with command line interface"""
    if len(sys.argv) < 2:
        print("Prime CSV Generator - Three Mathematical Datasets")
        print("=" * 60)
        print("Generates three datasets based on different prime number insights:")
        print("")
        print("📊 Dataset 1: Odd Endings (1, 3, 7, 9)")
        print("   - Classic prime theory: primes > 2 must end in 1,3,7,9")
        print("")
        print("📊 Dataset 2: All Digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)")
        print("   - Complete coverage for baseline comparison")
        print("")
        print(
            "📊 Dataset 3: Extended Prime-Friendly Endings (1,3,7,9,11,13,17,19,21,23,27,29,...)"
        )
        print("   - All numbers ending in 1, 3, 7, or 9 (extends Dataset 1 pattern)")
        print("")
        print("Usage:")
        print("  python prime_generator.py sample [size] [max_ending]")
        print("  python prime_generator.py ml [size] [max_ending]")
        print("  python prime_generator.py full [max_ending]")
        print("  python prime_generator.py stats")
        print("")
        print("Examples:")
        print(
            "  python prime_generator.py sample 1000 30      # Sample with endings up to 30"
        )
        print(
            "  python prime_generator.py ml 5000 50          # ML dataset, endings up to 50"
        )
        print(
            "  python prime_generator.py full 100            # Full dataset, endings up to 100"
        )
        return

    command = sys.argv[1].lower()
    generator = PrimeCSVGenerator()

    try:
        generator.load_primes()

        if command == "sample":
            sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
            max_ending = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            generator.generate_all_datasets(
                sample_size, include_metadata=False, max_ending=max_ending
            )

        elif command == "ml":
            sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
            max_ending = int(sys.argv[3]) if len(sys.argv) > 3 else 50
            generator.generate_all_datasets(
                sample_size, include_metadata=True, max_ending=max_ending
            )

        elif command == "full":
            max_ending = int(sys.argv[2]) if len(sys.argv) > 2 else 50

            generator.show_statistics()
            response = input(
                f"\nGenerate all three full datasets? This may take several minutes (y/N): "
            )
            if response.lower() != "y":
                print("Generation cancelled.")
                return

            generator.generate_all_datasets(
                None, include_metadata=False, max_ending=max_ending
            )

        elif command == "stats":
            generator.show_statistics()

        else:
            print(f"Unknown command: {command}")
            print("Use 'sample', 'ml', 'full', or 'stats'")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure '../../data/raw/1m.csv' is in the same directory as this script.")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
