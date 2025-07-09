# Prime CSV Generator

Convert prime number data from `1m.csv` into ML-ready CSV formats with proper headers and enhanced features.

## Quick Start

1. **Setup Environment**:

   ```bash
   pip install pandas numpy openpyxl
   ```

2. **Place Files**: Put `1m.csv` and `prime_generator.py` in the same directory

3. **Test with Sample**:

   ```bash
   python prime_generator.py sample 5000 50
   ```

4. **Generate ML Dataset**:

   ```bash
   python prime_generator.py ml 10000 50
   ```

5. **Generate Full ML Dataset**:
   ```bash
   python prime_generator.py ml 50
   ```

## Commands

- `python prime_generator.py stats` - Show file statistics
- `python prime_generator.py ml [size]` - Generate ML-ready files ⭐**Recommended**
- `python prime_generator.py sample [size]` - Generate basic sample files
- `python prime_generator.py full` - Generate basic CSV files

## ML-Ready Output

**Enhanced for machine learning training:**

- `ml_primes_extended.csv` - 13 features + metadata (1,191,220 rows)
- `ml_primes_regular.csv` - 5 features + metadata (3,097,172 rows)

**Features:**

- ✅ Proper headers: `feature_00`, `feature_01`, etc.
- ✅ Metadata: `row_id`, `start_number`, `end_number`
- ✅ Statistics: `prime_count`, `prime_density`
- ✅ Ready for pandas/scikit-learn/TensorFlow

Values: `1` = prime, `0` = not prime

See the full setup guide for detailed instructions.
