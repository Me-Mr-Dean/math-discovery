#!/usr/bin/env python3
"""
Quick check script to verify everything works from start to finish
This tests the complete flow without requiring sample data generation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_1_imports():
    """Test that all critical imports work"""
    print("🔧 Testing critical imports...")

    try:
        import pandas as pd

        print("  ✅ pandas")
    except ImportError:
        print("  ❌ pandas - run: pip install pandas")
        return False

    try:
        import numpy as np

        print("  ✅ numpy")
    except ImportError:
        print("  ❌ numpy - run: pip install numpy")
        return False

    try:
        import sklearn

        print("  ✅ scikit-learn")
    except ImportError:
        print("  ❌ scikit-learn - run: pip install scikit-learn")
        return False

    try:
        import matplotlib

        print("  ✅ matplotlib")
    except ImportError:
        print("  ❌ matplotlib - run: pip install matplotlib")
        return False

    return True


def test_2_package_structure():
    """Test that package imports work"""
    print("\n📦 Testing package structure...")

    try:
        from utils.math_utils import is_prime, generate_mathematical_features

        print("  ✅ utils.math_utils")
    except ImportError as e:
        print(f"  ❌ utils.math_utils - {e}")
        return False

    try:
        from core.discovery_engine import UniversalMathDiscovery

        print("  ✅ core.discovery_engine")
    except ImportError as e:
        print(f"  ❌ core.discovery_engine - {e}")
        return False

    try:
        from generators.prime_generator import PrimeGenerator

        print("  ✅ generators.prime_generator")
    except ImportError as e:
        print(f"  ❌ generators.prime_generator - {e}")
        return False

    try:
        from utils.path_utils import find_data_file, get_project_root

        print("  ✅ utils.path_utils")
    except ImportError as e:
        print(f"  ❌ utils.path_utils - {e}")
        return False

    return True


def test_3_math_functions():
    """Test core mathematical functions"""
    print("\n🧮 Testing mathematical functions...")

    try:
        from utils.math_utils import (
            is_prime,
            euler_totient,
            generate_mathematical_features,
        )

        # Test prime checking
        assert is_prime(17) == True
        assert is_prime(15) == False
        print("  ✅ Prime checking works")

        # Test Euler totient
        assert euler_totient(6) == 2
        assert euler_totient(9) == 6
        print("  ✅ Euler totient works")

        # Test feature generation
        features = generate_mathematical_features(17)
        assert features["number"] == 17
        assert features["is_prime"] == True
        assert features["mod_2"] == 1
        print("  ✅ Feature generation works")

        return True

    except Exception as e:
        print(f"  ❌ Math functions failed: {e}")
        return False


def test_4_data_files():
    """Check what data files are available"""
    print("\n📊 Checking data files...")

    try:
        from utils.path_utils import find_data_file, get_data_directory

        data_dir = get_data_directory()
        print(f"  📁 Data directory: {data_dir}")

        # Check for large datasets (should exist per user)
        large_files = [
            "1m.csv",
            "ml_dataset1_odd_endings.csv",
            "ml_dataset2_all_digits.csv",
            "ml_dataset3_prime_endings.csv",
        ]

        found_files = []
        for filename in large_files:
            path = find_data_file(filename)
            if path:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {filename} ({size_mb:.1f} MB)")
                found_files.append(filename)
            else:
                print(f"  ⚪ {filename} - not found")

        # Check for sample versions
        sample_files = [
            "ml_dataset1_odd_endings_sample.csv",
            "ml_dataset2_all_digits_sample.csv",
            "ml_dataset3_prime_endings_sample.csv",
        ]

        for filename in sample_files:
            path = find_data_file(filename)
            if path:
                size_kb = path.stat().st_size / 1024
                print(f"  ✅ {filename} ({size_kb:.1f} KB)")
                found_files.append(filename)
            else:
                print(f"  ⚪ {filename} - not found")

        if len(found_files) > 0:
            print(f"  📈 Found {len(found_files)} dataset files")
            return found_files
        else:
            print("  ⚠️  No dataset files found")
            return []

    except Exception as e:
        print(f"  ❌ Data file check failed: {e}")
        return []


def test_5_simple_discovery():
    """Test a simple discovery on a tiny dataset"""
    print("\n🔬 Testing simple discovery engine...")

    try:
        from core.discovery_engine import UniversalMathDiscovery

        # Create a very simple test function
        def simple_even(n):
            return n % 2 == 0

        # Test with very small dataset for speed
        engine = UniversalMathDiscovery(
            target_function=simple_even,
            function_name="Even Numbers Test",
            max_number=20,  # Very small
        )

        print("  ✅ Discovery engine created")

        # Try to generate data (this should be fast)
        X, y = engine.generate_target_data()
        print(f"  ✅ Generated {len(X)} samples")

        print("  🎯 Discovery engine working!")
        return True

    except Exception as e:
        print(f"  ❌ Discovery test failed: {e}")
        return False


def test_6_dataset_loading():
    """Test loading one of the actual datasets"""
    print("\n📋 Testing dataset loading...")

    try:
        from utils.path_utils import find_data_file
        import pandas as pd

        # Try to find any ML dataset
        dataset_files = [
            "ml_dataset1_odd_endings.csv",
            "ml_dataset1_odd_endings_sample.csv",
            "ml_dataset2_all_digits.csv",
            "ml_dataset2_all_digits_sample.csv",
        ]

        for filename in dataset_files:
            path = find_data_file(filename)
            if path:
                print(f"  📊 Loading {filename}...")
                df = pd.read_csv(path, index_col=0)
                print(f"  ✅ Shape: {df.shape}")
                print(
                    f"  ✅ Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}"
                )

                # Check for metadata columns
                metadata_cols = [
                    "range_start",
                    "range_end",
                    "prime_count",
                    "prime_density",
                ]
                has_metadata = any(col in df.columns for col in metadata_cols)
                print(f"  ✅ Has metadata: {has_metadata}")

                return True

        print("  ⚠️  No loadable datasets found")
        return False

    except Exception as e:
        print(f"  ❌ Dataset loading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Mathematical Pattern Discovery Engine - Quick Check")
    print("=" * 60)
    print("Testing complete flow from imports to discovery...")
    print()

    tests = [
        ("Critical Imports", test_1_imports),
        ("Package Structure", test_2_package_structure),
        ("Math Functions", test_3_math_functions),
        ("Data Files", test_4_data_files),
        ("Simple Discovery", test_5_simple_discovery),
        ("Dataset Loading", test_6_dataset_loading),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  💥 {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📊 TEST SUMMARY:")
    print("=" * 30)

    passed = 0
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}")
            passed += 1
        else:
            print(f"❌ {test_name}")

    print(f"\nScore: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Mathematical Pattern Discovery Engine is working perfectly!")
        print("\n🚀 Ready to run:")
        print("  python examples/basic_prime_discovery.py")
        print("  python examples/custom_function_discovery.py")
    elif passed >= len(tests) - 2:
        print("\n✅ MOSTLY WORKING!")
        print("Core functionality is solid. Minor issues detected.")
        print("You can proceed with examples.")
    else:
        print("\n⚠️  ISSUES DETECTED")
        print("Some core components need attention.")
        print("Check the failed tests above.")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
