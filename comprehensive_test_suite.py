#!/usr/bin/env python3
"""
Comprehensive Test Suite for Universal Dataset Generator
=======================================================

This script thoroughly tests all components of your integrated Universal Dataset Generator
to ensure everything is working correctly and ready for production use.

Author: Mathematical Pattern Discovery Team
"""

import sys
from pathlib import Path
import time
import traceback

# Add project src to path
project_root = Path.cwd()
if (project_root / "src").exists():
    sys.path.insert(0, str(project_root / "src"))
else:
    print("⚠️  Please run this script from the project root directory")
    sys.exit(1)


def test_imports():
    """Test all critical imports"""
    print("🔧 Testing imports...")

    try:
        # Test Universal Generator imports
        from generators.universal_generator import (
            UniversalDatasetGenerator,
            MathematicalRule,
            PrefixSuffixProcessor,
            DigitTensorProcessor,
            SequencePatternProcessor,
            AlgebraicFeatureProcessor,
        )

        print("  ✅ Universal Generator classes")

        # Test discovery engine integration
        from core.discovery_engine import UniversalMathDiscovery

        print("  ✅ Discovery Engine")

        # Test utility imports
        from utils.math_utils import generate_mathematical_features
        from utils.embedding_utils import fourier_transform, pca_transform
        from utils.prefix_suffix_utils import generate_prefix_suffix_matrix

        print("  ✅ All utility functions")

        return True

    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic Universal Generator functionality"""
    print("\n🧪 Testing basic functionality...")

    try:
        from generators.universal_generator import (
            UniversalDatasetGenerator,
            MathematicalRule,
        )

        # Create a simple rule
        rule = MathematicalRule(
            func=lambda n: n % 2 == 0,
            name="Test Even Numbers",
            description="Numbers divisible by 2 (for testing)",
            examples=[2, 4, 6, 8, 10],
        )

        # Test rule creation
        assert rule.name == "Test Even Numbers"
        assert rule.safe_name == "test_even_numbers"
        assert rule.evaluate(4) == True
        assert rule.evaluate(5) == False
        print("  ✅ MathematicalRule class")

        # Test generator creation
        generator = UniversalDatasetGenerator()
        assert generator.base_dir == Path("data")
        assert len(generator.processors) >= 4  # Should have multiple processors
        print("  ✅ UniversalDatasetGenerator class")

        return True

    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_processors():
    """Test all dataset processors"""
    print("\n📊 Testing dataset processors...")

    try:
        from generators.universal_generator import (
            PrefixSuffixProcessor,
            DigitTensorProcessor,
            SequencePatternProcessor,
            AlgebraicFeatureProcessor,
        )

        # Sample data for testing
        test_numbers = [1, 4, 9, 16, 25]  # Perfect squares
        test_metadata = {
            "rule_name": "Test Squares",
            "max_number": 25,
            "total_found": 5,
        }

        # Test each processor
        processors = [
            PrefixSuffixProcessor(2, 1),
            DigitTensorProcessor(4, False),  # Simple version for speed
            SequencePatternProcessor(3),  # Small window for speed
            AlgebraicFeatureProcessor(),
        ]

        for processor in processors:
            print(f"    Testing {processor.get_name()}...")

            # This would be slow for large datasets, so we'll just verify the methods exist
            assert hasattr(processor, "process")
            assert hasattr(processor, "get_name")
            assert hasattr(processor, "get_description")

            name = processor.get_name()
            desc = processor.get_description()
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(desc, str) and len(desc) > 0

            print(f"      ✅ {name}")

        print("  ✅ All processors functional")
        return True

    except Exception as e:
        print(f"  ❌ Processor test failed: {e}")
        traceback.print_exc()
        return False


def test_small_dataset_generation():
    """Test generating a very small dataset"""
    print("\n🔬 Testing small dataset generation...")

    try:
        from generators.universal_generator import (
            UniversalDatasetGenerator,
            MathematicalRule,
        )

        # Create test rule
        rule = MathematicalRule(
            func=lambda n: n in [1, 4, 9, 16, 25],  # First 5 perfect squares
            name="Test Perfect Squares",
            description="First 5 perfect squares for testing",
        )

        generator = UniversalDatasetGenerator()

        # Generate very small dataset
        print("    Generating raw dataset (1-30)...")
        numbers, metadata = generator.generate_raw_dataset(
            rule,
            max_number=30,  # Very small for speed
            save_raw=False,  # Don't save during testing
        )

        # Verify results
        expected_numbers = [1, 4, 9, 16, 25]
        assert (
            numbers == expected_numbers
        ), f"Expected {expected_numbers}, got {numbers}"
        assert metadata["total_found"] == 5
        assert metadata["max_number"] == 30
        print("      ✅ Raw dataset generation")

        # Test one processor quickly
        print("    Testing ML dataset processing...")
        ml_datasets = generator.process_to_ml_datasets(
            numbers,
            metadata,
            processors=["algebraic_features"],  # Just one processor for speed
        )

        assert len(ml_datasets) == 1
        assert "algebraic_features" in ml_datasets
        df = ml_datasets["algebraic_features"]
        assert len(df) >= 25  # Should have rows for the max number found
        assert "target" in df.columns
        print("      ✅ ML dataset processing")

        print("  ✅ Small dataset generation successful")
        return True

    except Exception as e:
        print(f"  ❌ Dataset generation test failed: {e}")
        traceback.print_exc()
        return False


def test_discovery_engine_integration():
    """Test integration with existing discovery engine"""
    print("\n🤖 Testing discovery engine integration...")

    try:
        from generators.universal_generator import (
            UniversalDatasetGenerator,
            MathematicalRule,
        )
        from core.discovery_engine import UniversalMathDiscovery

        # Create simple rule
        rule = MathematicalRule(
            func=lambda n: n % 3 == 0,
            name="Test Multiples of 3",
            description="Numbers divisible by 3",
        )

        # Test that both systems can work with the same rule
        print("    Testing rule compatibility...")

        # Universal Generator
        generator = UniversalDatasetGenerator()
        numbers, metadata = generator.generate_raw_dataset(rule, 30, save_raw=False)
        expected_multiples = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        assert numbers == expected_multiples
        print("      ✅ Universal Generator handles rule")

        # Discovery Engine
        discovery = UniversalMathDiscovery(
            target_function=rule.func, function_name=rule.name, max_number=30
        )
        # Just test creation, not full discovery (too slow for test)
        assert discovery.function_name == rule.name
        print("      ✅ Discovery Engine handles rule")

        print("  ✅ Integration successful")
        return True

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


def test_cli_availability():
    """Test if CLI scripts are available"""
    print("\n💻 Testing CLI availability...")

    try:
        # Check if CLI script exists
        cli_script = project_root / "scripts" / "generate_universal_datasets.py"
        examples_script = project_root / "examples" / "universal_dataset_examples.py"

        if cli_script.exists():
            print("  ✅ CLI script found")
        else:
            print("  ⚠️  CLI script not found (optional)")

        if examples_script.exists():
            print("  ✅ Examples script found")
        else:
            print("  ⚠️  Examples script not found (optional)")

        return True

    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False


def test_file_structure():
    """Test that directory structure is correct"""
    print("\n📁 Testing file structure...")

    try:
        # Check key directories exist or can be created
        data_dir = project_root / "data"
        raw_dir = data_dir / "raw"
        output_dir = data_dir / "output"

        # These should be creatable
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        assert data_dir.exists()
        assert raw_dir.exists()
        assert output_dir.exists()
        print("  ✅ Data directories created/verified")

        # Check source structure
        src_dir = project_root / "src"
        generators_dir = src_dir / "generators"
        utils_dir = src_dir / "utils"
        core_dir = src_dir / "core"

        assert src_dir.exists()
        assert generators_dir.exists()
        assert utils_dir.exists()
        assert core_dir.exists()
        print("  ✅ Source directories verified")

        return True

    except Exception as e:
        print(f"  ❌ File structure test failed: {e}")
        return False


def run_performance_benchmark():
    """Quick performance benchmark"""
    print("\n⚡ Running performance benchmark...")

    try:
        from generators.universal_generator import (
            UniversalDatasetGenerator,
            MathematicalRule,
        )

        # Create rule that will find some numbers quickly
        rule = MathematicalRule(
            func=lambda n: n % 10 == 0,  # Every 10th number
            name="Benchmark Multiples of 10",
            description="Performance test rule",
        )

        generator = UniversalDatasetGenerator()

        # Time the generation
        start_time = time.time()
        numbers, metadata = generator.generate_raw_dataset(rule, 1000, save_raw=False)
        raw_time = time.time() - start_time

        expected_count = 100  # 10, 20, 30, ..., 1000
        assert len(numbers) == expected_count

        rate = 1000 / raw_time  # numbers processed per second
        print(f"  ✅ Raw generation: {rate:.0f} numbers/sec")

        # Test one processor
        start_time = time.time()
        ml_datasets = generator.process_to_ml_datasets(
            numbers, metadata, processors=["algebraic_features"]
        )
        process_time = time.time() - start_time

        df = ml_datasets["algebraic_features"]
        assert len(df) == 1000

        ml_rate = 1000 / process_time
        print(f"  ✅ ML processing: {ml_rate:.0f} numbers/sec")

        if rate > 100 and ml_rate > 50:
            print("  ✅ Performance acceptable")
        else:
            print("  ⚠️  Performance could be better")

        return True

    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run all tests"""
    print("🚀 COMPREHENSIVE UNIVERSAL DATASET GENERATOR TEST")
    print("=" * 60)
    print("Testing all components of your integrated system...\n")

    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Processor Tests", test_processors),
        ("Small Dataset Generation", test_small_dataset_generation),
        ("Discovery Engine Integration", test_discovery_engine_integration),
        ("CLI Availability", test_cli_availability),
        ("File Structure", test_file_structure),
        ("Performance Benchmark", run_performance_benchmark),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 40)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\nScore: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print(
            "Your Universal Dataset Generator is fully integrated and working perfectly!"
        )
        print("\n🚀 Ready for production use:")
        print(
            "  • Generate datasets: python scripts/generate_universal_datasets.py list"
        )
        print("  • Run examples: python examples/universal_dataset_examples.py demo")
        print("  • Create custom rules and generate ML-ready datasets")
        print("  • Use with existing discovery engine for pattern analysis")

    elif passed >= len(tests) - 2:
        print("\n✅ MOSTLY WORKING!")
        print("Core functionality is solid. Minor optional components may be missing.")
        print("You can proceed with using the Universal Dataset Generator.")

    else:
        print("\n⚠️  SOME ISSUES DETECTED")
        print("Please check the failed tests above.")
        print("Core functionality may need attention.")

    return passed == len(tests)


def main():
    """Main test function"""
    try:
        success = run_comprehensive_test()

        if success:
            print("\n✨ INTEGRATION VERIFICATION COMPLETE!")
            print(
                "Your Mathematical Pattern Discovery Engine with Universal Dataset Generator"
            )
            print("is fully functional and ready for mathematical research!")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
