#!/usr/bin/env python3
"""
Validation script for Mathematical Pattern Discovery Engine setup
Checks dependencies, data files, and runs basic functionality tests
"""

import sys
import importlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def check_python_version():
    """Check if Python version is supported"""
    print("🐍 Checking Python version...")

    version = sys.version_info
    if version >= (3, 8):
        print(
            f"  ✅ Python {version.major}.{version.minor}.{version.micro} - Supported"
        )
        return True
    else:
        print(
            f"  ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+"
        )
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking dependencies...")

    required_packages = {
        "pandas": "Data manipulation library",
        "numpy": "Numerical computing library",
        "sklearn": "Machine learning library",
        "matplotlib": "Plotting library",
        "yaml": "YAML configuration files",
    }

    optional_packages = {
        "seaborn": "Statistical visualization",
        "plotly": "Interactive plots",
        "jupyter": "Notebook environment",
    }

    missing_required = []
    missing_optional = []

    # Check required packages
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"  ✅ {package:<12} - {description}")
        except ImportError:
            missing_required.append(package)
            print(f"  ❌ {package:<12} - MISSING ({description})")

    # Check optional packages
    for package, description in optional_packages.items():
        try:
            importlib.import_module(package)
            print(f"  ⭐ {package:<12} - {description}")
        except ImportError:
            missing_optional.append(package)
            print(f"  ⚪ {package:<12} - Optional ({description})")

    return missing_required, missing_optional


def check_package_structure():
    """Check if the package structure is correct"""
    print("\n📁 Checking package structure...")

    essential_paths = [
        "src/",
        "src/core/",
        "src/core/discovery_engine.py",
        "src/analyzers/",
        "src/generators/",
        "src/utils/",
        "setup.py",
        "requirements.txt",
    ]

    missing_paths = []

    for path_str in essential_paths:
        path = project_root / path_str
        if path.exists():
            print(f"  ✅ {path_str}")
        else:
            missing_paths.append(path_str)
            print(f"  ❌ {path_str} - MISSING")

    return missing_paths


def check_data_files():
    """Check if required data files exist"""
    print("\n📊 Checking data files...")

    try:
        from utils.path_utils import find_data_file, get_data_directory

        required_files = [
            "1m.csv",
            "ml_dataset1_odd_endings.csv",
            "ml_dataset1_odd_endings_sample.csv",
        ]

        missing_files = []
        found_files = []

        for filename in required_files:
            path = find_data_file(filename)
            if path:
                print(f"  ✅ {filename} -> {path}")
                found_files.append(filename)
            else:
                missing_files.append(filename)
                print(f"  ❌ {filename} - MISSING")

        data_dir = get_data_directory()
        print(f"\n  📁 Data directory: {data_dir}")

        return missing_files, found_files

    except ImportError as e:
        print(f"  ❌ Cannot import path utilities: {e}")
        return ["Unable to check"], []


def test_basic_functionality():
    """Test basic functionality of core modules"""
    print("\n🧪 Testing basic functionality...")

    try:
        # Test math utilities
        from utils.math_utils import (
            is_prime,
            euler_totient,
            generate_mathematical_features,
        )

        # Test basic functions
        assert is_prime(17) == True
        assert is_prime(15) == False
        assert euler_totient(6) == 2

        features = generate_mathematical_features(17)
        assert features["number"] == 17
        assert features["is_prime"] == True

        print("  ✅ Math utilities work correctly")

        # Test discovery engine import
        from core.discovery_engine import UniversalMathDiscovery

        print("  ✅ Discovery engine imports successfully")

        # Test a very simple discovery
        def simple_even(n):
            return n % 2 == 0

        engine = UniversalMathDiscovery(
            target_function=simple_even,
            function_name="Even Numbers",
            max_number=20,  # Very small for quick test
        )
        print("  ✅ Discovery engine can be instantiated")

        return True

    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False


def provide_recommendations(missing_req, missing_opt, missing_paths, missing_files):
    """Provide specific recommendations based on validation results"""
    print("\n💡 RECOMMENDATIONS:")
    print("=" * 50)

    if missing_req:
        print("🔥 CRITICAL - Install missing dependencies:")
        print(f"  pip install {' '.join(missing_req)}")
        print()

    if missing_paths:
        print("🔥 CRITICAL - Package structure issues:")
        print("  Make sure you're running from the project root directory")
        print("  Check if the repository was cloned correctly")
        print()

    if missing_files:
        print("⚠️  IMPORTANT - Generate missing data files:")
        print("  python scripts/generate_sample_data.py")
        print()

    if missing_opt:
        print("⭐ OPTIONAL - Install additional features:")
        print(f"  pip install {' '.join(missing_opt)}")
        print()

    if not any([missing_req, missing_paths, missing_files]):
        print("🎉 READY TO GO!")
        print("  python examples/basic_prime_discovery.py")
        print("  python -m pytest tests/")
        print()


def main():
    """Main validation function"""
    print("🔍 Mathematical Pattern Discovery Engine - Setup Validation")
    print("=" * 70)
    print("This script checks if your environment is properly configured.\n")

    # Run all validation checks
    python_ok = check_python_version()
    missing_req, missing_opt = check_dependencies()
    missing_paths = check_package_structure()
    missing_files, found_files = check_data_files()
    functionality_ok = test_basic_functionality()

    # Summary
    print("\n📋 VALIDATION SUMMARY:")
    print("=" * 50)

    if python_ok:
        print("✅ Python version: Compatible")
    else:
        print("❌ Python version: Incompatible")

    if not missing_req:
        print("✅ Required dependencies: All installed")
    else:
        print(f"❌ Required dependencies: {len(missing_req)} missing")

    if not missing_paths:
        print("✅ Package structure: Complete")
    else:
        print(f"❌ Package structure: {len(missing_paths)} issues")

    if not missing_files:
        print("✅ Data files: Available")
    else:
        print(f"⚠️  Data files: {len(missing_files)} missing")

    if functionality_ok:
        print("✅ Basic functionality: Working")
    else:
        print("❌ Basic functionality: Issues detected")

    # Overall status
    critical_issues = (
        not python_ok or missing_req or missing_paths or not functionality_ok
    )

    if critical_issues:
        print("\n🚨 CRITICAL ISSUES DETECTED")
        print("The system is not ready for use.")
    elif missing_files:
        print("\n⚠️  MINOR ISSUES DETECTED")
        print("Core functionality works, but examples may fail.")
    else:
        print("\n🎉 ALL SYSTEMS GO!")
        print("Your environment is properly configured.")

    # Provide specific recommendations
    provide_recommendations(missing_req, missing_opt, missing_paths, missing_files)

    # Exit with appropriate code
    sys.exit(1 if critical_issues else 0)


if __name__ == "__main__":
    main()
