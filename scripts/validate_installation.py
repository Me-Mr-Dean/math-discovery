#!/usr/bin/env python3
"""
Validation script to ensure the Mathematical Pattern Discovery Engine is properly installed
"""

import sys
import importlib

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "numpy", "pandas", "scikit-learn", "matplotlib", "yaml"
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} - MISSING")
    
    return missing

def check_package_structure():
    """Check if the package structure is correct"""
    try:
        import src
        print("✓ Main package structure")
        return True
    except ImportError:
        print("✗ Package structure - MISSING")
        return False

def main():
    print("Mathematical Pattern Discovery Engine - Installation Validation")
    print("=" * 60)
    
    print("\nChecking dependencies...")
    missing_deps = check_dependencies()
    
    print("\nChecking package structure...")
    structure_ok = check_package_structure()
    
    print("\n" + "=" * 60)
    if missing_deps:
        print(f"❌ Installation incomplete. Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -e .")
        sys.exit(1)
    elif not structure_ok:
        print("❌ Package structure incorrect. Please check installation.")
        sys.exit(1)
    else:
        print("✅ Installation validation successful!")
        print("The Mathematical Pattern Discovery Engine is ready to use.")

if __name__ == "__main__":
    main()
