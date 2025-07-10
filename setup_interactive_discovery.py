#!/usr/bin/env python3
"""
Simple Setup for Interactive Discovery Features
==============================================

A simpler setup script that just creates the necessary directories
and provides instructions without modifying existing files.

Usage:
    python simple_setup.py

Author: Mathematical Pattern Discovery Team
"""

from pathlib import Path
import sys


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")

    directories = [
        "scripts",
        "examples",
        "data/raw",
        "data/output",
        "tests/test_interactive",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")


def check_dependencies():
    """Check if required dependencies are available"""
    print("\n🔍 Checking dependencies...")

    required_modules = [
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("pathlib", "Path handling"),
        ("json", "JSON processing"),
    ]

    missing = []

    for module, description in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module} - {description}")
        except ImportError:
            print(f"   ❌ {module} - {description} (MISSING)")
            missing.append(module)

    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✅ All dependencies available!")
        return True


def validate_existing_structure():
    """Validate existing project structure"""
    print("\n🔍 Validating project structure...")

    essential_paths = [
        "src/core/discovery_engine.py",
        "src/generators/universal_generator.py",
        "src/utils/path_utils.py",
    ]

    missing_paths = []

    for path_str in essential_paths:
        path = Path(path_str)
        if path.exists():
            print(f"   ✅ {path_str}")
        else:
            missing_paths.append(path_str)
            print(f"   ❌ {path_str} - MISSING")

    if missing_paths:
        print(f"\n⚠️  Missing essential files: {missing_paths}")
        print("Please ensure the Universal Dataset Generator is properly set up first.")
        return False
    else:
        print("\n✅ Project structure looks good!")
        return True


def create_usage_guide():
    """Create a comprehensive usage guide"""
    print("\n📝 Creating usage guide...")

    guide_file = Path("INTERACTIVE_DISCOVERY_GUIDE.md")

    guide_content = """# Interactive Discovery Setup Guide

## 🚀 Quick Setup

### Step 1: Create the Script Files

Save the provided artifacts as these files:

1. **Interactive Discovery Engine** → `scripts/interactive_discovery.py`
2. **Discovery CLI** → `scripts/discover_patterns.py`  
3. **Enhanced Examples** → `examples/enhanced_discovery_examples.py`

### Step 2: Test the Setup

```bash
# Test basic functionality
python scripts/discover_patterns.py list

# Test interactive mode
python scripts/interactive_discovery.py
```

### Step 3: Generate Some Test Data (if none exists)

```bash
# Generate test datasets
python src/generators/universal_generator.py interactive

# Or create a quick demo dataset
python src/generators/universal_generator.py demo
```

### Step 4: Start Discovering Patterns!

```bash
# Full interactive mode (recommended)
python scripts/interactive_discovery.py

# Command-line mode
python scripts/discover_patterns.py interactive
```

## 🎯 Usage Examples

### Interactive Mode
```bash
python scripts/interactive_discovery.py
```

This will:
- Scan for all available datasets
- Show you what's available in a nice format
- Let you choose analysis options
- Guide you through the discovery process

### Command Line Mode
```bash
# List available datasets
python scripts/discover_patterns.py list

# Analyze specific dataset
python scripts/discover_patterns.py analyze data/output/rule_name/dataset.csv

# Analyze entire folder
python scripts/discover_patterns.py folder data/output/rule_name/

# Quick analysis
python scripts/discover_patterns.py analyze dataset.csv --quick

# Deep analysis with embeddings
python scripts/discover_patterns.py analyze dataset.csv --embedding fourier
```

## 📊 Dataset Types Supported

The interactive discovery engine automatically detects and handles:

- **Algebraic Features** - Comprehensive mathematical properties
- **Digit Tensors** - Digit-based patterns with optional embeddings  
- **Sequence Patterns** - Gap analysis and local density features
- **Prefix-Suffix Matrices** - Structural digit pattern analysis
- **Legacy ML Datasets** - Original format datasets

## 🔧 Analysis Modes

- **Quick** - Fast analysis with 10K samples
- **Standard** - Balanced approach with 50K samples  
- **Deep** - Thorough analysis with 100K samples + embeddings
- **Custom** - Full control over all parameters

## 📁 Output Files

Results are saved as:
- `interactive_discovery_results.json` - Full interactive session results
- `discovery_results_<dataset>.json` - Individual dataset analysis
- `matrix_analysis_<dataset>.json` - Matrix-specific analysis

## 🚨 Troubleshooting

### "No datasets found"
- Run the Universal Dataset Generator first: `python src/generators/universal_generator.py interactive`
- Check that datasets exist in `data/output/` or `data/raw/`

### Import errors
- Make sure you're running from the project root directory
- Install dependencies: `pip install -e .`

### Performance issues
- Use quick mode: `--quick`
- Reduce sample size: `--max-samples 10000`
- Analyze smaller datasets first

## 💡 Tips

1. **Start Small** - Generate a small test dataset first
2. **Use Interactive Mode** - It guides you through everything
3. **Quick Mode** - Use for initial exploration
4. **Save Results** - Always save for later comparison
5. **Batch Analysis** - Use folder mode for multiple datasets

## 🎉 Next Steps

Once you have the interactive discovery working:

1. Generate datasets for your mathematical rules
2. Use the discovery engine to find patterns
3. Compare different dataset formats
4. Export results for further analysis
5. Integrate with your research workflow

Happy discovering! 🧮✨
"""

    guide_file.write_text(guide_content, encoding="utf-8")
    print(f"   ✅ Created comprehensive guide: {guide_file}")


def show_file_creation_instructions():
    """Show detailed instructions for creating the files"""
    print("\n📋 FILE CREATION INSTRUCTIONS")
    print("=" * 50)
    print()
    print("Create these files from the provided artifacts:")
    print()

    files_to_create = [
        {
            "artifact": "Interactive Discovery Engine",
            "file": "scripts/interactive_discovery.py",
            "description": "Main interactive interface",
        },
        {
            "artifact": "Discovery Engine CLI",
            "file": "scripts/discover_patterns.py",
            "description": "Command-line interface",
        },
        {
            "artifact": "Enhanced Discovery Examples",
            "file": "examples/enhanced_discovery_examples.py",
            "description": "Usage examples and demos",
        },
    ]

    for i, file_info in enumerate(files_to_create, 1):
        print(f"{i}. **{file_info['artifact']}**")
        print(f"   → Save as: `{file_info['file']}`")
        print(f"   📝 {file_info['description']}")
        print()

    print("💡 After creating these files, test with:")
    print("   python scripts/discover_patterns.py list")
    print("   python scripts/interactive_discovery.py")


def run_quick_test():
    """Run a quick test of the existing infrastructure"""
    print("\n🧪 Testing existing infrastructure...")

    try:
        # Test basic Python functionality first
        import pandas as pd
        import numpy as np

        print("   ✅ Basic dependencies work")

        # Test path utilities without importing discovery engine
        sys.path.insert(0, str(Path("src")))

        try:
            from utils.path_utils import get_data_directory

            data_dir = get_data_directory()
            output_dir = data_dir.parent / "output"
            print(f"   ✅ Path utilities work: {data_dir}")
        except ImportError as e:
            print(f"   ⚠️  Path utilities import issue: {e}")
            print("   💡 This is expected if utils aren't set up yet")

        # Test if Universal Generator exists without importing discovery engine
        try:
            from generators.universal_generator import MathematicalRule

            # Simple test without using discovery engine
            rule = MathematicalRule(
                func=lambda n: n % 5 == 0,
                name="Test Rule",
                description="Test for setup",
            )

            # Test that the rule works
            test_result = rule.evaluate(10)  # Should be True
            assert test_result == True

            print("   ✅ Universal Generator basic functionality works")
            print("   ✅ Infrastructure test passed!")

            return True

        except ImportError as e:
            print(f"   ⚠️  Universal Generator import issue: {e}")
            print("   💡 Make sure Universal Generator is properly set up")
            return False

    except Exception as e:
        print(f"   ❌ Infrastructure test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🛠️  SIMPLE INTERACTIVE DISCOVERY SETUP")
    print("=" * 50)
    print("Setting up directories and validating your environment...")
    print()

    # Create directories
    create_directories()

    # Check dependencies
    deps_ok = check_dependencies()

    # Validate structure
    structure_ok = validate_existing_structure()

    # Create usage guide
    create_usage_guide()

    # Test infrastructure
    if deps_ok and structure_ok:
        test_ok = run_quick_test()
    else:
        test_ok = False

    # Show instructions
    show_file_creation_instructions()

    # Summary
    print("\n📊 SETUP SUMMARY")
    print("=" * 25)
    print(f"✅ Dependencies: {'OK' if deps_ok else 'ISSUES'}")
    print(f"✅ Project Structure: {'OK' if structure_ok else 'ISSUES'}")
    print(f"✅ Infrastructure Test: {'OK' if test_ok else 'ISSUES'}")

    if deps_ok and structure_ok and test_ok:
        print("\n🎉 Setup successful! Ready to create the interactive files.")
        print("\n🚀 Next steps:")
        print("1. Create the 3 script files shown above")
        print("2. Test with: python scripts/discover_patterns.py list")
        print("3. Start discovering: python scripts/interactive_discovery.py")
    else:
        print("\n⚠️  Setup had some issues. Please check the output above.")
        if not deps_ok:
            print("   • Install missing dependencies")
        if not structure_ok:
            print("   • Ensure Universal Dataset Generator is set up")
        if not test_ok:
            print("   • Check that basic functionality works")

    print(f"\n📚 See INTERACTIVE_DISCOVERY_GUIDE.md for detailed instructions")


if __name__ == "__main__":
    main()
