#!/usr/bin/env python3
"""
Setup Script for Universal Dataset Generator
===========================================

This script helps you set up the Universal Dataset Generator in your
Mathematical Pattern Discovery Engine project.

Usage:
    python setup_universal_generator.py

Author: Mathematical Pattern Discovery Team
"""

import shutil
from pathlib import Path
import os


def create_directory_structure():
    """Create the necessary directory structure"""
    print("📁 Creating directory structure...")

    base_dirs = ["data/raw", "data/output", "src/generators", "scripts", "examples"]

    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")


def update_generators_init():
    """Update the generators __init__.py file"""
    print("\n📝 Updating src/generators/__init__.py...")

    init_file = Path("src/generators/__init__.py")

    if init_file.exists():
        content = init_file.read_text()
        if "universal_generator" not in content:
            # Add universal_generator to __all__
            if "__all__" in content:
                content = content.replace(
                    '"sequence_generator",',
                    '"sequence_generator",\n    "universal_generator",',
                )
            else:
                # Add __all__ if it doesn't exist
                content += '\n\n__all__ = [\n    "prime_generator",\n    "sequence_generator",\n    "universal_generator",\n]\n'

            init_file.write_text(content)
            print("   ✅ Updated __init__.py")
        else:
            print("   ✅ Already up to date")
    else:
        # Create new __init__.py
        content = '''"""Generator subpackage with universal dataset generation."""

__all__ = [
    "prime_generator",
    "sequence_generator", 
    "universal_generator",
]
'''
        init_file.write_text(content)
        print("   ✅ Created __init__.py")


def create_gitignore_additions():
    """Add dataset-related entries to .gitignore"""
    print("\n📝 Updating .gitignore...")

    gitignore_file = Path(".gitignore")

    additions = [
        "\n# Universal Dataset Generator outputs",
        "data/output/*/",
        "data/raw/*/numbers_*.csv",
        "data/raw/*/metadata_*.json",
    ]

    if gitignore_file.exists():
        content = gitignore_file.read_text()

        # Check if additions already exist
        if "Universal Dataset Generator" not in content:
            content += "\n" + "\n".join(additions)
            gitignore_file.write_text(content)
            print("   ✅ Added dataset entries to .gitignore")
        else:
            print("   ✅ .gitignore already updated")
    else:
        gitignore_file.write_text("\n".join(additions))
        print("   ✅ Created .gitignore with dataset entries")


def check_dependencies():
    """Check if required dependencies are available"""
    print("\n🔍 Checking dependencies...")

    required_modules = [
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("pathlib", "Path handling"),
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


def create_readme_section():
    """Create a README section for the Universal Dataset Generator"""
    print("\n📝 Creating README section...")

    readme_section = """
## 🧮 Universal Dataset Generator

The Universal Dataset Generator allows you to create ML-ready datasets from any mathematical rule or function.

### Quick Start

```bash
# List available mathematical rules
python scripts/generate_universal_datasets.py list

# Generate datasets for perfect squares
python scripts/generate_universal_datasets.py generate 1 --max 10000

# Interactive rule creation
python scripts/generate_universal_datasets.py interactive

# Quick demo
python examples/universal_dataset_examples.py demo
```

### Features

- **Multiple ML Representations**: Prefix-suffix matrices, digit tensors, sequence patterns, algebraic features
- **Organized Storage**: Automatic folder organization in `data/raw/` and `data/output/`
- **Pure Mathematical Discovery**: No hard-coded knowledge, learns patterns from structure
- **Scalable**: Handle up to millions of numbers
- **Extensible**: Easy to add custom processors

### Generated Dataset Types

1. **Prefix-Suffix Matrices** - Structural digit patterns
2. **Digit Tensors** - Full digit decomposition with embeddings
3. **Sequence Patterns** - Gap analysis and local density features
4. **Algebraic Features** - Comprehensive number-theoretic properties

### Usage Examples

```python
from src.generators.universal_generator import UniversalDatasetGenerator, MathematicalRule

# Define your rule
rule = MathematicalRule(
    func=lambda n: n % 2 == 0,
    name="Even Numbers",
    description="Numbers divisible by 2"
)

# Generate all dataset types
generator = UniversalDatasetGenerator()
summary = generator.generate_complete_pipeline(rule, max_number=10000)
```

### Integration with Discovery Engine

Generated datasets work seamlessly with the existing discovery engine:

```python
# Use generated datasets with discovery engine
dataset_path = "data/output/even_numbers/algebraic_features_up_to_10000.csv"
# Load and analyze with your existing tools
```
"""

    readme_file = Path("README_universal_generator.md")
    readme_file.write_text(readme_section.strip())
    print(f"   ✅ Created {readme_file}")


def run_validation_test():
    """Run a quick validation test"""
    print("\n🧪 Running validation test...")

    try:
        # Try importing the universal generator (this will fail if not properly placed)
        import sys

        sys.path.insert(0, str(Path("src")))

        from generators.universal_generator import (
            UniversalDatasetGenerator,
            MathematicalRule,
        )

        # Create a simple test rule
        rule = MathematicalRule(
            func=lambda n: n % 5 == 0,
            name="Test Multiples of 5",
            description="Test rule for validation",
        )

        generator = UniversalDatasetGenerator()

        # Generate very small dataset for testing
        summary = generator.generate_complete_pipeline(
            rule=rule,
            max_number=50,  # Very small for quick test
            processors=["algebraic_features"],
        )

        print(f"   ✅ Test successful!")
        print(f"   📊 Found {len(summary['raw_numbers'])} numbers")
        print(f"   💾 Generated {summary['total_datasets']} datasets")

        # Clean up test files
        test_dir = Path("data/output/test_multiples_of_5")
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("   🧹 Cleaned up test files")

        return True

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def show_next_steps():
    """Show what to do next"""
    print("\n🚀 SETUP COMPLETE!")
    print("=" * 50)
    print()
    print("📁 File locations:")
    print("   Main generator: src/generators/universal_generator.py")
    print("   CLI interface:  scripts/generate_universal_datasets.py")
    print("   Examples:       examples/universal_dataset_examples.py")
    print()
    print("🎯 Next steps:")
    print(
        "   1. Copy the universal generator code to src/generators/universal_generator.py"
    )
    print("   2. Copy the CLI wrapper to scripts/generate_universal_datasets.py")
    print("   3. Copy the examples to examples/universal_dataset_examples.py")
    print("   4. Run: python examples/universal_dataset_examples.py demo")
    print()
    print("⚡ Quick commands to try:")
    print("   python scripts/generate_universal_datasets.py list")
    print("   python scripts/generate_universal_datasets.py demo")
    print("   python examples/universal_dataset_examples.py demo")
    print()
    print("📚 For full documentation, see: README_universal_generator.md")


def main():
    """Main setup function"""
    print("🛠️  UNIVERSAL DATASET GENERATOR - SETUP")
    print("=" * 50)
    print("Setting up the Universal Dataset Generator in your project...")
    print()

    # Run setup steps
    create_directory_structure()
    update_generators_init()
    create_gitignore_additions()

    # Check dependencies
    deps_ok = check_dependencies()

    create_readme_section()

    # Only run validation if dependencies are OK
    if deps_ok:
        test_ok = run_validation_test()
        if not test_ok:
            print(
                "\n⚠️  Validation test failed. You may need to copy the code files first."
            )

    show_next_steps()

    print("\n✨ Setup script complete!")


if __name__ == "__main__":
    main()
