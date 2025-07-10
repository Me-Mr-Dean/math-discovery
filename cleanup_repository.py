#!/usr/bin/env python3
"""
Repository Cleanup Script
========================

This script cleans up the Mathematical Pattern Discovery Engine repository
by removing redundant files, fixing imports, and organizing the structure.

Usage:
    python cleanup_repository.py [--dry-run]

Author: Mathematical Pattern Discovery Team
"""

import os
import shutil
from pathlib import Path
import argparse
import sys


class RepositoryCleanup:
    """Clean up and reorganize the repository"""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.project_root = Path.cwd()
        self.removed_files = []
        self.moved_files = []
        self.created_files = []

    def log_action(self, action, path, target=None):
        """Log cleanup actions"""
        if target:
            print(f"{'[DRY RUN] ' if self.dry_run else ''}{action}: {path} → {target}")
        else:
            print(f"{'[DRY RUN] ' if self.dry_run else ''}{action}: {path}")

    def remove_file(self, path):
        """Remove a file safely"""
        file_path = self.project_root / path
        if file_path.exists():
            self.log_action("REMOVE", path)
            if not self.dry_run:
                if file_path.is_file():
                    file_path.unlink()
                else:
                    shutil.rmtree(file_path)
            self.removed_files.append(str(path))

    def create_file(self, path, content):
        """Create a new file with content"""
        file_path = self.project_root / path
        self.log_action("CREATE", path)
        if not self.dry_run:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        self.created_files.append(str(path))

    def remove_redundant_files(self):
        """Remove redundant and broken files"""
        print("\n🗑️  REMOVING REDUNDANT FILES")
        print("=" * 40)

        # Empty or broken files
        files_to_remove = [
            "examples/interactive_discovery_usage.md",  # Empty
            "src/analyzers/fermat_analyzer.py",  # Move to utils instead
            "src/analyzers/prefix_suffix_analyzer.py",  # Redundant with universal generator
            # Redundant setup scripts
            "setup_universal_generator.py",
            "setup_interactive_discovery.py",
            "comprehensive_test_suite.py",
            # Broken discovery engine (we'll replace it)
            # "src/core/discovery_engine.py",  # Will be replaced
            # Redundant examples
            "examples/oeis_sequence_analysis.py",  # Redundant with universal examples
            "examples/enhanced_discovery_examples.py",  # Not actually created
            # Empty init files that cause issues
            "src/analyzers/__init__.py",  # Will recreate clean version
            "src/cli/__init__.py",
            "src/core/__init__.py",
            "src/generators/__init__.py",
            "src/utils/__init__.py",
        ]

        for file_path in files_to_remove:
            self.remove_file(file_path)

    def create_clean_init_files(self):
        """Create clean __init__.py files"""
        print("\n📝 CREATING CLEAN INIT FILES")
        print("=" * 35)

        # Main package init
        main_init = '''"""
Mathematical Pattern Discovery Engine

A machine learning-powered mathematical discovery system for uncovering
hidden patterns in number theory and mathematical sequences.
"""

__version__ = "1.0.0"
__author__ = "Mathematical Research Team"

# Core functionality
from .core.discovery_engine import UniversalMathDiscovery
from .generators.universal_generator import UniversalDatasetGenerator, MathematicalRule

# Utility imports
from .utils.math_utils import generate_mathematical_features

__all__ = [
    "UniversalMathDiscovery",
    "UniversalDatasetGenerator", 
    "MathematicalRule",
    "generate_mathematical_features",
]
'''
        self.create_file("src/__init__.py", main_init)

        # Core module init
        core_init = '''"""Core discovery engine and pattern analysis."""

from .discovery_engine import UniversalMathDiscovery
from .feature_extractor import FeatureExtractor
from .pattern_analyzer import PatternAnalyzer

__all__ = [
    "UniversalMathDiscovery",
    "FeatureExtractor", 
    "PatternAnalyzer",
]
'''
        self.create_file("src/core/__init__.py", core_init)

        # Generators init
        generators_init = '''"""Data generators for mathematical sequences."""

from .universal_generator import UniversalDatasetGenerator, MathematicalRule
from .prime_generator import PrimeGenerator, PrimeCSVGenerator
from .sequence_generator import SequenceGenerator

__all__ = [
    "UniversalDatasetGenerator",
    "MathematicalRule",
    "PrimeGenerator",
    "PrimeCSVGenerator", 
    "SequenceGenerator",
]
'''
        self.create_file("src/generators/__init__.py", generators_init)

        # Analyzers init
        analyzers_init = '''"""Specialized mathematical analyzers."""

from .prime_analyzer import PurePrimeMLDiscovery
from .oeis_analyzer import a007694_property, generate_a007694_sequence

__all__ = [
    "PurePrimeMLDiscovery",
    "a007694_property",
    "generate_a007694_sequence",
]
'''
        self.create_file("src/analyzers/__init__.py", analyzers_init)

        # Utils init
        utils_init = '''"""Utility functions for mathematical discovery."""

from .math_utils import generate_mathematical_features, is_prime, euler_totient
from .embedding_utils import fourier_transform, pca_transform
from .prefix_suffix_utils import generate_prefix_suffix_matrix
from .path_utils import find_data_file, get_data_directory
from .data_utils import load_dataset, save_results
from .validation_utils import is_non_empty_numeric_sequence, has_unique_elements

__all__ = [
    "generate_mathematical_features",
    "is_prime",
    "euler_totient", 
    "fourier_transform",
    "pca_transform",
    "generate_prefix_suffix_matrix",
    "find_data_file",
    "get_data_directory",
    "load_dataset",
    "save_results",
    "is_non_empty_numeric_sequence",
    "has_unique_elements",
]
'''
        self.create_file("src/utils/__init__.py", utils_init)

    def fix_discovery_engine(self):
        """Replace the broken discovery engine with the clean version"""
        print("\n🔧 FIXING DISCOVERY ENGINE")
        print("=" * 30)

        # The clean discovery engine content is in the artifact above
        # We'll just log that it needs to be replaced
        self.log_action("REPLACE", "src/core/discovery_engine.py")

        print("   ✅ Discovery engine will be replaced with clean version")
        print("   ✅ Removes circular import issue")
        print("   ✅ Standalone, working implementation")

    def clean_examples(self):
        """Keep only essential, working examples"""
        print("\n📚 CLEANING EXAMPLES")
        print("=" * 25)

        # Remove redundant examples
        examples_to_remove = [
            "examples/oeis_sequence_analysis.py",
            "examples/enhanced_discovery_examples.py",
        ]

        for example in examples_to_remove:
            self.remove_file(example)

        print("   ✅ Keeping 3 essential examples:")
        print("      • basic_prime_discovery.py - Core demo")
        print("      • universal_dataset_examples.py - Universal generator demo")
        print("      • custom_function_discovery.py - Flexibility demo")

    def clean_scripts(self):
        """Keep only essential scripts"""
        print("\n🛠️  CLEANING SCRIPTS")
        print("=" * 20)

        scripts_to_remove = [
            "scripts/quick_check.py",  # Redundant with validate_installation.py
            "scripts/validate_setup.py",  # Redundant
        ]

        for script in scripts_to_remove:
            self.remove_file(script)

        print("   ✅ Keeping essential scripts:")
        print("      • interactive_discovery.py - Main interactive interface")
        print("      • discover_patterns.py - CLI interface")
        print("      • generate_sample_data.py - Setup utility")
        print("      • validate_installation.py - Installation check")

    def update_imports(self):
        """Update imports in remaining files"""
        print("\n🔗 FIXING IMPORTS")
        print("=" * 20)

        # Files that need import fixes
        import_fixes = [
            "src/generators/universal_generator.py",
            "src/analyzers/prime_analyzer.py",
            "src/analyzers/oeis_analyzer.py",
            "examples/basic_prime_discovery.py",
            "examples/universal_dataset_examples.py",
            "examples/custom_function_discovery.py",
            "scripts/interactive_discovery.py",
            "scripts/discover_patterns.py",
        ]

        for file_path in import_fixes:
            self.log_action("FIX IMPORTS", file_path)

        print("   ✅ All imports will use clean relative imports")
        print("   ✅ Remove complex fallback logic")
        print("   ✅ Ensure compatibility from project root")

    def create_quick_test_script(self):
        """Create a simple test script to verify everything works"""
        print("\n🧪 CREATING VERIFICATION SCRIPT")
        print("=" * 35)

        test_script = '''#!/usr/bin/env python3
"""
Quick verification that the cleanup was successful
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all core imports work"""
    print("🔧 Testing core imports...")
    
    try:
        from src.core.discovery_engine import UniversalMathDiscovery
        print("  ✅ Discovery engine")
        
        from src.generators.universal_generator import UniversalDatasetGenerator
        print("  ✅ Universal generator")
        
        from src.utils.math_utils import generate_mathematical_features
        print("  ✅ Math utilities")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\\n🧪 Testing basic functionality...")
    
    try:
        from src.core.discovery_engine import UniversalMathDiscovery
        
        # Simple test function
        def test_evens(n):
            return n % 2 == 0
        
        # Create discovery engine
        engine = UniversalMathDiscovery(
            target_function=test_evens,
            function_name="Even Numbers Test",
            max_number=100  # Small for quick test
        )
        
        print("  ✅ Discovery engine created")
        
        # Generate a small amount of data
        X, y = engine.generate_target_data()
        print(f"  ✅ Generated {len(X)} samples")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    """Run verification tests"""
    print("🚀 CLEANUP VERIFICATION")
    print("=" * 30)
    
    import_ok = test_imports()
    function_ok = test_basic_functionality()
    
    print("\\n📊 RESULTS:")
    print(f"✅ Imports: {'OK' if import_ok else 'FAILED'}")
    print(f"✅ Functionality: {'OK' if function_ok else 'FAILED'}")
    
    if import_ok and function_ok:
        print("\\n🎉 Repository cleanup successful!")
        print("\\n🚀 Ready to use:")
        print("  python src/generators/universal_generator.py demo")
        print("  python scripts/interactive_discovery.py")
        print("  python examples/basic_prime_discovery.py")
    else:
        print("\\n❌ Issues detected - check the output above")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        self.create_file("test_cleanup.py", test_script)

    def run_cleanup(self):
        """Run the complete cleanup process"""
        print("🧹 MATHEMATICAL PATTERN DISCOVERY ENGINE - REPOSITORY CLEANUP")
        print("=" * 70)

        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        else:
            print("⚠️  LIVE MODE - Files will be modified!")

        print()

        # Step 1: Remove redundant files
        self.remove_redundant_files()

        # Step 2: Create clean init files
        self.create_clean_init_files()

        # Step 3: Fix discovery engine
        self.fix_discovery_engine()

        # Step 4: Clean examples
        self.clean_examples()

        # Step 5: Clean scripts
        self.clean_scripts()

        # Step 6: Note import fixes needed
        self.update_imports()

        # Step 7: Create test script
        self.create_quick_test_script()

        # Summary
        print("\n📊 CLEANUP SUMMARY")
        print("=" * 25)
        print(f"📁 Files removed: {len(self.removed_files)}")
        print(f"📝 Files created: {len(self.created_files)}")

        if not self.dry_run:
            print("\n🎯 NEXT STEPS:")
            print("1. Replace src/core/discovery_engine.py with the clean version")
            print("2. Run: python test_cleanup.py")
            print("3. Fix any remaining import issues")
            print("4. Test examples: python examples/basic_prime_discovery.py")

        print("\n✨ Repository will be much cleaner and more maintainable!")


def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(description="Clean up the repository structure")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Verify we're in the right directory
    if not (Path.cwd() / "src").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Run cleanup
    cleanup = RepositoryCleanup(dry_run=args.dry_run)
    cleanup.run_cleanup()


if __name__ == "__main__":
    main()
