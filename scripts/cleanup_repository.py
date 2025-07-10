#!/usr/bin/env python3
"""
Repository Cleanup Script
Removes unnecessary files and organizes the project structure
"""

import os
import shutil
from pathlib import Path


def cleanup_repository():
    """Clean up unnecessary files from the repository"""

    project_root = Path(__file__).parent.parent

    print("🧹 Mathematical Pattern Discovery Engine - Repository Cleanup")
    print("=" * 65)
    print("This script will remove unnecessary/redundant files.")
    print()

    # Files and directories to remove
    files_to_remove = [
        # Backup directory (entire folder)
        "backup_original/",
        # Redundant files in root
        "numpy.py",  # Stub file no longer needed
        "install.py",  # Duplicate content
        # Redundant scripts
        "scripts/quick_test.py",  # pytest.skip, redundant with quick_check.py
        "scripts/test_import_fix.py",  # Setup-only, not permanent
        "scripts/validate_migration.py",  # Migration-specific
        "scripts/setup_environment.py",  # Replaced by install.py
        # Validation/testing files that aren't examples
        "examples/validate_real_performance.py",  # Validation, not example
    ]

    removed_count = 0
    skipped_count = 0

    print("🗑️  REMOVING UNNECESSARY FILES:")
    print("-" * 40)

    for file_path in files_to_remove:
        full_path = project_root / file_path

        try:
            if full_path.exists():
                if full_path.is_dir():
                    # Remove directory and all contents
                    shutil.rmtree(full_path)
                    print(f"  ✅ Removed directory: {file_path}")
                else:
                    # Remove single file
                    full_path.unlink()
                    print(f"  ✅ Removed file: {file_path}")
                removed_count += 1
            else:
                print(f"  ⚪ Already missing: {file_path}")
                skipped_count += 1

        except Exception as e:
            print(f"  ❌ Failed to remove {file_path}: {e}")

    print()
    print("📊 CLEANUP SUMMARY:")
    print("-" * 20)
    print(f"  Removed: {removed_count} items")
    print(f"  Skipped: {skipped_count} items (already missing)")

    # Show what's left
    print()
    print("✅ REMAINING CLEAN STRUCTURE:")
    print("-" * 35)

    essential_paths = [
        "src/core/",
        "src/analyzers/",
        "src/generators/",
        "src/utils/",
        "src/cli/",
        "examples/",
        "tests/",
        "docs/",
        "scripts/",
        "configs/",
        "data/",
    ]

    for path in essential_paths:
        full_path = project_root / path
        if full_path.exists():
            if full_path.is_dir():
                file_count = len(list(full_path.glob("*.py")))
                print(f"  📁 {path:<20} ({file_count} Python files)")
            else:
                print(f"  📄 {path}")
        else:
            print(f"  ❌ {path} - Missing")

    print()
    print("🎯 RECOMMENDED NEXT STEPS:")
    print("-" * 30)
    print("  1. Update .gitignore to ignore data/ directory")
    print("  2. Run: git add -A && git commit -m 'Clean up repository structure'")
    print("  3. Verify examples still work: python examples/basic_prime_discovery.py")
    print("  4. Update documentation if needed")

    return removed_count


def show_final_structure():
    """Show the clean final structure"""
    project_root = Path(__file__).parent.parent

    print("\n📂 FINAL CLEAN REPOSITORY STRUCTURE:")
    print("=" * 45)

    def show_tree(path, prefix="", max_depth=2, current_depth=0):
        """Show directory tree structure"""
        if current_depth >= max_depth:
            return

        try:
            items = sorted(path.iterdir())
            dirs = [
                item
                for item in items
                if item.is_dir() and not item.name.startswith(".")
            ]
            files = [
                item for item in items if item.is_file() and item.name.endswith(".py")
            ]

            # Show directories
            for i, item in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                connector = "└── " if is_last_dir else "├── "
                print(f"{prefix}{connector}{item.name}/")

                extension = "    " if is_last_dir else "│   "
                show_tree(item, prefix + extension, max_depth, current_depth + 1)

            # Show Python files
            for i, item in enumerate(files):
                is_last = i == len(files) - 1
                connector = "└── " if is_last else "├── "
                print(f"{prefix}{connector}{item.name}")

        except PermissionError:
            pass

    print("math-discovery/")
    show_tree(project_root)


def main():
    """Main cleanup function"""

    print("⚠️  This will permanently delete files. Continue? (y/N): ", end="")
    response = input().strip().lower()

    if response != "y":
        print("❌ Cleanup cancelled.")
        return

    print()
    removed_count = cleanup_repository()

    if removed_count > 0:
        print(f"\n🎉 Successfully cleaned up {removed_count} unnecessary items!")
        print("Your repository is now cleaner and more organized.")

        show_final_structure()

    else:
        print("\n✨ Repository was already clean!")

    print(f"\n🚀 Your Mathematical Pattern Discovery Engine is ready!")


if __name__ == "__main__":
    main()
