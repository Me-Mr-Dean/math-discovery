"""
Path resolution utilities for Mathematical Pattern Discovery Engine
Handles finding data files, project root, and cross-platform paths
"""

import os
import sys
from pathlib import Path
from typing import Optional, List


def get_project_root() -> Path:
    """
    Find the project root directory by looking for setup.py or src/ directory

    Returns:
        Path: The project root directory
    """
    current = Path(__file__).parent

    # Walk up the directory tree looking for indicators of project root
    for parent in [current] + list(current.parents):
        # Look for setup.py (definitive indicator)
        if (parent / "setup.py").exists():
            return parent

        # Look for src directory with our modules
        if (parent / "src" / "core").exists():
            return parent

        # Look for README.md and requirements.txt together
        if (parent / "README.md").exists() and (parent / "requirements.txt").exists():
            return parent

    # Fallback: assume we're in src/utils/ and go up two levels
    return current.parent.parent


def find_data_file(
    filename: str, subdirs: Optional[List[str]] = None
) -> Optional[Path]:
    """
    Find a data file in various possible locations

    Args:
        filename: Name of the file to find
        subdirs: List of subdirectories to search in (default: common data dirs)

    Returns:
        Path object if file found, None otherwise
    """
    if subdirs is None:
        subdirs = ["data/raw", "data", ".", "examples", "tests"]

    project_root = get_project_root()

    # Search locations in order of preference
    search_paths = []

    # 1. Relative to project root
    for subdir in subdirs:
        search_paths.append(project_root / subdir / filename)

    # 2. Relative to current working directory
    for subdir in subdirs:
        search_paths.append(Path.cwd() / subdir / filename)

    # 3. In current directory
    search_paths.append(Path.cwd() / filename)

    # 4. Common variations of the filename
    if not filename.endswith("_sample.csv") and filename.endswith(".csv"):
        base_name = filename[:-4]
        sample_name = f"{base_name}_sample.csv"
        for subdir in subdirs:
            search_paths.append(project_root / subdir / sample_name)

    # Search for the file
    for path in search_paths:
        if path.exists() and path.is_file():
            return path

    return None


def find_dataset_file(dataset_name: str) -> Path:
    """
    Find a specific dataset file with helpful error messages

    Args:
        dataset_name: Name of the dataset (e.g., 'ml_dataset1_odd_endings')

    Returns:
        Path: Path to the dataset file

    Raises:
        FileNotFoundError: If dataset file cannot be found
    """
    # Try various filename patterns
    possible_names = [
        f"{dataset_name}.csv",
        f"{dataset_name}_sample.csv",
    ]

    for name in possible_names:
        path = find_data_file(name)
        if path:
            return path

    # File not found - provide helpful error message
    project_root = get_project_root()

    error_msg = f"Dataset '{dataset_name}' not found!\n\n"
    error_msg += "Searched in these locations:\n"

    for name in possible_names:
        for subdir in ["data/raw", "data", "."]:
            search_path = project_root / subdir / name
            error_msg += f"  • {search_path}\n"

    error_msg += "\n🔧 To fix this:\n"
    error_msg += "  1. Run: python scripts/generate_sample_data.py\n"
    error_msg += "  2. Or place your dataset file in: data/raw/\n"
    error_msg += "  3. Or run from the project root directory\n"

    raise FileNotFoundError(error_msg)


def setup_project_paths():
    """
    Add project src directory to Python path for imports
    Call this at the start of examples and scripts
    """
    project_root = get_project_root()
    src_path = project_root / "src"

    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def get_data_directory(create: bool = False) -> Path:
    """
    Get the main data directory path

    Args:
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path: Path to the data directory
    """
    project_root = get_project_root()
    data_dir = project_root / "data" / "raw"

    if create and not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def ensure_data_structure():
    """
    Ensure the data directory structure exists
    Creates directories and .gitkeep files as needed
    """
    project_root = get_project_root()

    dirs_to_create = [
        "data/raw",
        "data/processed",
        "data/models",
        "data/results",
    ]

    for dir_path in dirs_to_create:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)

        # Create .gitkeep file to preserve directory in git
        gitkeep_path = full_path / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.write_text("# Keep this directory in git\n")


def validate_environment():
    """
    Validate that the environment is set up correctly

    Returns:
        dict: Validation results with status and messages
    """
    results = {
        "valid": True,
        "issues": [],
        "warnings": [],
    }

    project_root = get_project_root()

    # Check for essential files
    essential_files = [
        "setup.py",
        "requirements.txt",
        "src/core/discovery_engine.py",
    ]

    for file_path in essential_files:
        full_path = project_root / file_path
        if not full_path.exists():
            results["valid"] = False
            results["issues"].append(f"Missing essential file: {file_path}")

    # Check for data directory
    data_dir = project_root / "data" / "raw"
    if not data_dir.exists():
        results["warnings"].append(
            "Data directory not found - run generate_sample_data.py"
        )

    # Check for sample data files
    sample_files = [
        "data/raw/1m.csv",
        "data/raw/ml_dataset1_odd_endings_sample.csv",
    ]

    missing_samples = []
    for file_path in sample_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_samples.append(file_path)

    if missing_samples:
        results["warnings"].append(
            f"Missing sample data files: {', '.join(missing_samples)}"
        )

    return results
