#!/usr/bin/env python3
"""
Environment setup script for the Mathematical Pattern Discovery Engine
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    print("✓ Dependencies installed")

def create_data_directories():
    """Ensure data directories exist"""
    data_dirs = ["data/raw", "data/processed", "data/models", "data/results"]
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Data directories created")

def main():
    print("Setting up Mathematical Pattern Discovery Engine environment...")
    print("=" * 60)
    
    install_dependencies()
    create_data_directories()
    
    print("\n✅ Environment setup complete!")
    print("Run 'python scripts/validate_installation.py' to verify installation.")

if __name__ == "__main__":
    main()
