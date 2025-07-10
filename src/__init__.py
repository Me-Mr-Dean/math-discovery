"""
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
