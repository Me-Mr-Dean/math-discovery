"""
Mathematical Pattern Discovery Engine

A machine learning-powered mathematical discovery system for uncovering
hidden patterns in number theory and mathematical sequences.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import discovery_engine, feature_extractor, pattern_analyzer
from .generators from src.generators import prime_generator, sequence_generator
from .analyzers import prime_analyzer, oeis_analyzer

__all__ = [
    "discovery_engine",
    "feature_extractor", 
    "pattern_analyzer",
    "prime_generator",
    "sequence_generator",
    "prime_analyzer",
    "oeis_analyzer"
]
