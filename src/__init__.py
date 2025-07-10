"""
Mathematical Pattern Discovery Engine

A machine learning-powered mathematical discovery system for uncovering
hidden patterns in number theory and mathematical sequences.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Lazy imports to avoid heavy dependencies during package initialization
# Submodules can be imported explicitly by users when needed.

__all__ = [
    "discovery_engine",
    "feature_extractor",
    "pattern_analyzer",
    "prime_generator",
    "sequence_generator",
    "prime_analyzer",
    "oeis_analyzer",
]

def __getattr__(name):
    if name == "discovery_engine":
        from .core import discovery_engine
        return discovery_engine
    if name == "feature_extractor":
        from .core import feature_extractor
        return feature_extractor
    if name == "pattern_analyzer":
        from .core import pattern_analyzer
        return pattern_analyzer
    if name == "prime_generator":
        from .generators import prime_generator
        return prime_generator
    if name == "sequence_generator":
        from .generators import sequence_generator
        return sequence_generator
    if name == "prime_analyzer":
        from .analyzers import prime_analyzer
        return prime_analyzer
    if name == "oeis_analyzer":
        from .analyzers import oeis_analyzer
        return oeis_analyzer
    raise AttributeError(f"module 'src' has no attribute {name}")
