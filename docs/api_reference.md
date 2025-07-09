# API Reference

## Core Modules

### discovery_engine

Main discovery engine for pattern analysis.

```python
class DiscoveryEngine:
    def __init__(self, config_path=None):
        """Initialize the discovery engine."""
        pass
    
    def discover_patterns(self, sequence):
        """Discover patterns in a mathematical sequence."""
        pass
    
    def predict_next_terms(self, sequence, n_terms=5):
        """Predict the next n terms in a sequence."""
        pass
```

### feature_extractor

Mathematical feature extraction utilities.

```python
class FeatureExtractor:
    def extract_features(self, sequence):
        """Extract mathematical features from a sequence."""
        pass
    
    def modular_features(self, sequence, modulus):
        """Extract modular arithmetic features."""
        pass
```

## Analyzers

### prime_analyzer

Specialized prime number analysis.

### oeis_analyzer

OEIS sequence analysis and validation.

## Generators

### prime_generator

Prime number generation and dataset creation.

### sequence_generator

General mathematical sequence generation.
