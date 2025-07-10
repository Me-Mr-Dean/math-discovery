# API Reference

## Core Modules

### discovery_engine

Main discovery engine for pattern analysis.
Supports difference/ratio features and optional Fourier or PCA embeddings of
digit sequences.

```python
class UniversalMathDiscovery:
    def __init__(
        self,
        target_function,
        function_name,
        max_number=100000,
        embedding=None,
        embedding_components=None,
    ):
        """Initialize the discovery engine."""
        ...

    def run_complete_discovery(self):
        """Run training and analysis on the target function."""
        ...
```

### feature_extractor

Mathematical feature extraction utilities. When previous numbers are supplied,
the extractor computes difference and ratio features and sliding window
statistics.

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

`PrimeCSVGenerator` exposes a helper `generate_prefix_suffix_dataset()` that
wraps ``generate_prefix_suffix_matrix`` from ``src.utils``.  It allows building
custom prefix-suffix matrices from any list of numbers.

```python
from src.generators.prime_generator import PrimeCSVGenerator

gen = PrimeCSVGenerator()
gen.load_primes()
df = gen.generate_prefix_suffix_dataset(prefix_digits=2, suffix_digits=2)
```

### sequence_generator

General mathematical sequence generation.
