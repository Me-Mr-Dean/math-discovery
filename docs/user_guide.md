# User Guide

## Getting Started

### Basic Usage

```python
# Import the engine from ``src`` once installed
from src.core.discovery_engine import UniversalMathDiscovery

# Create discovery engine
engine = UniversalMathDiscovery(lambda n: n in [2, 3, 5, 7, 11, 13], "Primes")

# Analyze prime numbers
patterns = engine.run_complete_discovery()
```

### Command Line Interface

```bash
# Discover patterns in a sequence
math-discover --sequence "2,3,5,7,11,13,17,19,23"

# Generate prime datasets
math-generate --type prime --count 1000 --output data/primes.csv

# Analyze OEIS sequences
math-analyze --oeis A000040 --max-terms 10000
```

## Configuration

The system uses YAML configuration files in the `configs/` directory:

- `default_config.yaml` - General settings
- `prime_config.yaml` - Prime number analysis
- `oeis_config.yaml` - OEIS sequence analysis

## New Capabilities

- Difference and ratio features automatically computed when a history is
  provided.
- `PrimeCSVGenerator.generate_prefix_suffix_dataset` creates custom
  prefix–suffix matrices for training.
- Use `embedding="fourier"` or `embedding="pca"` in `UniversalMathDiscovery`
  to embed digit patterns before model training.
