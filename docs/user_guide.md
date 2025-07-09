# User Guide

## Getting Started

### Basic Usage

```python
from math_discovery.core import discovery_engine

# Create discovery engine
engine = discovery_engine.DiscoveryEngine()

# Analyze prime numbers
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
patterns = engine.discover_patterns(primes)
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
