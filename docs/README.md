# Mathematical Pattern Discovery Engine Documentation

Welcome to the documentation for the Mathematical Pattern Discovery Engine.

## Contents

- [Installation Guide](installation.md)
- [User Guide](user_guide.md)
- [API Reference](api_reference.md)
- [Mathematical Basis](mathematical_basis.md)
- [Examples](examples/)

## Quick Start

```python
from math_discovery.core.discovery_engine import UniversalMathDiscovery

# Initialize the discovery engine
engine = UniversalMathDiscovery(lambda n: n % 2 == 0, "Even Numbers")

# Analyze a mathematical sequence
results = engine.run_complete_discovery()
```

## Project Structure

The project is organized into several key modules:

- `core/` - Main discovery engine and pattern analysis
- `generators/` - Data generation and sequence creation
- `analyzers/` - Specialized mathematical analyzers
- `utils/` - Utility functions and helpers
- `cli/` - Command-line interfaces

## New Capabilities

- Difference and ratio-based features for sequence analysis
- Prefix–suffix dataset generation utilities
- Optional Fourier or PCA embeddings of digit sequences
