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
from math_discovery.core import discovery_engine

# Initialize the discovery engine
engine = discovery_engine.DiscoveryEngine()

# Analyze a mathematical sequence
results = engine.discover_patterns([2, 3, 5, 7, 11, 13, 17, 19, 23])
```

## Project Structure

The project is organized into several key modules:

- `core/` - Main discovery engine and pattern analysis
- `generators/` - Data generation and sequence creation
- `analyzers/` - Specialized mathematical analyzers
- `utils/` - Utility functions and helpers
- `cli/` - Command-line interfaces
