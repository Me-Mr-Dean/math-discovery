#!/usr/bin/env python3
"""
Command-line interface for mathematical pattern discovery
"""

import argparse
from src.core.discovery_engine import UniversalMathDiscovery

def main():
    parser = argparse.ArgumentParser(description="Discover mathematical patterns")
    parser.add_argument("--sequence", required=True, help="Comma-separated sequence")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Parse sequence
    sequence = [int(x.strip()) for x in args.sequence.split(",")]
    
    # Initialize discovery engine using the provided sequence
    engine = UniversalMathDiscovery(
        target_function=lambda n: n in sequence,
        function_name="Input Sequence",
        max_number=max(sequence),
    )

    prediction_fn = engine.run_complete_discovery()

    print("Discovered predictions for input sequence:")
    for n in sequence:
        result = prediction_fn(n)
        print(
            f"{n}: prediction={result['prediction']} (prob: {result['probability']:.3f})"
        )

if __name__ == "__main__":
    main()
