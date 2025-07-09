#!/usr/bin/env python3
"""
Command-line interface for mathematical pattern discovery
"""

import argparse
from ..core import discovery_engine

def main():
    parser = argparse.ArgumentParser(description="Discover mathematical patterns")
    parser.add_argument("--sequence", required=True, help="Comma-separated sequence")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Parse sequence
    sequence = [int(x.strip()) for x in args.sequence.split(",")]
    
    # Initialize engine
    engine = discovery_engine.DiscoveryEngine(config_path=args.config)
    
    # Discover patterns
    patterns = engine.discover_patterns(sequence)
    
    print("Discovered patterns:")
    for pattern in patterns:
        print(f"- {pattern}")

if __name__ == "__main__":
    main()
