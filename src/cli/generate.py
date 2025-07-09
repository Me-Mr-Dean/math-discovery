#!/usr/bin/env python3
"""
Command-line interface for data generation
"""

import argparse
from ..generators from src.generators import prime_generator

def main():
    parser = argparse.ArgumentParser(description="Generate mathematical datasets")
    parser.add_argument("--type", choices=["prime"], required=True, help="Type of data to generate")
    parser.add_argument("--count", type=int, default=1000, help="Number of items to generate")
    parser.add_argument("--output", required=True, help="Output file path")
    
    args = parser.parse_args()
    
    if args.type == "prime":
        generator = prime_generator.PrimeGenerator()
        data = generator.generate_primes(args.count)
        
        with open(args.output, "w") as f:
            for item in data:
                f.write(f"{item}\n")
        
        print(f"Generated {len(data)} primes to {args.output}")

if __name__ == "__main__":
    main()
