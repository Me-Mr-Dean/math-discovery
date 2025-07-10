#!/usr/bin/env python3
"""
Command-line interface for sequence analysis
"""

import argparse
from src.analyzers import oeis_analyzer
from src.core.discovery_engine import UniversalMathDiscovery

def main():
    parser = argparse.ArgumentParser(description="Analyze mathematical sequences")
    parser.add_argument("--oeis", help="OEIS sequence ID (e.g., A000040)")
    parser.add_argument("--max-terms", type=int, default=1000, help="Maximum terms to analyze")
    
    args = parser.parse_args()
    
    if args.oeis:
        if args.oeis.upper() == "A007694":
            sequence = oeis_analyzer.generate_a007694_sequence(args.max_terms)
        else:
            print("Only A007694 analysis is supported in this demo.")
            return

        engine = UniversalMathDiscovery(
            target_function=lambda n: n in sequence,
            function_name=args.oeis.upper(),
            max_number=max(sequence),
        )

        prediction_fn = engine.run_complete_discovery()

        print(f"Analysis of {args.oeis} (first {len(sequence)} terms):")
        for n in sequence[:10]:
            result = prediction_fn(n)
            print(
                f"{n}: prediction={result['prediction']} (prob: {result['probability']:.3f})"
            )

if __name__ == "__main__":
    main()
