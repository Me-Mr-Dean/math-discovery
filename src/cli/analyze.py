#!/usr/bin/env python3
"""
Command-line interface for sequence analysis
"""

import argparse
from ..analyzers import oeis_analyzer

def main():
    parser = argparse.ArgumentParser(description="Analyze mathematical sequences")
    parser.add_argument("--oeis", help="OEIS sequence ID (e.g., A000040)")
    parser.add_argument("--max-terms", type=int, default=1000, help="Maximum terms to analyze")
    
    args = parser.parse_args()
    
    if args.oeis:
        analyzer = oeis_analyzer.OEISAnalyzer()
        sequence = analyzer.load_sequence(args.oeis, max_terms=args.max_terms)
        results = analyzer.analyze_sequence(sequence)
        
        print(f"Analysis of {args.oeis}:")
        print(f"- Terms analyzed: {len(sequence)}")
        print(f"- Patterns found: {len(results.get('patterns', []))}")

if __name__ == "__main__":
    main()
