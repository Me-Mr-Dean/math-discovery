#!/usr/bin/env python3
"""
Example of OEIS sequence analysis
"""

from math_discovery.analyzers import oeis_analyzer

def main():
    # Analyze OEIS sequence A000040 (prime numbers)
    analyzer = oeis_analyzer.OEISAnalyzer()
    
    # Load sequence from OEIS
    sequence = analyzer.load_sequence("A000040", max_terms=1000)
    
    # Analyze patterns
    analysis = analyzer.analyze_sequence(sequence)
    
    print(f"Analysis of OEIS sequence A000040:")
    print(f"- Sequence length: {len(sequence)}")
    print(f"- Detected patterns: {analysis['patterns']}")
    print(f"- Confidence scores: {analysis['confidence']}")

if __name__ == "__main__":
    main()
