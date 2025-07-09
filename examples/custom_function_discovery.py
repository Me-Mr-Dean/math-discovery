#!/usr/bin/env python3
"""
Example of discovering patterns in custom mathematical functions
"""

from math_discovery.core import discovery_engine
import math

def fibonacci(n):
    """Generate first n Fibonacci numbers"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def main():
    # Generate Fibonacci sequence
    fib_sequence = fibonacci(50)
    
    # Initialize discovery engine
    engine = discovery_engine.DiscoveryEngine()
    
    # Discover patterns
    patterns = engine.discover_patterns(fib_sequence)
    
    print("Patterns discovered in Fibonacci sequence:")
    for pattern in patterns:
        print(f"- {pattern}")

if __name__ == "__main__":
    main()
