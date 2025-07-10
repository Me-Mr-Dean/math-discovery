#!/usr/bin/env python3
"""
Example of discovering patterns in custom mathematical functions
"""

from src.core.discovery_engine import UniversalMathDiscovery
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
    engine = UniversalMathDiscovery(
        target_function=lambda n: n in fib_sequence,
        function_name="Fibonacci Numbers",
        max_number=max(fib_sequence),
    )

    prediction_fn = engine.run_complete_discovery()

    print("Predictions for Fibonacci sequence members:")
    for n in fib_sequence:
        result = prediction_fn(n)
        print(
            f"{n}: prediction={result['prediction']} (prob: {result['probability']:.3f})"
        )

if __name__ == "__main__":
    main()
