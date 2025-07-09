#!/usr/bin/env python3
"""
Fermat Prime Conjecture Test
============================

Computational investigation of the empirical conjecture:
"If φ(n) is a power of 2, then all prime factors of n must be Fermat primes"

This script tests whether every n with φ(n) = 2^k can be written as:
n = 2^a × 3^b × 5^c × 17^d × 257^e × 65537^f

Author: Mathematical Investigation
Purpose: Testing OEIS A007694 pattern hypothesis
"""

import math
from collections import defaultdict, Counter


def euler_totient(n):
    """
    Compute Euler's totient function φ(n).

    φ(n) counts the positive integers up to n that are coprime to n.

    Args:
        n (int): Positive integer

    Returns:
        int: φ(n)
    """
    if n == 1:
        return 1

    result = n
    p = 2

    # Apply the formula φ(n) = n * ∏(1 - 1/p) for all prime factors p
    while p * p <= n:
        if n % p == 0:
            # Remove all factors of p
            while n % p == 0:
                n //= p
            # Apply the multiplicative formula
            result -= result // p
        p += 1

    # If n > 1, then it's a prime factor
    if n > 1:
        result -= result // n

    return result


def is_power_of_2(n):
    """
    Check if n is a power of 2.

    Uses bit manipulation: powers of 2 have exactly one bit set.

    Args:
        n (int): Positive integer

    Returns:
        bool: True if n = 2^k for some k ≥ 0
    """
    return n > 0 and (n & (n - 1)) == 0


def get_prime_factors(n):
    """
    Get all prime factors of n (with repetition).

    Args:
        n (int): Positive integer

    Returns:
        list: List of prime factors
    """
    if n <= 1:
        return []

    factors = []
    d = 2

    # Trial division to find all prime factors
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1

    # If n > 1, then it's a prime factor
    if n > 1:
        factors.append(n)

    return factors


def get_unique_prime_factors(n):
    """
    Get unique prime factors of n (no repetition).

    Args:
        n (int): Positive integer

    Returns:
        set: Set of unique prime factors
    """
    return set(get_prime_factors(n))


def test_fermat_prime_conjecture(max_n=1000000, verbose=False, report_interval=100000):
    """
    Test the Fermat Prime Conjecture computationally.

    Args:
        max_n (int): Maximum value to test (default: 1,000,000)
        verbose (bool): Print detailed progress (default: False)
        report_interval (int): Progress reporting interval (default: 100,000)

    Returns:
        dict: Results summary
    """

    # Known Fermat primes (only 5 are known to exist)
    FERMAT_PRIMES = {2, 3, 5, 17, 257, 65537}

    # Results tracking
    total_tested = 0
    total_with_phi_power_of_2 = 0
    counterexamples = []
    valid_numbers = []
    fermat_prime_usage = defaultdict(int)
    phi_power_distribution = defaultdict(int)

    print(f"🔍 Testing Fermat Prime Conjecture up to n = {max_n:,}")
    print(f"Known Fermat primes: {sorted(FERMAT_PRIMES)}")
    print("=" * 60)

    for n in range(1, max_n + 1):
        total_tested += 1

        # Progress reporting
        if verbose and n % report_interval == 0:
            print(f"Progress: {n:,} / {max_n:,} ({100*n/max_n:.1f}%)")

        # Compute φ(n)
        phi_n = euler_totient(n)

        # Check if φ(n) is a power of 2
        if is_power_of_2(phi_n):
            total_with_phi_power_of_2 += 1

            # Track the power of 2
            power_of_2 = int(math.log2(phi_n)) if phi_n > 0 else 0
            phi_power_distribution[power_of_2] += 1

            # Get all unique prime factors of n
            prime_factors = get_unique_prime_factors(n)

            # Check if all prime factors are Fermat primes
            if prime_factors.issubset(FERMAT_PRIMES):
                # Valid case - follows the conjecture
                valid_numbers.append(n)

                # Track which Fermat primes are used
                for p in prime_factors:
                    fermat_prime_usage[p] += 1

            else:
                # Counterexample found!
                non_fermat_primes = prime_factors - FERMAT_PRIMES
                counterexamples.append(
                    {
                        "n": n,
                        "phi_n": phi_n,
                        "power_of_2": power_of_2,
                        "all_primes": sorted(prime_factors),
                        "non_fermat_primes": sorted(non_fermat_primes),
                    }
                )

                if verbose:
                    print(f"❌ COUNTEREXAMPLE: n={n}, φ({n})={phi_n}=2^{power_of_2}")
                    print(f"   Prime factors: {sorted(prime_factors)}")
                    print(f"   Non-Fermat primes: {sorted(non_fermat_primes)}")

    # Compile results
    results = {
        "total_tested": total_tested,
        "total_with_phi_power_of_2": total_with_phi_power_of_2,
        "total_valid": len(valid_numbers),
        "total_counterexamples": len(counterexamples),
        "counterexamples": counterexamples,
        "valid_numbers": valid_numbers,
        "fermat_prime_usage": dict(fermat_prime_usage),
        "phi_power_distribution": dict(phi_power_distribution),
        "conjecture_holds": len(counterexamples) == 0,
    }

    return results


def print_detailed_results(results):
    """
    Print detailed analysis of the test results.

    Args:
        results (dict): Results from test_fermat_prime_conjecture()
    """

    print("\n" + "=" * 60)
    print("📊 FERMAT PRIME CONJECTURE TEST RESULTS")
    print("=" * 60)

    # Summary statistics
    print(f"🔢 Total numbers tested: {results['total_tested']:,}")
    print(f"🎯 Numbers with φ(n) = 2^k: {results['total_with_phi_power_of_2']:,}")
    print(f"✅ Valid (Fermat primes only): {results['total_valid']:,}")
    print(f"❌ Counterexamples found: {results['total_counterexamples']:,}")

    # Percentage breakdown
    if results["total_with_phi_power_of_2"] > 0:
        valid_percentage = (
            results["total_valid"] / results["total_with_phi_power_of_2"]
        ) * 100
        print(f"📈 Conjecture success rate: {valid_percentage:.2f}%")

    # Conjecture verdict
    print(
        f"\n🏆 CONJECTURE STATUS: {'HOLDS' if results['conjecture_holds'] else 'FALSIFIED'}"
    )

    # Counterexamples detail
    if results["counterexamples"]:
        print(f"\n❌ COUNTEREXAMPLES FOUND:")
        print("-" * 40)
        for i, ce in enumerate(results["counterexamples"][:10], 1):  # Show first 10
            print(
                f"{i:2d}. n={ce['n']:,}, φ({ce['n']})={ce['phi_n']}=2^{ce['power_of_2']}"
            )
            print(f"    Prime factors: {ce['all_primes']}")
            print(f"    Non-Fermat: {ce['non_fermat_primes']}")

        if len(results["counterexamples"]) > 10:
            print(f"    ... and {len(results['counterexamples']) - 10} more")

    # Fermat prime usage analysis
    if results["fermat_prime_usage"]:
        print(f"\n📊 FERMAT PRIME USAGE FREQUENCY:")
        print("-" * 35)
        for prime in sorted(results["fermat_prime_usage"].keys()):
            count = results["fermat_prime_usage"][prime]
            percentage = (count / results["total_valid"]) * 100
            print(f"Prime {prime:5d}: {count:5,} times ({percentage:5.1f}%)")

    # φ(n) power distribution
    if results["phi_power_distribution"]:
        print(f"\n📊 φ(n) POWER DISTRIBUTION:")
        print("-" * 30)
        for power in sorted(results["phi_power_distribution"].keys()):
            count = results["phi_power_distribution"][power]
            print(f"φ(n) = 2^{power:2d}: {count:4,} numbers")

    # Some examples of valid numbers
    if results["valid_numbers"]:
        print(f"\n✅ EXAMPLES OF VALID NUMBERS:")
        print("-" * 35)
        examples = results["valid_numbers"][:20]  # First 20 examples
        print(f"First 20: {examples}")

        if len(results["valid_numbers"]) > 20:
            print(f"... and {len(results['valid_numbers']) - 20:,} more")


def main():
    """
    Main function to run the Fermat Prime Conjecture test.
    """

    print("🧮 FERMAT PRIME CONJECTURE COMPUTATIONAL TEST")
    print("=" * 60)
    print("Testing: If φ(n) = 2^k, then n = 2^a × 3^b × 5^c × 17^d × 257^e × 65537^f")
    print("(All prime factors of n must be Fermat primes)")

    # Configuration
    MAX_N = 1000000  # Test up to 1 million
    VERBOSE = True  # Show progress updates

    print(f"\nConfiguration:")
    print(f"• Testing range: 1 to {MAX_N:,}")
    print(f"• Progress updates: {'Enabled' if VERBOSE else 'Disabled'}")

    # Run the test
    print(f"\n🚀 Starting computational test...")
    results = test_fermat_prime_conjecture(
        max_n=MAX_N, verbose=VERBOSE, report_interval=100000
    )

    # Print detailed results
    print_detailed_results(results)

    # Final summary
    print(f"\n" + "=" * 60)
    if results["conjecture_holds"]:
        print("🎉 CONCLUSION: Fermat Prime Conjecture HOLDS for all tested values!")
        print("   This provides strong computational evidence for the pattern.")
    else:
        print("⚠️  CONCLUSION: Fermat Prime Conjecture FALSIFIED!")
        print(f"   Found {results['total_counterexamples']} counterexample(s).")

    print(f"\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()
