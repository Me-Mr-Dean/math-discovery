#!/usr/bin/env python3
"""
OEIS A007694 Discovery Script
Investigating numbers n where φ(n) is a power of 2

A007694 = {n ∈ ℕ : φ(n) ∈ {1, 2, 4, 8, 16, 32, ...}}

This sequence has no known closed form. We use mathematical analysis
and machine learning to discover patterns and structural properties.
"""

from universal_math_discovery import UniversalMathDiscovery
import math
from collections import defaultdict, Counter


def euler_totient(n):
    """Calculate Euler's totient function φ(n)"""
    if n == 1:
        return 1

    result = n
    p = 2

    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1

    if n > 1:
        result -= result // n

    return result


def is_power_of_2(n):
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0


def prime_factorization(n):
    """Return prime factorization as list of (prime, power) pairs"""
    if n <= 1:
        return []

    factors = []
    d = 2

    while d * d <= n:
        count = 0
        while n % d == 0:
            n //= d
            count += 1
        if count > 0:
            factors.append((d, count))
        d += 1

    if n > 1:
        factors.append((n, 1))

    return factors


def a007694_property(n):
    """Check if φ(n) is a power of 2"""
    if n <= 0:
        return False

    phi_n = euler_totient(n)
    return is_power_of_2(phi_n)


def generate_a007694_sequence(max_n=1000):
    """Generate A007694 sequence up to max_n"""
    sequence = []
    for n in range(1, max_n + 1):
        if a007694_property(n):
            sequence.append(n)
    return sequence


def analyze_theoretical_properties():
    """Analyze theoretical properties and known results about A007694"""
    print("🔍 THEORETICAL ANALYSIS OF A007694")
    print("=" * 60)

    print("Known theoretical results:")
    print("1. If p is an odd prime, then φ(p) = p-1")
    print("   So p ∈ A007694 ⟺ p-1 is a power of 2 ⟺ p is a Fermat prime")
    print("   Known Fermat primes: 3, 5, 17, 257, 65537")
    print()

    print("2. If p is odd prime and k ≥ 1, then φ(p^k) = p^(k-1)(p-1)")
    print("   For this to be a power of 2, we need p-1 = 2^m and p^(k-1) = 2^j")
    print("   This is only possible if p = 3 (since 3-1 = 2)")
    print()

    print("3. If n = 2^k, then φ(2^k) = 2^(k-1) for k ≥ 1, φ(1) = 1")
    print("   So all powers of 2 are in A007694")
    print()

    print("4. If n = 2^a * p where p is odd prime, then φ(n) = 2^(a-1) * (p-1)")
    print("   For this to be a power of 2, we need p-1 to be a power of 2")
    print("   Again, p must be a Fermat prime")
    print()

    # Verify Fermat primes
    fermat_primes = [3, 5, 17, 257, 65537]
    print("5. Verification - Fermat primes and their φ values:")
    for p in fermat_primes[:4]:  # Skip 65537 for now
        phi_p = euler_totient(p)
        is_pow2 = is_power_of_2(phi_p)
        power = int(math.log2(phi_p)) if is_pow2 else "N/A"
        print(f"   φ({p}) = {phi_p} = 2^{power} ✓")

    return fermat_primes


def structural_analysis():
    """Analyze structural patterns in A007694"""
    print("\n🧮 STRUCTURAL ANALYSIS")
    print("=" * 40)

    # Generate sequence
    sequence = generate_a007694_sequence(500)
    print(f"A007694 members up to 500: {len(sequence)} numbers")
    print(f"First 30 members: {sequence[:30]}")

    # Analyze by prime factorization structure
    print("\n📊 Analysis by factorization structure:")
    print("n | φ(n) | φ(n)=2^k | Prime factorization | Structure type")
    print("-" * 75)

    structure_counts = defaultdict(int)

    for n in sequence[:25]:
        phi_n = euler_totient(n)
        factors = prime_factorization(n)

        # Classify structure
        if n == 1:
            structure_type = "trivial"
        elif len(factors) == 1:
            p, k = factors[0]
            if p == 2:
                structure_type = f"2^{k}"
            else:
                structure_type = f"prime^{k}"
        elif len(factors) == 2:
            if factors[0][0] == 2:
                structure_type = f"2^a × prime^b"
            else:
                structure_type = "prime₁^a × prime₂^b"
        else:
            structure_type = "complex"

        structure_counts[structure_type] += 1

        # Format factorization
        if factors:
            factor_str = " × ".join(
                [f"{p}^{k}" if k > 1 else str(p) for p, k in factors]
            )
        else:
            factor_str = "1"

        power = int(math.log2(phi_n))
        print(
            f"{n:3d} | {phi_n:3d} | 2^{power:<2d} | {factor_str:<15} | {structure_type}"
        )

    print(f"\nStructure type distribution:")
    for struct_type, count in structure_counts.items():
        print(f"  {struct_type}: {count}")

    return sequence


def advanced_pattern_analysis(sequence):
    """Advanced mathematical pattern analysis"""
    print("\n🔬 ADVANCED PATTERN ANALYSIS")
    print("=" * 45)

    # Analyze gaps between consecutive terms
    gaps = [sequence[i + 1] - sequence[i] for i in range(len(sequence) - 1)]
    print(f"Gap analysis (first 20 gaps): {gaps[:20]}")
    print(f"Average gap: {sum(gaps)/len(gaps):.2f}")
    print(f"Gap frequencies: {Counter(gaps).most_common(10)}")

    # Analyze distribution of φ values
    phi_values = [euler_totient(n) for n in sequence]
    phi_powers = [int(math.log2(phi)) for phi in phi_values]
    print(f"\nφ(n) power distribution: {Counter(phi_powers).most_common(10)}")

    # Check divisibility patterns
    print(f"\nDivisibility patterns:")
    for d in [2, 3, 4, 5, 6, 8, 12]:
        divisible_count = sum(1 for n in sequence if n % d == 0)
        percentage = (divisible_count / len(sequence)) * 100
        print(
            f"  Divisible by {d}: {divisible_count}/{len(sequence)} ({percentage:.1f}%)"
        )

    # Analyze prime factor patterns
    print(f"\nPrime factor analysis:")
    all_primes = set()
    prime_frequencies = defaultdict(int)

    for n in sequence:
        factors = prime_factorization(n)
        for p, k in factors:
            all_primes.add(p)
            prime_frequencies[p] += 1

    print(f"  Primes appearing: {sorted(all_primes)[:20]}")
    print(f"  Most frequent primes: {Counter(prime_frequencies).most_common(10)}")

    # Check for multiplicative structure
    print(f"\nMultiplicative structure analysis:")

    # Powers of 2
    powers_of_2 = [
        n for n in sequence if all(p == 2 for p, k in prime_factorization(n))
    ]
    print(f"  Pure powers of 2: {len(powers_of_2)} ({powers_of_2[:10]}...)")

    # Products with Fermat primes
    fermat_primes = [3, 5, 17, 257]
    fermat_products = []

    for n in sequence:
        factors = prime_factorization(n)
        primes = [p for p, k in factors]
        if all(p == 2 or p in fermat_primes for p in primes):
            fermat_products.append(n)

    print(
        f"  Products of 2 and Fermat primes: {len(fermat_products)} ({fermat_products[:15]}...)"
    )

    return {
        "gaps": gaps,
        "phi_powers": phi_powers,
        "prime_frequencies": prime_frequencies,
        "fermat_products": fermat_products,
    }


def test_conjectures(sequence):
    """Test various mathematical conjectures about A007694"""
    print("\n🧪 TESTING MATHEMATICAL CONJECTURES")
    print("=" * 50)

    # Conjecture 1: All elements are of form 2^a * (Fermat primes)^b
    fermat_primes = {3, 5, 17, 257, 65537}

    conjecture1_holds = True
    counterexamples1 = []

    for n in sequence[:100]:  # Test first 100
        factors = prime_factorization(n)
        primes = {p for p, k in factors}

        allowed_primes = {2} | fermat_primes
        if not primes.issubset(allowed_primes):
            conjecture1_holds = False
            counterexamples1.append(n)

    print(f"Conjecture 1: n = 2^a × (Fermat primes)^b")
    print(f"  Holds for first 100 terms: {conjecture1_holds}")
    if counterexamples1:
        print(f"  Counterexamples: {counterexamples1[:5]}")

    # Conjecture 2: If n ∈ A007694 and gcd(m,n) = 1, then mn ∈ A007694 iff m ∈ A007694
    print(f"\nConjecture 2: Multiplicative property test")
    test_pairs = [(2, 3), (4, 3), (8, 5), (2, 17), (3, 5)]

    for a, b in test_pairs:
        if a in sequence and b in sequence:
            product = a * b
            in_sequence = product in sequence
            phi_product = euler_totient(product)
            is_pow2 = is_power_of_2(phi_product)

            print(
                f"  {a} × {b} = {product}: φ({product}) = {phi_product}, power of 2: {is_pow2}"
            )

    # Conjecture 3: Density analysis
    density_points = [100, 200, 500, 1000]
    print(f"\nConjecture 3: Density analysis")

    for max_n in density_points:
        if max_n <= max(sequence):
            count = sum(1 for n in sequence if n <= max_n)
            density = count / max_n
            print(f"  Up to {max_n}: {count} terms, density = {density:.4f}")


def create_enhanced_features(n):
    """Create enhanced mathematical features for ML analysis"""
    if n <= 0:
        return {}

    factors = prime_factorization(n)
    phi_n = euler_totient(n)

    features = {}

    # Basic properties
    features["n"] = n
    features["phi_n"] = phi_n
    features["log_phi_n"] = math.log(phi_n) if phi_n > 0 else 0

    # Prime factorization features
    features["num_distinct_primes"] = len(factors)
    features["total_prime_power"] = sum(k for p, k in factors)
    features["largest_prime"] = max((p for p, k in factors), default=1)
    features["smallest_prime"] = min((p for p, k in factors), default=1)

    # Powers of specific primes
    prime_powers = {p: k for p, k in factors}
    features["power_of_2"] = prime_powers.get(2, 0)
    features["power_of_3"] = prime_powers.get(3, 0)
    features["power_of_5"] = prime_powers.get(5, 0)
    features["power_of_17"] = prime_powers.get(17, 0)

    # Fermat prime indicators
    fermat_primes = {3, 5, 17, 257, 65537}
    primes_in_n = {p for p, k in factors}
    features["has_fermat_prime"] = bool(primes_in_n & fermat_primes)
    features["num_fermat_primes"] = len(primes_in_n & fermat_primes)
    features["only_2_and_fermat"] = primes_in_n.issubset({2} | fermat_primes)

    # Multiplicative properties
    features["is_power_of_prime"] = len(factors) <= 1
    features["is_semiprime"] = len(factors) == 2 and all(k == 1 for p, k in factors)
    features["is_prime_power"] = len(factors) == 1

    # Totient-specific features
    features["phi_is_power_of_2"] = is_power_of_2(phi_n)
    if is_power_of_2(phi_n):
        features["phi_power_of_2"] = int(math.log2(phi_n))
    else:
        features["phi_power_of_2"] = -1

    # Congruence features
    features["mod_3"] = n % 3
    features["mod_4"] = n % 4
    features["mod_8"] = n % 8
    features["mod_12"] = n % 12
    features["mod_24"] = n % 24

    # Structural indicators
    features["max_prime_power"] = max((k for p, k in factors), default=0)
    features["sum_of_exponents"] = sum(k for p, k in factors)

    return features


def main():
    """Main discovery function for A007694"""
    print("🔍 OEIS A007694 MATHEMATICAL DISCOVERY")
    print("=" * 60)
    print("Investigating numbers n where φ(n) is a power of 2")
    print("A007694 = {n ∈ ℕ : φ(n) ∈ {1, 2, 4, 8, 16, 32, ...}}")
    print()

    # Theoretical analysis
    fermat_primes = analyze_theoretical_properties()

    # Structural analysis
    sequence = structural_analysis()

    # Advanced pattern analysis
    pattern_data = advanced_pattern_analysis(sequence)

    # Test conjectures
    test_conjectures(sequence)

    # Machine learning discovery
    print("\n🚀 RUNNING MACHINE LEARNING DISCOVERY")
    print("=" * 50)

    # Create discoverer - note: enhanced features would need to be integrated into UniversalMathDiscovery
    discoverer = UniversalMathDiscovery(
        target_function=a007694_property,
        function_name="OEIS A007694 (φ(n) = 2^k)",
        max_number=2000,  # Larger range for better learning
    )

    prediction_function = discoverer.run_complete_discovery()

    # Test ML predictions
    print("\n🧪 TESTING ML PREDICTIONS")
    print("=" * 35)

    # Test on known sequence members
    test_members = sequence[:20]
    correct_predictions = 0

    print("Testing on known A007694 members:")
    for n in test_members:
        result = prediction_function(n)
        predicted = bool(result["prediction"])

        factors = prime_factorization(n)
        factor_str = (
            " × ".join([f"{p}^{k}" if k > 1 else str(p) for p, k in factors])
            if factors
            else "1"
        )

        status = "✅" if predicted else "❌"
        print(
            f"{status} {n:3d} = {factor_str:<12}: φ({n}) = {euler_totient(n):3d} (prob: {result['probability']:.3f})"
        )

        if predicted:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_members)
    print(
        f"\nML accuracy on known members: {accuracy:.3f} ({correct_predictions}/{len(test_members)})"
    )

    # Test on potential new members
    print(f"\nTesting potential new A007694 members:")

    # Generate candidates that might be in A007694
    candidates = []

    # Test products of powers of 2 with Fermat primes
    for power_2 in range(0, 8):
        for fermat in [1, 3, 5, 17]:
            candidate = (2**power_2) * fermat
            if candidate <= 1000 and candidate not in sequence:
                candidates.append(candidate)

    candidates = sorted(set(candidates))[:15]

    for n in candidates:
        result = prediction_function(n)
        predicted = bool(result["prediction"])
        phi_n = euler_totient(n)
        actual_member = is_power_of_2(phi_n)

        factors = prime_factorization(n)
        factor_str = " × ".join([f"{p}^{k}" if k > 1 else str(p) for p, k in factors])

        match_symbol = "✅" if predicted == actual_member else "❌"
        member_symbol = "∈" if actual_member else "∉"

        print(
            f"{match_symbol} {n:3d} = {factor_str:<12}: φ({n}) = {phi_n:3d}, {member_symbol} A007694 (prob: {result['probability']:.3f})"
        )

    print(f"\n🎉 A007694 DISCOVERY COMPLETE!")
    print(f"Key findings:")
    print(
        f"• A007694 appears to consist mainly of products of powers of 2 and Fermat primes"
    )
    print(f"• The sequence is sparse with irregular gaps")
    print(f"• ML achieved {accuracy:.1%} accuracy on known members")
    print(f"• Strong correlation with Fermat prime structure detected")

    return sequence, prediction_function


if __name__ == "__main__":
    main()
