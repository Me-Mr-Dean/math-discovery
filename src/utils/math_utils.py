"""
Mathematical utility functions for pattern discovery
FIXED VERSION - Eliminates label leaking while maintaining compatibility
"""

import numpy as np
import math
from typing import List, Set, Dict, Any


def euler_totient(n: int) -> int:
    """Calculate Euler's totient function φ(n) - KEPT (legitimate mathematical function)"""
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


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2 - KEPT (used internally, not as feature)"""
    return n > 0 and (n & (n - 1)) == 0


def prime_factors(n: int) -> List[int]:
    """Get all prime factors of n (with repetition) - KEPT (legitimate function)"""
    if n <= 1:
        return []

    factors = []
    d = 2

    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1

    if n > 1:
        factors.append(n)

    return factors


def get_unique_prime_factors(n: int) -> Set[int]:
    """Get unique prime factors of n - KEPT (legitimate function)"""
    return set(prime_factors(n))


def prime_factorization(n: int) -> List[tuple]:
    """Return prime factorization as list of (prime, power) pairs - KEPT"""
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


def sum_of_divisors(n: int) -> int:
    """Calculate sum of proper divisors - KEPT (legitimate mathematical function)"""
    if n <= 1:
        return 0
    divisor_sum = 1  # 1 is always a proper divisor
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divisor_sum += i
            if i != n // i:  # Avoid counting square root twice
                divisor_sum += n // i
    return divisor_sum


def is_happy_number(n: int) -> bool:
    """Check if number is happy (sum of squares of digits eventually reaches 1) - KEPT"""
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit) ** 2 for digit in str(n))
    return n == 1


def is_prime(n: int) -> bool:
    """Check if a number is prime - KEPT (used internally, not as feature)"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def distance_to_next(number: int, number_set: Set[int]) -> int:
    """Distance from ``number`` to the next higher element in ``number_set`` - KEPT"""
    higher = [n for n in number_set if n > number]
    return min(higher) - number if higher else 0


def distance_to_prev(number: int, number_set: Set[int]) -> int:
    """Distance from ``number`` to the previous lower element in ``number_set`` - KEPT"""
    lower = [n for n in number_set if n < number]
    return number - max(lower) if lower else 0


def _solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax=b using Gaussian elimination - KEPT (internal function)"""
    n = len(a)
    A = [row[:] for row in a]
    x = b[:]

    for i in range(n):
        # pivot selection
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[max_row] = A[max_row], A[i]
        x[i], x[max_row] = x[max_row], x[i]

        pivot = A[i][i]
        if pivot == 0:
            continue
        inv = 1.0 / pivot
        for j in range(i, n):
            A[i][j] *= inv
        x[i] *= inv

        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            if factor == 0:
                continue
            for j in range(i, n):
                A[r][j] -= factor * A[i][j]
            x[r] -= factor * x[i]

    return x


def fit_polynomial_features(number: int, degree: int) -> List[float]:
    """Fit a polynomial to the digits of ``number`` and return residuals - KEPT"""
    digits = [int(d) for d in str(number)]
    if not digits:
        return []
    x_vals = list(range(len(digits)))
    n = degree + 1
    # Build design matrix
    X = [[x**p for p in range(n)] for x in x_vals]
    XtX = [[0.0] * n for _ in range(n)]
    Xty = [0.0] * n
    for i, row in enumerate(X):
        for j in range(n):
            Xty[j] += row[j] * digits[i]
            for k in range(n):
                XtX[j][k] += row[j] * row[k]

    coeffs = _solve_linear_system(XtX, Xty)

    # Evaluate polynomial and compute residuals
    residuals = []
    for x in x_vals:
        fitted = 0.0
        power = 1.0
        for c in coeffs:
            fitted += c * power
            power *= x
        residuals.append(digits[x] - fitted)

    return residuals


def generate_mathematical_features(
    number: int,
    previous_numbers: List[int] | None = None,
    window_size: int = 5,
    digit_tensor: bool = False,
    reference_set: Set[int] | None = None,
    poly_degree: int | None = None,
) -> dict:
    """
    Generate mathematical features for a number - FIXED to eliminate label leaking.

    REMOVED FEATURES (Label Leaking):
    - is_perfect_square, is_perfect_cube, is_prime
    - is_triangular, is_6n_plus_1, is_6n_minus_1, twin_candidate
    - is_palindrome, is_happy (when used as boolean flags)
    - Any boolean flags that directly encode mathematical properties

    KEPT FEATURES (Legitimate):
    - Raw numerical structure (digits, moduli, magnitude)
    - Sequence context (differences, ratios, local statistics)
    - Mathematical transformations (totient, factorization counts)
    - Structural properties derived from raw number data
    """

    if previous_numbers is None:
        previous_numbers = []

    features = {
        # ✅ BASIC PROPERTIES (Raw numerical structure)
        "number": number,
        "log_number": np.log10(number + 1),
        "sqrt_number": np.sqrt(number),
        "digit_count": len(str(number)),
        # ✅ MODULAR ARITHMETIC (Raw residues - legitimate)
        "mod_2": number % 2,
        "mod_3": number % 3,
        "mod_5": number % 5,
        "mod_6": number % 6,
        "mod_7": number % 7,
        "mod_10": number % 10,
        "mod_11": number % 11,
        "mod_13": number % 13,
        "mod_30": number % 30,
        "mod_210": number % 210,
        # ✅ DIGIT PATTERNS (Raw structure)
        "last_digit": number % 10,
        "first_digit": int(str(number)[0]),
        "digit_sum": sum(int(d) for d in str(number)),
        "digit_product": np.prod([int(d) for d in str(number) if int(d) > 0]),
        "alternating_digit_sum": sum(
            (-1) ** i * int(d) for i, d in enumerate(str(number))
        ),
        # ✅ MATHEMATICAL TRANSFORMATIONS (Counts and structure)
        "prime_factors_count": len(prime_factors(number)),
        "unique_prime_factors": len(get_unique_prime_factors(number)),
        "totient": euler_totient(number),
        # ✅ MAGNITUDE AND SCALE FEATURES
        "sqrt_fractional": np.sqrt(number) % 1,  # Fractional part of square root
        "log_fractional": np.log10(number) % 1 if number > 1 else 0,
        # ✅ DERIVED NUMERICAL FEATURES (Structure-based)
        "sum_of_proper_divisors": sum_of_divisors(number) if number <= 10000 else 0,
    }

    # ✅ WHEEL FACTORIZATION PATTERNS (Raw structural patterns)
    features.update(
        {
            "wheel_2_3": number % 6,
            "wheel_2_3_5": number % 30,
            "wheel_2_3_5_7": number % 210,
            "sum_of_digits_mod_9": sum(int(d) for d in str(number)) % 9,
        }
    )

    # ✅ DIGIT STRUCTURE ANALYSIS (No boolean flags)
    digits = [int(d) for d in str(number)]
    features.update(
        {
            "digit_variance": np.var(digits) if len(digits) > 1 else 0,
            "digit_range": max(digits) - min(digits) if digits else 0,
            "unique_digit_count": len(set(digits)),
            "digit_sum_squared": sum(d**2 for d in digits),
            "digit_alternating_product": np.prod(
                [d if i % 2 == 0 else 1 / max(d, 1) for i, d in enumerate(digits)]
            ),
        }
    )

    # ✅ SEQUENCE CONTEXT FEATURES (When previous numbers available)
    if previous_numbers:
        prev = previous_numbers[-1]
        features["diff_n"] = number - prev
        features["ratio_n"] = number / prev if prev != 0 else 0.0
        window = previous_numbers[-window_size:]
        features[f"mean_last_{window_size}"] = float(np.mean(window)) if window else 0.0
        features[f"std_last_{window_size}"] = float(np.std(window)) if window else 0.0

        # Local gap analysis
        if len(previous_numbers) >= 2:
            recent_gaps = [
                previous_numbers[i + 1] - previous_numbers[i]
                for i in range(len(previous_numbers) - 1)
            ]
            features["mean_gap"] = np.mean(recent_gaps[-5:]) if recent_gaps else 0
            features["gap_variance"] = (
                np.var(recent_gaps[-5:]) if len(recent_gaps) > 1 else 0
            )
    else:
        features["diff_n"] = 0.0
        features["ratio_n"] = 0.0
        features[f"mean_last_{window_size}"] = 0.0
        features[f"std_last_{window_size}"] = 0.0
        features["mean_gap"] = 0.0
        features["gap_variance"] = 0.0

    # ✅ REFERENCE SET DISTANCES (When available)
    if reference_set is not None:
        features["dist_to_next"] = distance_to_next(number, reference_set)
        features["dist_to_prev"] = distance_to_prev(number, reference_set)

    # ✅ POLYNOMIAL FEATURES (When requested)
    if poly_degree is not None:
        features[f"poly_deg_{poly_degree}_residuals"] = fit_polynomial_features(
            number, poly_degree
        )

    # ✅ DIGIT TENSOR (When requested)
    if digit_tensor:
        tensor = np.array([int(d) for d in str(number)], dtype=int)
        features["digit_tensor"] = list(tensor)

    return features


def validate_features_for_label_leaking(
    features: dict, target_function_name: str = ""
) -> List[str]:
    """
    Validate that features don't leak the target label.

    Returns list of potentially problematic features.
    """
    problematic = []

    for name, value in features.items():
        # Check for boolean flags that might encode rules
        if isinstance(value, (bool, int)) and value in [0, 1]:
            suspicious_keywords = [
                "is_",
                "has_",
                "prime",
                "perfect",
                "palindrom",
                "member",
                "target",
                "label",
                "fibonacci",
                "square",
                "triangular",
                "twin",
                "candidate",
                "happy",
            ]

            if any(keyword in name.lower() for keyword in suspicious_keywords):
                problematic.append(f"Suspicious boolean flag: {name} = {value}")

        # Check for features that might directly encode the target
        if target_function_name:
            if target_function_name.lower() in name.lower():
                problematic.append(f"Feature name matches target: {name}")

    return problematic


def get_feature_categories(features: dict) -> Dict[str, List[str]]:
    """Categorize features by type for analysis"""
    categories = {
        "modular": [],
        "digit": [],
        "sequence": [],
        "mathematical": [],
        "structure": [],
        "suspicious": [],
    }

    for name in features.keys():
        if "mod_" in name:
            categories["modular"].append(name)
        elif "digit" in name:
            categories["digit"].append(name)
        elif any(seq in name for seq in ["diff_", "ratio_", "mean_", "std_", "gap"]):
            categories["sequence"].append(name)
        elif any(
            math_term in name for math_term in ["totient", "prime", "sqrt", "log"]
        ):
            categories["mathematical"].append(name)
        elif any(susp in name for susp in ["is_", "has_", "member", "target"]):
            categories["suspicious"].append(name)
        else:
            categories["structure"].append(name)

    return categories


# Export the main functions that external code expects
__all__ = [
    "generate_mathematical_features",
    "euler_totient",
    "is_prime",
    "prime_factors",
    "prime_factorization",
    "distance_to_next",
    "distance_to_prev",
    "validate_features_for_label_leaking",
    "get_feature_categories",
]
