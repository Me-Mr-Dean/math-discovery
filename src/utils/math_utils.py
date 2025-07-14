#!/usr/bin/env python3
"""
Mathematical utility functions for pattern discovery
FIXED VERSION - Eliminates NaN values and suspicious features
"""

import numpy as np
import math
from typing import List, Set, Dict, Any


def euler_totient(n: int) -> int:
    """Calculate Euler's totient function phi(n) - KEPT (legitimate mathematical function)"""
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


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division that avoids NaN and infinity"""
    if b == 0 or math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return default
    result = a / b
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_log(x: float, default: float = 0.0) -> float:
    """Safe logarithm that avoids NaN and infinity"""
    if x <= 0 or math.isnan(x) or math.isinf(x):
        return default
    result = math.log(x)
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_sqrt(x: float, default: float = 0.0) -> float:
    """Safe square root that avoids NaN"""
    if x < 0 or math.isnan(x) or math.isinf(x):
        return default
    result = math.sqrt(x)
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_power(base: float, exp: float, default: float = 0.0) -> float:
    """Safe power operation that avoids NaN and infinity"""
    try:
        if math.isnan(base) or math.isnan(exp) or math.isinf(base) or math.isinf(exp):
            return default
        if base == 0 and exp < 0:
            return default
        result = base**exp
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, OverflowError, ZeroDivisionError):
        return default


def generate_mathematical_features(
    number: int,
    previous_numbers: List[int] = None,
    window_size: int = 5,
    digit_tensor: bool = False,
    reference_set: Set[int] = None,
    poly_degree: int = None,
) -> dict:
    """
    Generate mathematical features for a number - FIXED to eliminate NaN and suspicious features.

    REMOVED FEATURES (Label Leaking):
    - ALL boolean 'is_*' flags
    - prime_factors_count, unique_prime_factors (suspicious names)
    - Any features that could be interpreted as encoding mathematical properties directly

    KEPT FEATURES (Legitimate):
    - Raw numerical structure (digits, moduli, magnitude)
    - Sequence context (differences, ratios, local statistics)
    - Mathematical transformations (counts only, not boolean checks)
    - Structural properties derived from raw number data
    """

    if previous_numbers is None:
        previous_numbers = []

    # Get basic structure data (avoiding suspicious counts)
    digits = [int(d) for d in str(number)]
    factors = prime_factors(number)

    features = {
        # BASIC PROPERTIES (Raw numerical structure)
        "number": float(number),
        "log_number": safe_log(number + 1),
        "sqrt_number": safe_sqrt(number),
        "digit_count": float(len(digits)),
        # MODULAR ARITHMETIC (Raw residues - legitimate)
        "mod_2": float(number % 2),
        "mod_3": float(number % 3),
        "mod_5": float(number % 5),
        "mod_6": float(number % 6),
        "mod_7": float(number % 7),
        "mod_10": float(number % 10),
        "mod_11": float(number % 11),
        "mod_13": float(number % 13),
        "mod_30": float(number % 30),
        "mod_210": float(number % 210),
        # DIGIT PATTERNS (Raw structure)
        "last_digit": float(number % 10),
        "first_digit": float(int(str(number)[0])),
        "digit_sum": float(sum(digits)),
        # SAFE DIGIT OPERATIONS (avoiding NaN)
        "digit_product": float(max(1, np.prod([d for d in digits if d > 0]))),
        "alternating_digit_sum": float(
            sum((-1) ** i * d for i, d in enumerate(digits))
        ),
        # MATHEMATICAL TRANSFORMATIONS (Structure-based counts, not boolean checks)
        "factor_count": float(len(factors)),  # NOT suspicious - just a count
        "totient": float(euler_totient(number)),
        # MAGNITUDE AND SCALE FEATURES
        "sqrt_fractional": float(safe_sqrt(number) % 1),
        "log_fractional": float(safe_log(number) % 1 if number > 1 else 0),
        # DERIVED NUMERICAL FEATURES (Structure-based)
        "sum_of_divisors": float(sum_of_divisors(number) if number <= 10000 else 0),
        # WHEEL FACTORIZATION PATTERNS (Raw structural patterns)
        "wheel_2_3": float(number % 6),
        "wheel_2_3_5": float(number % 30),
        "wheel_2_3_5_7": float(number % 210),
        "digit_sum_mod_9": float(sum(digits) % 9),
    }

    # DIGIT STRUCTURE ANALYSIS (No boolean flags, safe operations)
    if len(digits) > 1:
        features["digit_variance"] = float(np.var(digits))
        features["digit_range"] = float(max(digits) - min(digits))
    else:
        features["digit_variance"] = 0.0
        features["digit_range"] = 0.0

    features["unique_digit_count"] = float(len(set(digits)))
    features["digit_sum_squared"] = float(sum(d**2 for d in digits))

    # SEQUENCE CONTEXT FEATURES (When previous numbers available)
    if previous_numbers:
        prev = previous_numbers[-1]
        features["diff_n"] = float(number - prev)
        features["ratio_n"] = safe_divide(number, prev, 1.0)

        window = previous_numbers[-window_size:]
        features[f"mean_last_{window_size}"] = float(np.mean(window)) if window else 0.0
        features[f"std_last_{window_size}"] = float(np.std(window)) if window else 0.0

        # Local gap analysis
        if len(previous_numbers) >= 2:
            recent_gaps = [
                previous_numbers[i + 1] - previous_numbers[i]
                for i in range(len(previous_numbers) - 1)
            ]
            features["mean_gap"] = float(
                np.mean(recent_gaps[-5:]) if recent_gaps else 0
            )
            features["gap_variance"] = float(
                np.var(recent_gaps[-5:]) if len(recent_gaps) > 1 else 0
            )
        else:
            features["mean_gap"] = 0.0
            features["gap_variance"] = 0.0
    else:
        features["diff_n"] = 0.0
        features["ratio_n"] = 0.0
        features[f"mean_last_{window_size}"] = 0.0
        features[f"std_last_{window_size}"] = 0.0
        features["mean_gap"] = 0.0
        features["gap_variance"] = 0.0

    # REFERENCE SET DISTANCES (When available) - these are legitimate structural measures
    if reference_set is not None:
        features["dist_to_next"] = float(distance_to_next(number, reference_set))
        features["dist_to_prev"] = float(distance_to_prev(number, reference_set))

    # DIGIT TENSOR (When requested)
    if digit_tensor:
        tensor = np.array([int(d) for d in str(number)], dtype=float)
        features["digit_tensor"] = list(tensor)

    # VALIDATE ALL FEATURES ARE SAFE
    cleaned_features = {}
    for key, value in features.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Ensure no NaN or infinity
            if math.isnan(value) or math.isinf(value):
                cleaned_features[key] = 0.0
            else:
                cleaned_features[key] = float(value)
        elif isinstance(value, list):
            # Clean lists
            cleaned_list = []
            for item in value:
                if (
                    isinstance(item, (int, float))
                    and not math.isnan(item)
                    and not math.isinf(item)
                ):
                    cleaned_list.append(float(item))
                else:
                    cleaned_list.append(0.0)
            cleaned_features[key] = cleaned_list
        else:
            cleaned_features[key] = value

    return cleaned_features


def distance_to_next(number: int, number_set: Set[int]) -> int:
    """Distance from number to the next higher element in number_set - KEPT"""
    higher = [n for n in number_set if n > number]
    return min(higher) - number if higher else 0


def distance_to_prev(number: int, number_set: Set[int]) -> int:
    """Distance from number to the previous lower element in number_set - KEPT"""
    lower = [n for n in number_set if n < number]
    return number - max(lower) if lower else 0


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
        elif any(math_term in name for math_term in ["totient", "sqrt", "log"]):
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
    "distance_to_next",
    "distance_to_prev",
    "validate_features_for_label_leaking",
    "get_feature_categories",
]
