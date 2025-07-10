"""
Mathematical utility functions for pattern discovery
"""

import numpy as np
import math
from typing import List, Set


def euler_totient(n: int) -> int:
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


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0


def prime_factors(n: int) -> List[int]:
    """Get all prime factors of n (with repetition)"""
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
    """Get unique prime factors of n"""
    return set(prime_factors(n))


def prime_factorization(n: int) -> List[tuple]:
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


def sum_of_divisors(n: int) -> int:
    """Calculate sum of proper divisors"""
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
    """Check if number is happy (sum of squares of digits eventually reaches 1)"""
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit) ** 2 for digit in str(n))
    return n == 1




def is_prime(n: int) -> bool:
    """Check if a number is prime"""
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


def generate_mathematical_features(
    number: int,
    previous_numbers: List[int] | None = None,
    window_size: int = 5,
    digit_tensor: bool = False,
) -> dict:
    """Generate comprehensive mathematical features for a number.

    Args:
        number: The current integer to featurize.
        previous_numbers: Optional list of prior numbers in the sequence. When
            provided, difference/ratio and sliding window statistics will be
            computed using this history.
        window_size: Window length for mean/std statistics of ``previous_numbers``.
        digit_tensor: If True, include an array representation of the digits.
    """

    if previous_numbers is None:
        previous_numbers = []

    features = {
        # Basic properties
        "number": number,
        "log_number": np.log10(number + 1),
        "sqrt_number": np.sqrt(number),
        "digit_count": len(str(number)),
        
        # Modular arithmetic
        "mod_2": number % 2,
        "mod_3": number % 3,
        "mod_5": number % 5,
        "mod_6": number % 6,
        "mod_7": number % 7,
        "mod_10": number % 10,
        "mod_30": number % 30,
        "mod_210": number % 210,
        
        # Digit patterns
        "last_digit": number % 10,
        "first_digit": int(str(number)[0]),
        "digit_sum": sum(int(d) for d in str(number)),
        "digit_product": np.prod([int(d) for d in str(number) if int(d) > 0]),
        
        # Mathematical properties
        "is_perfect_square": int(int(np.sqrt(number)) ** 2 == number),
        "is_perfect_cube": int(round(number ** (1/3)) ** 3 == number),
        "is_power_of_2": int(is_power_of_2(number)),

        # Number theory
        "prime_factors_count": len(prime_factors(number)),
        "unique_prime_factors": len(get_unique_prime_factors(number)),
        "totient": euler_totient(number),
        "is_prime": is_prime(number),
        "sum_of_proper_divisors": sum_of_divisors(number) if number <= 10000 else 0,
    }

    # Additional modular arithmetic features used by the discovery engine
    features.update({
        "mod_11": number % 11,
        "mod_13": number % 13,
    })

    # Extra digit and positional patterns
    features.update({
        "alternating_digit_sum": sum((-1) ** i * int(d) for i, d in enumerate(str(number))),
        "is_triangular": int(int(((8 * number + 1) ** 0.5 - 1) / 2) ** 2 == number),
        "is_6n_plus_1": int(number % 6 == 1),
        "is_6n_minus_1": int(number % 6 == 5),
        "twin_candidate": int((number % 6) in [1, 5]),
        "wheel_2_3": number % 6,
        "wheel_2_3_5": number % 30,
        "wheel_2_3_5_7": number % 210,
        "sum_of_digits_mod_9": sum(int(d) for d in str(number)) % 9,
        "is_happy": int(is_happy_number(number)) if number <= 10000 else 0,
    })

    # Difference and ratio to the previous number
    if previous_numbers:
        prev = previous_numbers[-1]
        features["diff_n"] = number - prev
        features["ratio_n"] = number / prev if prev != 0 else 0.0
        window = previous_numbers[-window_size:]
        features[f"mean_last_{window_size}"] = float(np.mean(window)) if window else 0.0
        features[f"std_last_{window_size}"] = float(np.std(window)) if window else 0.0
    else:
        features["diff_n"] = 0.0
        features["ratio_n"] = 0.0
        features[f"mean_last_{window_size}"] = 0.0
        features[f"std_last_{window_size}"] = 0.0

    if digit_tensor:
        features["digit_tensor"] = np.array([int(d) for d in str(number)], dtype=int)

    return features
