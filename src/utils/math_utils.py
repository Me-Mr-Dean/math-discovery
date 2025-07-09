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


def generate_mathematical_features(number: int) -> dict:
    """Generate comprehensive mathematical features for a number"""
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
    
    return features
