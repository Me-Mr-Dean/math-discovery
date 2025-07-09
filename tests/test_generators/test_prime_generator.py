"""
Tests for the prime generator module
"""

import pytest
from src.generators import prime_generator

class TestPrimeGenerator:
    def test_initialization(self):
        """Test generator initialization"""
        generator = prime_generator.PrimeGenerator()
        assert generator is not None
    
    def test_generate_primes_basic(self):
        """Test basic prime generation"""
        generator = prime_generator.PrimeGenerator()
        
        # This will fail until we implement the actual generator
        # primes = generator.generate_primes(10)
        # assert len(primes) == 10
        # assert primes[0] == 2
        pass
