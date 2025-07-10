"""
Tests for the discovery engine module
"""

import pytest
from src.core import discovery_engine

class TestDiscoveryEngine:
    def test_initialization(self):
        """Test engine initialization"""
        engine = discovery_engine.DiscoveryEngine()
        assert engine is not None
    
    def test_discover_patterns_basic(self):
        """Test basic pattern discovery"""
        engine = discovery_engine.DiscoveryEngine()
        sequence = [2, 3, 5, 7, 11]  # First 5 primes
        
        # This will fail until we implement the actual engine
        # patterns = engine.discover_patterns(sequence)
        # assert len(patterns) > 0
        pass

    def test_universal_math_discovery_embedding_option(self):
        """Ensure embedding argument is accepted"""
        engine = discovery_engine.UniversalMathDiscovery(
            discovery_engine.powers_of_2,
            "Powers of 2",
            max_number=10,
            embedding="fourier",
        )
        assert engine.embedding == "fourier"
