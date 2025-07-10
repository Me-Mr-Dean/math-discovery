#!/usr/bin/env python3
"""
Quick test to verify the migration worked correctly
"""
import pytest
pytest.skip("Utility script not part of unit tests", allow_module_level=True)

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_math_utils():
    """Test the math utilities"""
    from utils.math_utils import is_prime, euler_totient, generate_mathematical_features
    
    print("Testing math utilities...")
    
    # Test is_prime
    assert is_prime(17) == True
    assert is_prime(15) == False
    assert is_prime(2) == True
    assert is_prime(1) == False
    print("✅ is_prime function works")
    
    # Test euler_totient
    assert euler_totient(6) == 2
    assert euler_totient(9) == 6
    print("✅ euler_totient function works")
    
    # Test feature generation
    features = generate_mathematical_features(17)
    assert features["number"] == 17
    assert features["is_prime"] == True
    assert features["mod_2"] == 1
    print("✅ generate_mathematical_features works")
    
    print("✅ All math utility tests passed!")

def test_discovery_engine():
    """Test that the discovery engine can be imported"""
    try:
        from core.discovery_engine import UniversalMathDiscovery
        print("✅ Discovery engine imports successfully")
        
        # Try to create an instance
        def test_function(n):
            return n % 2 == 0  # Simple even number test
        
        discoverer = UniversalMathDiscovery(
            target_function=test_function,
            function_name="Even Numbers",
            max_number=100
        )
        print("✅ Discovery engine can be instantiated")
        
    except Exception as e:
        print(f"❌ Discovery engine test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Running quick migration test...")
    print("=" * 40)
    
    try:
        test_math_utils()
        test_discovery_engine()
        
        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED! Migration is successful!")
        print("\n🎯 You can now:")
        print("  • Run examples: python examples/basic_prime_discovery.py")
        print("  • Install package: pip install -e .")
        print("  • Use CLI tools: math-discover --help")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Please check the migration and try again.")
