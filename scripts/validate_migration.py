"""
Basic validation test for migrated code
"""

import sys
from pathlib import Path

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all main modules can be imported"""
    try:
        from core import discovery_engine
        print("✅ Core discovery engine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import discovery engine: {e}")
        return False
    
    try:
        from generators import prime_generator
        print("✅ Prime generator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import prime generator: {e}")
        return False
    
    try:
        from utils import math_utils
        print("✅ Math utilities imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import math utilities: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    try:
        from utils.math_utils import euler_totient, is_power_of_2, generate_mathematical_features, is_prime
        
        # Test mathematical functions
        assert euler_totient(6) == 2, "Euler totient function failed"
        assert is_power_of_2(8) == True, "Power of 2 check failed"
        assert is_power_of_2(6) == False, "Power of 2 check failed"
        assert is_prime(17) == True, "Prime check failed"
        assert is_prime(15) == False, "Prime check failed"
        
        # Test feature generation
        features = generate_mathematical_features(17)
        assert features["number"] == 17, "Feature generation failed"
        assert features["is_prime"] == True or "is_prime" not in features, "Prime check failed"
        
        print("✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running post-migration validation tests...")
    print("=" * 50)
    
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if imports_ok and functionality_ok:
        print("✅ All validation tests passed! Migration successful.")
    else:
        print("❌ Some tests failed. Please check the migration.")
