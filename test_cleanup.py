#!/usr/bin/env python3
"""
Quick verification that the cleanup was successful
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all core imports work"""
    print("🔧 Testing core imports...")
    
    try:
        from src.core.discovery_engine import UniversalMathDiscovery
        print("  ✅ Discovery engine")
        
        from src.generators.universal_generator import UniversalDatasetGenerator
        print("  ✅ Universal generator")
        
        from src.utils.math_utils import generate_mathematical_features
        print("  ✅ Math utilities")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from src.core.discovery_engine import UniversalMathDiscovery
        
        # Simple test function
        def test_evens(n):
            return n % 2 == 0
        
        # Create discovery engine
        engine = UniversalMathDiscovery(
            target_function=test_evens,
            function_name="Even Numbers Test",
            max_number=100  # Small for quick test
        )
        
        print("  ✅ Discovery engine created")
        
        # Generate a small amount of data
        X, y = engine.generate_target_data()
        print(f"  ✅ Generated {len(X)} samples")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    """Run verification tests"""
    print("🚀 CLEANUP VERIFICATION")
    print("=" * 30)
    
    import_ok = test_imports()
    function_ok = test_basic_functionality()
    
    print("\n📊 RESULTS:")
    print(f"✅ Imports: {'OK' if import_ok else 'FAILED'}")
    print(f"✅ Functionality: {'OK' if function_ok else 'FAILED'}")
    
    if import_ok and function_ok:
        print("\n🎉 Repository cleanup successful!")
        print("\n🚀 Ready to use:")
        print("  python src/generators/universal_generator.py demo")
        print("  python scripts/interactive_discovery.py")
        print("  python examples/basic_prime_discovery.py")
    else:
        print("\n❌ Issues detected - check the output above")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
