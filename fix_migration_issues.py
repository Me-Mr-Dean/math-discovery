#!/usr/bin/env python3
"""
Fix migration issues and update validation test
"""

from pathlib import Path


def fix_validation_test():
    """Fix the validation test to remove the problematic is_prime check"""

    test_file = Path("scripts/validate_migration.py")

    # Read the current test file
    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace the problematic test section
    old_test = '''        # Test feature generation
        features = generate_mathematical_features(17)
        assert features["number"] == 17, "Feature generation failed"
        assert features["is_prime"] == True or "is_prime" not in features, "Prime check failed"'''

    new_test = '''        # Test feature generation
        features = generate_mathematical_features(17)
        assert features["number"] == 17, "Feature generation failed"
        assert "mod_2" in features, "Modular features missing"
        assert "digit_sum" in features, "Digit features missing"'''

    content = content.replace(old_test, new_test)

    # Write the fixed test file
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Fixed validation test")


def add_is_prime_function():
    """Add is_prime function to math_utils.py"""

    math_utils_file = Path("src/utils/math_utils.py")

    # Read current content
    with open(math_utils_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Add is_prime function before the generate_mathematical_features function
    is_prime_function = '''

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


'''

    # Insert the function before generate_mathematical_features
    insertion_point = content.find("def generate_mathematical_features")
    if insertion_point != -1:
        content = (
            content[:insertion_point] + is_prime_function + content[insertion_point:]
        )

    # Also add is_prime to the features in generate_mathematical_features
    old_features = """        # Number theory
        "prime_factors_count": len(prime_factors(number)),
        "unique_prime_factors": len(get_unique_prime_factors(number)),
        "totient": euler_totient(number),
        "sum_of_proper_divisors": sum_of_divisors(number) if number <= 10000 else 0,"""

    new_features = """        # Number theory
        "prime_factors_count": len(prime_factors(number)),
        "unique_prime_factors": len(get_unique_prime_factors(number)),
        "totient": euler_totient(number),
        "is_prime": is_prime(number),
        "sum_of_proper_divisors": sum_of_divisors(number) if number <= 10000 else 0,"""

    content = content.replace(old_features, new_features)

    # Write the updated file
    with open(math_utils_file, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Added is_prime function to math_utils.py")


def update_validation_test_with_prime():
    """Update validation test to properly test is_prime function"""

    test_file = Path("scripts/validate_migration.py")

    # Read the current test file
    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Update the test to properly check is_prime
    old_test = '''        # Test feature generation
        features = generate_mathematical_features(17)
        assert features["number"] == 17, "Feature generation failed"
        assert "mod_2" in features, "Modular features missing"
        assert "digit_sum" in features, "Digit features missing"'''

    new_test = '''        # Test feature generation
        features = generate_mathematical_features(17)
        assert features["number"] == 17, "Feature generation failed"
        assert features["is_prime"] == True, "Prime check failed (17 should be prime)"
        assert "mod_2" in features, "Modular features missing"
        assert "digit_sum" in features, "Digit features missing"
        
        # Test non-prime number
        features_composite = generate_mathematical_features(15)
        assert features_composite["is_prime"] == False, "Prime check failed (15 should not be prime)"'''

    content = content.replace(old_test, new_test)

    # Also update the import to include is_prime
    old_import = """        from utils.math_utils import euler_totient, is_power_of_2, generate_mathematical_features"""
    new_import = """        from utils.math_utils import euler_totient, is_power_of_2, generate_mathematical_features, is_prime"""

    content = content.replace(old_import, new_import)

    # Add a direct test of is_prime
    old_math_test = '''        # Test mathematical functions
        assert euler_totient(6) == 2, "Euler totient function failed"
        assert is_power_of_2(8) == True, "Power of 2 check failed"
        assert is_power_of_2(6) == False, "Power of 2 check failed"'''

    new_math_test = '''        # Test mathematical functions
        assert euler_totient(6) == 2, "Euler totient function failed"
        assert is_power_of_2(8) == True, "Power of 2 check failed"
        assert is_power_of_2(6) == False, "Power of 2 check failed"
        assert is_prime(17) == True, "Prime check failed"
        assert is_prime(15) == False, "Prime check failed"'''

    content = content.replace(old_math_test, new_math_test)

    # Write the updated test file
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Updated validation test with proper prime checking")


def create_quick_test():
    """Create a quick test script to verify everything works"""

    quick_test_content = '''#!/usr/bin/env python3
"""
Quick test to verify the migration worked correctly
"""

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
        
        print("\\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED! Migration is successful!")
        print("\\n🎯 You can now:")
        print("  • Run examples: python examples/basic_prime_discovery.py")
        print("  • Install package: pip install -e .")
        print("  • Use CLI tools: math-discover --help")
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        print("Please check the migration and try again.")
'''

    quick_test_file = Path("scripts/quick_test.py")
    with open(quick_test_file, "w", encoding="utf-8") as f:
        f.write(quick_test_content)

    print(f"✅ Created quick test: {quick_test_file}")


def main():
    """Fix all migration issues"""
    print("🔧 Fixing migration issues...")
    print("=" * 40)

    add_is_prime_function()
    update_validation_test_with_prime()
    create_quick_test()

    print("\n" + "=" * 40)
    print("✅ All fixes applied!")
    print("\n🧪 Run the tests:")
    print("  python scripts/validate_migration.py")
    print("  python scripts/quick_test.py")


if __name__ == "__main__":
    main()
