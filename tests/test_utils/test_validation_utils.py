from src.utils.validation_utils import (
    is_non_empty_numeric_sequence,
    has_unique_elements,
    is_increasing,
)


def test_validation_helpers():
    assert is_non_empty_numeric_sequence([1, 2, 3])
    assert not is_non_empty_numeric_sequence([])
    assert not is_non_empty_numeric_sequence([1, "a"])

    assert has_unique_elements([1, 2, 3])
    assert not has_unique_elements([1, 1, 2])

    assert is_increasing([1, 2, 3])
    assert not is_increasing([1, 1, 2])
