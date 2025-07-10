from src.generators.sequence_generator import SequenceGenerator


def test_sequence_generator():
    gen = SequenceGenerator()
    assert gen.arithmetic_sequence(1, 2, 4) == [1, 3, 5, 7]
    assert gen.geometric_sequence(2, 2, 4) == [2, 4, 8, 16]
    assert gen.fibonacci(5) == [1, 1, 2, 3, 5]
