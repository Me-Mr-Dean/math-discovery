from src.core.pattern_analyzer import PatternAnalyzer


def test_pattern_detection():
    analyzer = PatternAnalyzer()
    assert analyzer.is_arithmetic([1, 3, 5, 7])
    assert analyzer.common_difference([1, 3, 5, 7]) == 2
    assert analyzer.is_geometric([2, 4, 8, 16])
    assert analyzer.common_ratio([2, 4, 8, 16]) == 2
    assert not analyzer.is_arithmetic([1, 2, 4])
