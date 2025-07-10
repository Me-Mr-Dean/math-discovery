import numpy as np
from src.utils.math_utils import (
    generate_mathematical_features,
    distance_to_next,
    distance_to_prev,
    fit_polynomial_features,
)


def test_diff_ratio_sliding_and_digit_tensor():
    prev = [1, 2, 3, 4]
    features = generate_mathematical_features(5, previous_numbers=prev, window_size=3, digit_tensor=True)
    assert features["diff_n"] == 1
    assert np.isclose(features["ratio_n"], 5/4)
    assert np.isclose(features["mean_last_3"], np.mean([2, 3, 4]))
    assert np.isclose(features["std_last_3"], np.std([2, 3, 4]))
    assert "digit_tensor" in features
    assert isinstance(features["digit_tensor"], list)


def test_defaults_without_history():
    features = generate_mathematical_features(10)
    assert "diff_n" in features and features["diff_n"] == 0
    assert "ratio_n" in features and features["ratio_n"] == 0
    assert "mean_last_5" in features
    assert "std_last_5" in features


def test_distance_helpers_and_polyfit():
    num_set = {2, 5, 10}
    assert distance_to_next(4, num_set) == 1
    assert distance_to_prev(4, num_set) == 2
    assert distance_to_next(10, num_set) == 0
    assert distance_to_prev(2, num_set) == 0

    residuals = fit_polynomial_features(1234, degree=1)
    assert isinstance(residuals, list)
    assert len(residuals) == 4

    features = generate_mathematical_features(
        6, reference_set=num_set, poly_degree=2
    )
    assert features["dist_to_next"] == 4
    assert features["dist_to_prev"] == 1
    assert "poly_deg_2_residuals" in features
