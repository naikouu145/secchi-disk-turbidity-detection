import numpy as np

from app.services.score_calculator import VisibilityScoreCalculator


def test_weights_are_normalized_when_custom_sum_is_not_one():
    calc = VisibilityScoreCalculator(weights={"a": 2.0, "b": 1.0}, method="custom")

    assert np.isclose(sum(calc.weights.values()), 1.0)


def test_calculate_score_is_clipped_to_unit_interval():
    calc = VisibilityScoreCalculator(method="balanced")
    features = {name: 10.0 for name in calc.weights.keys()}

    score = calc.calculate_score(features)

    assert 0.0 <= score <= 1.0
    assert score == 1.0


def test_compare_weighting_methods_returns_all_modes():
    calc = VisibilityScoreCalculator(method="balanced")
    features = {name: 0.5 for name in calc.weights.keys()}

    result = calc.compare_weighting_methods(features)

    assert {"physics", "balanced", "edge_focused", "current"}.issubset(result.keys())
