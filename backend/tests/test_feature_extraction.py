import numpy as np

from app.services.feature_extraction import SecchiDiskFeatureExtractor


def test_extract_features_invalid_bbox_returns_defaults():
    extractor = SecchiDiskFeatureExtractor()
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    features = extractor.extract_features(image, [20, 20, 10, 10], 0.9)

    assert set(features.keys()) == set(extractor.feature_names)
    assert all(value == 0.0 for value in features.values())


def test_normalize_features_handles_non_finite_values():
    extractor = SecchiDiskFeatureExtractor()
    features = {"edge_clarity_canny": np.inf, "contrast_std": 30.0}

    normalized = extractor.normalize_features(features)

    assert normalized["edge_clarity_canny"] == 0.0
    assert 0.0 <= normalized["contrast_std"] <= 1.0
