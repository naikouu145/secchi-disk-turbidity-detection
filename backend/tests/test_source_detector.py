import numpy as np

from app.services.source_detector import TurbiditySourceDetector


def test_default_source_structure():
    detector = TurbiditySourceDetector()

    result = detector._default_source()

    assert result["primary_source"] == "unknown"
    assert "confidence" in result
    assert "algal_score" in result
    assert "sediment_score" in result


def test_detect_source_returns_valid_label():
    detector = TurbiditySourceDetector()

    image = np.full((80, 80, 3), 140, dtype=np.uint8)
    bbox = [20, 20, 60, 60]
    features = {"contrast_range": 90}

    result = detector.detect_source(image, bbox, features)

    assert result["primary_source"] in {"algal", "sediment", "mixed"}
    assert 0.0 <= result["confidence"] <= 1.0
