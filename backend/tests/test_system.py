from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ultralytics")

from app.services.system import SecchiTurbiditySystem


class DummyTensor:
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class DummyBox:
    def __init__(self):
        self.conf = [0.9]
        self.xyxy = [DummyTensor([10, 10, 40, 40])]


class DummyYOLO:
    def __init__(self, _model_path):
        pass

    def predict(self, source, conf=0.15, verbose=False):
        _ = (source, conf, verbose)
        return [SimpleNamespace(boxes=[DummyBox()])]


def test_assess_single_image_smoke(monkeypatch):
    monkeypatch.setattr("app.services.system.YOLO", DummyYOLO)
    monkeypatch.setattr(
        "app.services.system.cv2.imread",
        lambda _path: np.full((64, 64, 3), 128, dtype=np.uint8),
    )

    system = SecchiTurbiditySystem("dummy.pt", standard="auto", weighting_method="balanced")
    result = system.assess_single_image("fake.jpg", visualize=False, verbose=False)

    assert result["disk_detected"] is True
    assert "turbidity_category" in result
    assert "visibility_score" in result
