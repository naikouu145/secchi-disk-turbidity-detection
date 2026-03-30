from pathlib import Path

from fastapi.testclient import TestClient


class DummySystem:
    def __init__(self, config=None):
        self.config = config
        self.closed = False

    def assess_single_image(
        self,
        image_path,
        visualize=False,
        verbose=False,
        adaptive_scoring=False,
        override_source=None,
    ):
        _ = (visualize, verbose, adaptive_scoring, override_source)
        return {
            "disk_detected": True,
            "image_path": image_path,
            "turbidity_category": "Clear Water",
            "visibility_score": 0.9,
        }

    def assess_batch(
        self,
        image_paths,
        save_results=False,
        adaptive_scoring=False,
        show_progress=False,
    ):
        _ = (save_results, adaptive_scoring, show_progress)
        return [
            {
                "image_path": Path(path).name,
                "disk_detected": True,
                "turbidity_category": "Clear Water",
                "visibility_score": 0.9,
            }
            for path in image_paths
        ]

    def close(self):
        self.closed = True


def test_requested_routes_exist_and_work(monkeypatch):
    import app.api.routes as routes_module
    import app.main as main_module

    monkeypatch.setattr(main_module, "SecchiTurbiditySystem", DummySystem)
    monkeypatch.setattr(routes_module, "SecchiTurbiditySystem", DummySystem)

    app = main_module.create_app()

    with TestClient(app) as client:
        prefix = app.state.config.normalized_api_prefix

        health = client.get(f"{prefix}/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        cfg = client.get(f"{prefix}/config")
        assert cfg.status_code == 200
        assert "model_path" in cfg.json()

        single = client.post(
            f"{prefix}/assess",
            files={"file": ("single.jpg", b"fake-bytes", "image/jpeg")},
        )
        assert single.status_code == 200
        assert single.json()["status"] == "success"
        assert single.json()["data"]["assessment"]["disk_detected"] is True

        batch = client.post(
            f"{prefix}/assess/batch",
            files=[
                ("files", ("a.jpg", b"x", "image/jpeg")),
                ("files", ("b.jpg", b"y", "image/jpeg")),
            ],
        )
        assert batch.status_code == 200
        assert batch.json()["status"] == "success"
        assert batch.json()["data"]["count"] == 2

        updated = client.post(
            f"{prefix}/config",
            json={
                "default_standard": "epa",
                "default_weighting_method": "physics",
            },
        )
        assert updated.status_code == 200
        assert updated.json()["message"] == "Configuration updated"
