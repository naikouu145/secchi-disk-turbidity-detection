from fastapi.testclient import TestClient


class DummySystem:
    def __init__(self, config=None):
        self.config = config
        self.closed = False

    def close(self):
        self.closed = True


def test_lifespan_initializes_system_and_cleans_up(monkeypatch):
    import app.main as main_module

    created_systems = []

    def build_dummy_system(config=None):
        system = DummySystem(config=config)
        created_systems.append(system)
        return system

    monkeypatch.setattr(main_module, "SecchiTurbiditySystem", build_dummy_system)

    app = main_module.create_app()

    with TestClient(app) as client:
        response = client.get(f"{app.state.config.normalized_api_prefix}/health")
        assert response.status_code == 200
        assert response.json()["system_initialized"] is True
        assert getattr(app.state, "system", None) is not None

    assert len(created_systems) == 1
    assert created_systems[0].closed is True
    assert getattr(app.state, "system", "missing") is None
