import os
from pathlib import Path

from app.core.config import AppConfig


def test_config_defaults_resolve_paths():
    config = AppConfig(base_dir=Path("D:/tmp/secchi"))

    assert config.model_path == Path("D:/tmp/secchi/app/models/secchi_disk_turbidity_model.pt")
    assert config.upload_incoming_dir == Path("D:/tmp/secchi/uploads/incoming")


def test_upload_directories_created(tmp_path):
    config = AppConfig(base_dir=tmp_path)

    dirs = config.ensure_upload_directories()

    assert dirs["incoming"].exists()
    assert dirs["processed"].exists()
    assert dirs["failed"].exists()


def test_normalization_params_round_trip(tmp_path):
    config = AppConfig(base_dir=tmp_path)
    payload = {"edge_clarity_canny": 0.12, "contrast_std": 50}

    saved_path = config.save_normalization_parameters(payload)
    loaded = config.load_normalization_parameters()

    assert saved_path.exists()
    assert loaded == payload


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("SECCHI_BASE_DIR", "D:/data/secchi")
    monkeypatch.setenv("SECCHI_MODEL_PATH", "models/custom.pt")
    monkeypatch.setenv("SECCHI_DEFAULT_STANDARD", "epa")
    monkeypatch.setenv("SECCHI_DEFAULT_WEIGHTING_METHOD", "physics")
    monkeypatch.setenv("SECCHI_DEFAULT_ADAPTIVE_SCORING", "true")

    config = AppConfig.from_env()

    assert config.base_dir == Path("D:/data/secchi")
    assert config.model_path == Path("D:/data/secchi/models/custom.pt")
    assert config.default_standard == "epa"
    assert config.default_weighting_method == "physics"
    assert config.default_adaptive_scoring is True

    monkeypatch.delenv("SECCHI_BASE_DIR", raising=False)
    monkeypatch.delenv("SECCHI_MODEL_PATH", raising=False)
    monkeypatch.delenv("SECCHI_DEFAULT_STANDARD", raising=False)
    monkeypatch.delenv("SECCHI_DEFAULT_WEIGHTING_METHOD", raising=False)
    monkeypatch.delenv("SECCHI_DEFAULT_ADAPTIVE_SCORING", raising=False)


def test_config_from_explicit_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SECCHI_BASE_DIR=D:/custom/secchi",
                "SECCHI_MODEL_PATH=models/model.pt",
                "SECCHI_DEFAULT_STANDARD=marine",
                "SECCHI_DEFAULT_WEIGHTING_METHOD=edge_focused",
                "SECCHI_DEFAULT_DETECTION_CONFIDENCE=0.2",
                "SECCHI_DEFAULT_ADAPTIVE_SCORING=true",
                "SECCHI_UPLOAD_ROOT=uploads_data",
                "SECCHI_UPLOAD_INCOMING_SUBDIR=in",
                "SECCHI_UPLOAD_PROCESSED_SUBDIR=done",
                "SECCHI_UPLOAD_FAILED_SUBDIR=bad",
                "SECCHI_NORMALIZATION_PARAMS_PATH=config/norm.json",
            ]
        ),
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_file=env_file)

    assert config.base_dir == Path("D:/custom/secchi")
    assert config.model_path == Path("D:/custom/secchi/models/model.pt")
    assert config.default_standard == "marine"
    assert config.default_weighting_method == "edge_focused"
    assert config.default_detection_confidence == 0.2
    assert config.default_adaptive_scoring is True
    assert config.upload_root == Path("D:/custom/secchi/uploads_data")
    assert config.upload_incoming_dir == Path("D:/custom/secchi/uploads_data/in")
    assert config.upload_processed_dir == Path("D:/custom/secchi/uploads_data/done")
    assert config.upload_failed_dir == Path("D:/custom/secchi/uploads_data/bad")
    assert config.normalization_params_path == Path("D:/custom/secchi/config/norm.json")
