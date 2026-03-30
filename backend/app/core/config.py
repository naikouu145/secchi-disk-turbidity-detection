import json
from dataclasses import dataclass, field
from pathlib import Path

from decouple import AutoConfig, Config, RepositoryEnv


@dataclass
class AppConfig:
    """Centralized configuration for Secchi turbidity backend services."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    model_relative_path: str = "app/models/secchi_disk_turbidity_model.pt"

    default_standard: str = "auto"
    default_weighting_method: str = "balanced"
    default_detection_confidence: float = 0.15
    default_adaptive_scoring: bool = False

    api_prefix: str = "/api"
    cors_allow_origins_raw: str = "http://localhost:5173,http://127.0.0.1:5173"

    upload_root_relative: str = "uploads"
    upload_incoming_subdir: str = "incoming"
    upload_processed_subdir: str = "processed"
    upload_failed_subdir: str = "failed"

    normalization_file_relative: str = "app/models/normalization_params.json"

    @property
    def normalized_api_prefix(self) -> str:
        cleaned = f"/{str(self.api_prefix or '').strip('/')}"
        return cleaned if cleaned != "/" else ""

    @property
    def cors_allow_origins(self) -> list[str]:
        origins = [item.strip() for item in self.cors_allow_origins_raw.split(",") if item.strip()]
        if not origins:
            return ["http://localhost:5173", "http://127.0.0.1:5173"]
        if "*" in origins:
            return ["*"]
        return origins

    @property
    def model_path(self) -> Path:
        return self.base_dir / self.model_relative_path

    @property
    def normalization_params_path(self) -> Path:
        return self.base_dir / self.normalization_file_relative

    @property
    def upload_root(self) -> Path:
        return self.base_dir / self.upload_root_relative

    @property
    def upload_incoming_dir(self) -> Path:
        return self.upload_root / self.upload_incoming_subdir

    @property
    def upload_processed_dir(self) -> Path:
        return self.upload_root / self.upload_processed_subdir

    @property
    def upload_failed_dir(self) -> Path:
        return self.upload_root / self.upload_failed_subdir

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "AppConfig":
        """Build configuration using python-decouple with automatic .env loading."""
        base_dir_default = Path(__file__).resolve().parents[2]

        if env_file is not None:
            initial_config = Config(RepositoryEnv(str(env_file)))
        else:
            initial_config = AutoConfig(search_path=str(base_dir_default))

        base_dir = Path(
            initial_config("SECCHI_BASE_DIR", default=str(base_dir_default))
        )

        if env_file is not None:
            config_source = Config(RepositoryEnv(str(env_file)))
        else:
            config_source = AutoConfig(search_path=str(base_dir))

        return cls(
            base_dir=base_dir,
            model_relative_path=config_source(
                "SECCHI_MODEL_PATH", default="app/models/secchi_disk_turbidity_model.pt"
            ),
            default_standard=config_source("SECCHI_DEFAULT_STANDARD", default="auto"),
            default_weighting_method=config_source(
                "SECCHI_DEFAULT_WEIGHTING_METHOD", default="balanced"
            ),
            default_detection_confidence=config_source(
                "SECCHI_DEFAULT_DETECTION_CONFIDENCE", default=0.15, cast=float
            ),
            default_adaptive_scoring=config_source(
                "SECCHI_DEFAULT_ADAPTIVE_SCORING", default=False, cast=bool
            ),
            api_prefix=config_source("SECCHI_API_PREFIX", default="/api"),
            cors_allow_origins_raw=config_source(
                "SECCHI_CORS_ALLOW_ORIGINS",
                default="http://localhost:5173,http://127.0.0.1:5173",
            ),
            upload_root_relative=config_source("SECCHI_UPLOAD_ROOT", default="uploads"),
            upload_incoming_subdir=config_source(
                "SECCHI_UPLOAD_INCOMING_SUBDIR", default="incoming"
            ),
            upload_processed_subdir=config_source(
                "SECCHI_UPLOAD_PROCESSED_SUBDIR", default="processed"
            ),
            upload_failed_subdir=config_source(
                "SECCHI_UPLOAD_FAILED_SUBDIR", default="failed"
            ),
            normalization_file_relative=config_source(
                "SECCHI_NORMALIZATION_PARAMS_PATH",
                default="app/models/normalization_params.json",
            ),
        )

    def ensure_upload_directories(self) -> dict[str, Path]:
        """Create upload directories if missing and return all paths."""
        dirs = {
            "root": self.upload_root,
            "incoming": self.upload_incoming_dir,
            "processed": self.upload_processed_dir,
            "failed": self.upload_failed_dir,
        }

        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        return dirs

    def load_normalization_parameters(self) -> dict | None:
        """Load normalization parameters from JSON if available."""
        path = self.normalization_params_path
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def save_normalization_parameters(self, params: dict) -> Path:
        """Persist normalization parameters to JSON and return saved path."""
        path = self.normalization_params_path
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(params, file, indent=2)

        return path
