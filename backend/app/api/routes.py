from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.core.config import AppConfig
from app.services.system import SecchiTurbiditySystem

router = APIRouter()


class ConfigUpdateRequest(BaseModel):
    model_relative_path: str | None = None
    default_standard: str | None = None
    default_weighting_method: str | None = None
    default_detection_confidence: float | None = None
    default_adaptive_scoring: bool | None = None
    upload_root_relative: str | None = None
    upload_incoming_subdir: str | None = None
    upload_processed_subdir: str | None = None
    upload_failed_subdir: str | None = None
    normalization_file_relative: str | None = None


def _structured_response(message: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "success",
        "message": message,
        "data": data,
    }


def _get_system_from_state(request: Request) -> SecchiTurbiditySystem:
    system = getattr(request.app.state, "system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="Assessment system is not initialized")
    return system


def _config_response(config: AppConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["base_dir"] = str(config.base_dir)
    payload["model_path"] = str(config.model_path)
    payload["normalization_params_path"] = str(config.normalization_params_path)
    payload["upload_root"] = str(config.upload_root)
    payload["upload_incoming_dir"] = str(config.upload_incoming_dir)
    payload["upload_processed_dir"] = str(config.upload_processed_dir)
    payload["upload_failed_dir"] = str(config.upload_failed_dir)
    return payload


async def _save_upload(file: UploadFile, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(file.filename or "").suffix
    filename = f"{uuid4().hex}{extension}"
    output_path = target_dir / filename

    content = await file.read()
    output_path.write_bytes(content)
    return output_path


@router.post("/assess")
async def assess_single(
    request: Request,
    file: UploadFile = File(...),
    adaptive_scoring: bool = False,
    override_source: str | None = None,
):
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Missing file name")

    system = _get_system_from_state(request)
    incoming_dir = request.app.state.config.upload_incoming_dir

    image_path = await _save_upload(file, incoming_dir)
    result = system.assess_single_image(
        image_path=str(image_path),
        visualize=False,
        verbose=False,
        adaptive_scoring=adaptive_scoring,
        override_source=override_source,
    )
    return _structured_response(
        message="Single image assessment completed",
        data={
            "filename": file.filename,
            "assessment": result,
        },
    )


@router.post("/assess/batch")
async def assess_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    adaptive_scoring: bool = False,
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    system = _get_system_from_state(request)
    incoming_dir = request.app.state.config.upload_incoming_dir

    image_paths: list[str] = []
    for file in files:
        saved = await _save_upload(file, incoming_dir)
        image_paths.append(str(saved))

    batch_result = system.assess_batch(
        image_paths=image_paths,
        save_results=False,
        adaptive_scoring=adaptive_scoring,
        show_progress=False,
    )

    if hasattr(batch_result, "to_dict"):
        records = batch_result.to_dict(orient="records")
    else:
        records = batch_result

    return _structured_response(
        message="Batch assessment completed",
        data={
            "count": len(records),
            "results": records,
        },
    )


@router.get("/config")
async def get_config(request: Request):
    config: AppConfig = request.app.state.config
    return _config_response(config)


@router.post("/config")
async def update_config(request: Request, payload: ConfigUpdateRequest):
    current_config: AppConfig = request.app.state.config
    update_data = payload.model_dump(exclude_none=True)

    if not update_data:
        return {
            "message": "No changes applied",
            "config": _config_response(current_config),
        }

    new_config = AppConfig(
        base_dir=current_config.base_dir,
        model_relative_path=update_data.get(
            "model_relative_path", current_config.model_relative_path
        ),
        default_standard=update_data.get(
            "default_standard", current_config.default_standard
        ),
        default_weighting_method=update_data.get(
            "default_weighting_method", current_config.default_weighting_method
        ),
        default_detection_confidence=update_data.get(
            "default_detection_confidence", current_config.default_detection_confidence
        ),
        default_adaptive_scoring=update_data.get(
            "default_adaptive_scoring", current_config.default_adaptive_scoring
        ),
        api_prefix=current_config.api_prefix,
        cors_allow_origins_raw=current_config.cors_allow_origins_raw,
        upload_root_relative=update_data.get(
            "upload_root_relative", current_config.upload_root_relative
        ),
        upload_incoming_subdir=update_data.get(
            "upload_incoming_subdir", current_config.upload_incoming_subdir
        ),
        upload_processed_subdir=update_data.get(
            "upload_processed_subdir", current_config.upload_processed_subdir
        ),
        upload_failed_subdir=update_data.get(
            "upload_failed_subdir", current_config.upload_failed_subdir
        ),
        normalization_file_relative=update_data.get(
            "normalization_file_relative", current_config.normalization_file_relative
        ),
    )

    old_system = request.app.state.system

    try:
        new_system = SecchiTurbiditySystem(config=new_config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to apply config: {exc}")

    request.app.state.config = new_config
    request.app.state.system = new_system

    if hasattr(old_system, "close") and callable(old_system.close):
        old_system.close()

    return {
        "message": "Configuration updated",
        "config": _config_response(new_config),
    }


@router.get("/health")
async def health_check(request: Request):
    has_system = getattr(request.app.state, "system", None) is not None
    return {"status": "ok", "system_initialized": has_system}
