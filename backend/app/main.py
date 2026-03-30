from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as api_router
from app.core.config import AppConfig
from app.services.system import SecchiTurbiditySystem


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up shared application resources."""
    config = getattr(app.state, "config", None)
    if config is None:
        config = AppConfig.from_env()
    app.state.config = config

    system = SecchiTurbiditySystem(config=config)
    app.state.system = system

    try:
        yield
    finally:
        # Keep shutdown cleanup optional and safe for future resource hooks.
        if hasattr(system, "close") and callable(system.close):
            system.close()
        app.state.system = None


def create_app() -> FastAPI:
    config = AppConfig.from_env()

    app = FastAPI(
        title="Secchi Disk Turbidity API",
        lifespan=lifespan,
    )

    app.state.config = config

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=config.normalized_api_prefix)

    return app


app = create_app()
