"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from idengine import __version__
from idengine.api.errors import install_error_handlers
from idengine.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from idengine.api.routes import documents, intelligence, system
from idengine.api.security import Authenticator
from idengine.config import Settings, load_settings
from idengine.engine import Engine
from idengine.errors import ConfigurationError
from idengine.observability.logging import configure_logging, get_logger

log = get_logger("api")


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    start_workers: bool = True,
) -> FastAPI:
    """Build the API.

    Pass ``engine`` to reuse an existing engine (tests); otherwise one is
    created on start-up from ``settings`` (or the environment) and closed on
    shutdown.
    """
    settings = settings or (engine.settings if engine else load_settings())
    if settings.api.auth_enabled and not settings.api.api_keys:
        raise ConfigurationError(
            "API authentication is enabled but no keys are configured: set IDE_API_KEYS "
            "(e.g. 'mykey123:admin') or IDE_AUTH_ENABLED=false for local experiments"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = engine is None
        active = engine or Engine(settings)
        app.state.engine = active
        if start_workers:
            active.start_workers()
        log.info(
            "api started",
            extra={
                "version": __version__,
                "workers": settings.jobs.workers if start_workers else 0,
            },
        )
        try:
            yield
        finally:
            if owned:
                active.close()
            else:
                active.jobs.stop()
            log.info("api stopped")

    app = FastAPI(
        title="Intelligent Document Engine",
        version=__version__,
        description="Local-first document ingestion, search, RAG and classification.",
        lifespan=lifespan,
    )
    app.state.authenticator = Authenticator(settings.api)
    if engine is not None:
        app.state.engine = engine
    install_error_handlers(app)
    app.include_router(documents.router)
    app.include_router(intelligence.router)
    app.include_router(system.router)
    # Added last = outermost: request context wraps the size limiter.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.api.max_request_bytes)
    metrics = engine.metrics if engine is not None else None
    app.add_middleware(RequestContextMiddleware, metrics=metrics or _LazyMetrics(app))
    return app


class _LazyMetrics:
    """Resolves the engine's metrics once the lifespan has created the engine."""

    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def __getattr__(self, name: str) -> object:
        return getattr(self._app.state.engine.metrics, name)


def build_app_from_env() -> FastAPI:
    """Entry point for ``uvicorn idengine.api.app:build_app_from_env --factory``."""
    settings = load_settings()
    configure_logging(settings.logging.level, json_format=settings.logging.json_format)
    return create_app(settings)
