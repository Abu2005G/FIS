"""
Financial Intelligence & Risk Signal Engine: application entry point.

Run from the repository root:
    uvicorn src.app.main:app
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.app.config import get_settings
from src.app.database import init_db
from src.app.routes import get_pipeline, public_router, router

logger = logging.getLogger("src.app")

API_VERSION = "2.0.0"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # fails fast on bad configuration
    configure_logging(settings.log_level)
    init_db()
    pipeline = get_pipeline()  # fails fast if the LLM client cannot be built
    logger.info(
        "Started: environment=%s llm_provider=%s llm_model=%s",
        settings.environment,
        settings.llm_provider,
        pipeline.client.model if pipeline else None,
    )
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    production = settings.environment == "production"

    application = FastAPI(
        title="Financial Intelligence & Risk Signal Engine",
        description="Credit risk assessment from bank statements.",
        version=API_VERSION,
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Never leak internals to the caller; the log holds the detail.
            logger.exception(
                "Unhandled error request_id=%s %s %s",
                request_id,
                request.method,
                request.url.path,
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s %s %s -> %s in %.0fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    application.include_router(public_router)
    application.include_router(router)

    @application.get("/", include_in_schema=False)
    def root() -> dict:
        return {"service": "financial-intelligence-engine", "version": API_VERSION}

    return application


app = create_app()
