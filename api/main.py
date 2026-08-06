"""FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.routes import (
    audit,
    cascade,
    chat,
    config,
    demo,
    employees,
    explain,
    frontend,
    model,
    pipeline,
    profiles,
    recommendations,
    scores,
)
from config.settings import settings
from ingestion.scheduler import configure_scheduler, scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start scheduler on startup; shut it down gracefully on shutdown."""
    configure_scheduler()
    scheduler.start()
    logger.info("Scheduler started")
    yield
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")


app = FastAPI(
    title="CognitiveSync API",
    version="0.1.0",
    description="Privacy-first burnout prediction platform — internal API.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Internal-only CORS — no external origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(scores.router, prefix="/api/v1", tags=["scores"])
app.include_router(cascade.router, prefix="/api/v1", tags=["cascade"])
app.include_router(pipeline.router, prefix="/api/v1", tags=["pipeline"])
app.include_router(employees.router, prefix="/api/v1", tags=["enrollment"])
app.include_router(profiles.router, prefix="/api/v1", tags=["profiles"])
app.include_router(recommendations.router, prefix="/api/v1", tags=["recommendations"])
app.include_router(config.router, prefix="/api/v1", tags=["config"])
app.include_router(audit.router, prefix="/api/v1", tags=["audit"])
app.include_router(model.router, prefix="/api/v1", tags=["model"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(explain.router, prefix="/api/v1", tags=["explainability"])

if settings.demo_enabled:
    app.include_router(demo.router, tags=["demo"])

app.include_router(frontend.router, tags=["dashboard"])

_dashboard_dir = Path("output/dashboard")
if _dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard")
