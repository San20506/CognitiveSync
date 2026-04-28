"""FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import (
    audit,
    cascade,
    config,
    demo,
    employees,
    model,
    pipeline,
    profiles,
    recommendations,
    scores,
)
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

# Internal-only CORS — no external origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT"],
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
app.include_router(demo.router, tags=["demo"])
