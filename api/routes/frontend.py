"""Production dashboard static file serving."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

logger = logging.getLogger(__name__)

router = APIRouter()

DASHBOARD_DIR = Path("output/dashboard")

if not DASHBOARD_DIR.exists():
    logger.warning("output/dashboard/ not found — dashboard static files unavailable")


@router.get("/dashboard")
async def dashboard_redirect() -> RedirectResponse:
    return RedirectResponse(url="/dashboard/")
