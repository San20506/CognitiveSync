"""Demo-mode endpoints — token minting and dashboard serving.

Only active when ADAPTER_MODE=mock. Never expose in production.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from jose import jwt

from config.settings import settings

router = APIRouter()

_DASHBOARD_HTML = Path(__file__).parent.parent.parent / "output" / "dashboard.html"


@router.get("/demo/token")
async def mint_demo_token(role: str = "hr_admin") -> dict[str, str]:
    """Mint a short-lived demo JWT. Only works in mock/demo mode."""
    if settings.adapter_mode != "mock":
        raise HTTPException(status_code=403, detail="Only available in mock mode.")

    valid_roles = {"it_admin", "hr_admin", "hr_analyst", "manager"}
    if role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Role must be one of {valid_roles}")

    now = int(time.time())
    payload = {
        "sub": str(uuid.uuid4()),
        "role": role,
        "org_id": str(uuid.uuid4()),
        "exp": now + 480 * 60,
        "iat": now,
    }
    token = jwt.encode(payload, settings.jwt_private_key, algorithm=settings.jwt_algorithm)
    return {"token": token, "role": role}


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the CognitiveSync POC dashboard."""
    if not _DASHBOARD_HTML.exists():
        raise HTTPException(status_code=404, detail="Dashboard not built.")
    return HTMLResponse(_DASHBOARD_HTML.read_text())
