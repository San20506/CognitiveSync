"""Demo-mode endpoints — token minting and dashboard serving.

Only active when ADAPTER_MODE=mock. Never expose in production.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from jose import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.limiter import limiter
from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import UserRole
from config.settings import settings
from ingestion.db.models import BurnoutScore, Employee
from ingestion.db.session import get_db

router = APIRouter()

_DASHBOARD_HTML = Path(__file__).parent.parent.parent / "output" / "dashboard.html"
_INGEST_HTML = Path(__file__).parent.parent.parent / "output" / "ingest.html"
_THRESHOLDS = {"critical": 0.75, "high": 0.65, "medium": 0.45, "low": 0.0}


def _latest_run_query() -> select:
    return select(BurnoutScore.run_id).order_by(BurnoutScore.window_end.desc()).limit(1)


def _sorted_top_signals(top_features: dict | None, limit: int = 3) -> list[dict[str, float | str]]:
    items: Iterable[tuple[str, object]] = (top_features or {}).items()
    ranked = sorted(
        (
            (feature, float(weight))
            for feature, weight in items
            if isinstance(weight, int | float)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": feature, "weight": round(weight, 2)}
        for feature, weight in ranked[:limit]
    ]


def _build_recommendations(score: BurnoutScore, seniority: str | None) -> list[str]:
    top_features = score.top_features or {}
    recommendations: list[str] = []

    if "after_hours_commits" in top_features or "after_hours_messages" in top_features:
        recommendations.append("Reduce after-hours Slack notifications")
    if "slack_response_latency" in top_features:
        recommendations.append("Audit Slack response expectations for this team")
    if "meeting_overload" in top_features or "meeting_density" in top_features:
        recommendations.append("Reduce recurring meeting load for the next sprint")
    if seniority == "senior":
        recommendations.append("Redistribute PR review load")

    if not recommendations:
        recommendations.extend(
            [
                "Check manager workload balance in the next 1:1",
                "Review focus-time fragmentation for this employee",
            ]
        )

    return recommendations[:2]


@router.get("/demo/token")
@limiter.limit("10/minute")
async def mint_demo_token(request: Request, role: str = "hr_admin") -> dict[str, str]:
    """Mint a short-lived demo JWT. Only works when DEMO_ENABLED=true.

    No authentication required — this is the entry point for demo access.
    """
    if not settings.demo_enabled:
        raise HTTPException(status_code=404, detail="Demo mode is not enabled.")
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


@router.get("/demo/thresholds")
async def get_demo_thresholds(
    _: TokenPayload = require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST),
) -> dict[str, float]:
    """Get burnout score thresholds. Requires HR_ADMIN or HR_ANALYST role."""
    return _THRESHOLDS


@router.get("/demo/employee/{pseudo_id}")
async def get_demo_employee(
    pseudo_id: uuid.UUID,
    _: TokenPayload = require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST),
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Get demo employee burnout data. Requires HR_ADMIN or HR_ANALYST role."""
    latest_run = await db.execute(_latest_run_query())
    run_id = latest_run.scalar_one_or_none()
    if run_id is None:
        raise HTTPException(status_code=404, detail="No scoring runs available.")

    score_result = await db.execute(
        select(BurnoutScore)
        .where(BurnoutScore.run_id == run_id, BurnoutScore.pseudo_id == pseudo_id)
        .limit(1)
    )
    score = score_result.scalar_one_or_none()
    if score is None:
        raise HTTPException(status_code=404, detail="Employee score not found.")

    employee_result = await db.execute(
        select(Employee).where(Employee.pseudo_id == pseudo_id).limit(1)
    )
    employee = employee_result.scalar_one_or_none()

    trend_result = await db.execute(
        select(BurnoutScore)
        .where(BurnoutScore.pseudo_id == pseudo_id)
        .order_by(BurnoutScore.window_end.desc())
        .limit(5)
    )
    trend_scores = list(reversed(trend_result.scalars().all()))

    team_id = (
        str(score.team_id)
        if score.team_id is not None
        else (employee.team_id if employee else None)
    )
    return {
        "pseudo_id": str(score.pseudo_id),
        "burnout_score": score.burnout_score,
        "confidence_low": score.confidence_low or max(score.burnout_score - 0.08, 0.0),
        "confidence_high": score.confidence_high or min(score.burnout_score + 0.08, 1.0),
        "team_id": team_id,
        "seniority": employee.seniority if employee and employee.seniority else "unknown",
        "top_signals": _sorted_top_signals(score.top_features),
        "score_trend": [round(item.burnout_score, 2) for item in trend_scores],
        "recommendations": _build_recommendations(score, employee.seniority if employee else None),
    }


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    _: TokenPayload = require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST),
) -> HTMLResponse:
    """Serve the CognitiveSync POC dashboard. Requires HR_ADMIN or HR_ANALYST role."""
    if not _DASHBOARD_HTML.exists():
        raise HTTPException(status_code=404, detail="Dashboard not built.")
    return HTMLResponse(_DASHBOARD_HTML.read_text())


@router.get("/ingest", response_class=HTMLResponse)
async def ingest_page(
    _: TokenPayload = require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST),
) -> HTMLResponse:
    """Serve the demo drag-and-drop ingestion page. Requires HR_ADMIN or HR_ANALYST role."""
    if not _INGEST_HTML.exists():
        raise HTTPException(status_code=404, detail="Ingest page not built.")
    return HTMLResponse(_INGEST_HTML.read_text())
