"""Individual employee profile endpoints — T-061."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import UserRole
from api.schemas.enrollment import ProfileResponse, ScoreSnapshot, TeamProfileSummary
from ingestion.db.models import EmployeeProfile
from ingestion.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

TREND_WINDOW = 10  # snapshots kept in score_trend


def _risk_trajectory(trend: list[dict]) -> str:  # type: ignore[type-arg]
    if len(trend) < 3:
        return "insufficient_data"
    recent = [s["score"] for s in trend[-3:]]
    delta = recent[-1] - recent[0]
    if delta > 0.05:
        return "worsening"
    if delta < -0.05:
        return "improving"
    return "stable"


def _build_profile_response(prof: EmployeeProfile) -> ProfileResponse:
    raw_trend = prof.score_trend or []
    snapshots = [
        ScoreSnapshot(
            run_id=UUID(s["run_id"]),
            score=s["score"],
            cascade_risk=s.get("cascade_risk", 0.0),
            ts=s["ts"],
        )
        for s in raw_trend
    ]
    return ProfileResponse(
        pseudo_id=prof.pseudo_id,
        team_id=prof.team_id,
        latest_score=prof.latest_score,
        avg_score_30d=prof.avg_score_30d,
        score_trend=snapshots,
        top_features=prof.top_features or {},
        cascade_exposure_count=prof.cascade_exposure_count or 0,
        run_count=prof.run_count or 0,
        risk_trajectory=_risk_trajectory(raw_trend),
        seed_data=prof.seed_data,
        updated_at=prof.updated_at,
    )


@router.get("/profiles/{pseudo_id}", response_model=ProfileResponse)
async def get_profile(
    pseudo_id: UUID,
    token: TokenPayload = require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST),
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Return self-developing profile for a single pseudonymised employee.

    Access: HR Admin, HR Analyst.
    """
    result = await db.execute(select(EmployeeProfile).where(EmployeeProfile.pseudo_id == pseudo_id))
    prof = result.scalar_one_or_none()
    if prof is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    return _build_profile_response(prof)


@router.get("/profiles/team/{team_id}", response_model=TeamProfileSummary)
async def get_team_profile(
    team_id: str,
    token: TokenPayload = require_role(UserRole.MANAGER, UserRole.HR_ADMIN, UserRole.HR_ANALYST),
    db: AsyncSession = Depends(get_db),
) -> TeamProfileSummary:
    """Return aggregated team-level profile summary.

    Access: Manager, HR Admin, HR Analyst.
    """
    result = await db.execute(select(EmployeeProfile).where(EmployeeProfile.team_id == team_id))
    profiles = result.scalars().all()
    if not profiles:
        raise HTTPException(status_code=404, detail=f"No profiles found for team {team_id}")

    scored = [p for p in profiles if p.latest_score is not None]
    avg_score = sum(p.latest_score for p in scored) / len(scored) if scored else 0.0  # type: ignore[arg-type]
    high_risk = sum(1 for p in scored if (p.latest_score or 0) >= 0.70)
    avg_cascade = sum(p.cascade_exposure_count or 0 for p in profiles) / len(profiles)

    all_features: dict[str, float] = {}
    for p in profiles:
        for feat, val in (p.top_features or {}).items():
            all_features[feat] = all_features.get(feat, 0.0) + float(val)
    top_signals = sorted(all_features, key=all_features.get, reverse=True)[:5]  # type: ignore[arg-type]

    return TeamProfileSummary(
        team_id=team_id,
        member_count=len(profiles),
        avg_score=avg_score,
        high_risk_count=high_risk,
        avg_cascade_exposure=avg_cascade,
        top_signals=top_signals,
    )


@router.get("/profiles", response_model=list[ProfileResponse])
async def list_profiles(
    team_id: str | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0.0, le=1.0),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    token: TokenPayload = require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST),
    db: AsyncSession = Depends(get_db),
) -> list[ProfileResponse]:
    """List profiles with optional team or risk-score filter.

    Access: HR Admin, HR Analyst.
    """
    query = select(EmployeeProfile)
    if team_id:
        query = query.where(EmployeeProfile.team_id == team_id)
    if min_score is not None:
        query = query.where(EmployeeProfile.latest_score >= min_score)
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    return [_build_profile_response(p) for p in result.scalars().all()]
