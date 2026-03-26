"""Burnout score endpoints — T-056."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import UserRole, score_to_risk_level
from api.schemas.response import BurnoutScoreResponse, TeamSummaryResponse
from ingestion.db.models import BurnoutScore
from ingestion.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/scores", response_model=list[BurnoutScoreResponse])
async def get_scores(
    run_id: UUID | None = Query(default=None),
    token: TokenPayload = require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST),
    db: AsyncSession = Depends(get_db),
) -> list[BurnoutScoreResponse]:
    """Return all burnout scores for the latest (or specified) scoring run.

    Access: HR Admin, HR Analyst only.
    """
    if run_id is None:
        result = await db.execute(
            select(BurnoutScore.run_id)
            .order_by(BurnoutScore.window_end.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return []
        run_id = row

    result = await db.execute(
        select(BurnoutScore).where(BurnoutScore.run_id == run_id)
    )
    scores = result.scalars().all()

    return [
        BurnoutScoreResponse.build(
            pseudo_id=s.pseudo_id,
            burnout_score=s.burnout_score,
            confidence_low=s.confidence_low or 0.0,
            confidence_high=s.confidence_high or 1.0,
            cascade_risk=s.cascade_risk or 0.0,
            cascade_sources=[
                UUID(x) for x in (s.cascade_sources or {}).get("sources", [])
            ],
            top_features=s.top_features or {},
            team_id=s.team_id,
            window_end=s.window_end,
            run_id=s.run_id,
        )
        for s in scores
    ]


@router.get("/scores/team-summary", response_model=list[TeamSummaryResponse])
async def get_team_summary(
    team_id: UUID | None = Query(default=None),
    token: TokenPayload = require_role(
        UserRole.MANAGER, UserRole.HR_ADMIN, UserRole.HR_ANALYST
    ),
    db: AsyncSession = Depends(get_db),
) -> list[TeamSummaryResponse]:
    """Return team-level burnout aggregates for the latest scoring run.

    Access: Manager, HR Admin, HR Analyst.
    No individual pseudo_ids are exposed in this response.
    """
    result = await db.execute(
        select(BurnoutScore.run_id)
        .order_by(BurnoutScore.window_end.desc())
        .limit(1)
    )
    latest_run_id = result.scalar_one_or_none()
    if latest_run_id is None:
        return []

    query = select(BurnoutScore).where(BurnoutScore.run_id == latest_run_id)
    if team_id is not None:
        query = query.where(BurnoutScore.team_id == team_id)

    result = await db.execute(query)
    scores = result.scalars().all()

    null_team_uuid = UUID("00000000-0000-0000-0000-000000000000")
    teams: dict[UUID, list[BurnoutScore]] = defaultdict(list)
    for s in scores:
        tid = s.team_id if s.team_id is not None else null_team_uuid
        teams[tid].append(s)

    summaries: list[TeamSummaryResponse] = []
    for tid, team_scores in teams.items():
        avg_score = sum(s.burnout_score for s in team_scores) / len(team_scores)
        max_score = max(s.burnout_score for s in team_scores)
        high_risk_count = sum(
            1 for s in team_scores if s.burnout_score >= 0.70
        )

        all_features: dict[str, float] = {}
        for s in team_scores:
            for feat, weight in (s.top_features or {}).items():
                all_features[feat] = all_features.get(feat, 0.0) + float(weight)

        top_signals = sorted(
            all_features, key=all_features.get, reverse=True  # type: ignore[arg-type]
        )[:5]

        summaries.append(
            TeamSummaryResponse(
                team_id=tid,
                avg_burnout_score=avg_score,
                max_burnout_score=max_score,
                risk_level=score_to_risk_level(avg_score),
                team_size=len(team_scores),
                high_risk_member_count=high_risk_count,
                top_contributing_signals=top_signals,
                window_end=team_scores[0].window_end,
            )
        )

    return summaries


@router.get("/scores/trend", response_model=list[BurnoutScoreResponse])
async def get_score_trend(
    pseudo_id: UUID = Query(...),
    days: int = Query(default=30, ge=1, le=90),
    token: TokenPayload = require_role(UserRole.HR_ADMIN),
    db: AsyncSession = Depends(get_db),
) -> list[BurnoutScoreResponse]:
    """Return per-run burnout score history for a single pseudonymized user.

    Access: HR Admin only.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        select(BurnoutScore)
        .where(
            BurnoutScore.pseudo_id == pseudo_id,
            BurnoutScore.window_end >= cutoff,
        )
        .order_by(BurnoutScore.window_end.asc())
    )
    scores = result.scalars().all()

    return [
        BurnoutScoreResponse.build(
            pseudo_id=s.pseudo_id,
            burnout_score=s.burnout_score,
            confidence_low=s.confidence_low or 0.0,
            confidence_high=s.confidence_high or 1.0,
            cascade_risk=s.cascade_risk or 0.0,
            cascade_sources=[
                UUID(x) for x in (s.cascade_sources or {}).get("sources", [])
            ],
            top_features=s.top_features or {},
            team_id=s.team_id,
            window_end=s.window_end,
            run_id=s.run_id,
        )
        for s in scores
    ]
