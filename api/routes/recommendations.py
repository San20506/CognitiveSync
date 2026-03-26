"""Workload redistribution recommendation endpoints — T-058."""

from __future__ import annotations

import logging
from collections import defaultdict
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import UserRole
from api.schemas.response import RecommendationResponse
from ingestion.db.models import BurnoutScore
from ingestion.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

_NULL_TEAM_UUID = UUID("00000000-0000-0000-0000-000000000000")


@router.get("/recommendations", response_model=list[RecommendationResponse])
async def get_recommendations(
    team_id: UUID | None = Query(default=None),
    token: TokenPayload = require_role(
        UserRole.HR_ADMIN, UserRole.HR_ANALYST, UserRole.MANAGER
    ),
    db: AsyncSession = Depends(get_db),
) -> list[RecommendationResponse]:
    """Return workload redistribution recommendations derived from burnout scores.

    Recommendations are generated at team granularity from the latest
    scoring run.  No individual pseudo_ids are included in the response.

    Access: HR Admin, HR Analyst (all teams); Manager (filtered by team_id).
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

    teams: dict[UUID, list[BurnoutScore]] = defaultdict(list)
    for s in scores:
        tid = s.team_id if s.team_id is not None else _NULL_TEAM_UUID
        teams[tid].append(s)

    recs: list[RecommendationResponse] = []

    for tid, team_scores in teams.items():
        avg_score = sum(s.burnout_score for s in team_scores) / len(team_scores)
        high_risk_count = sum(
            1 for s in team_scores if s.burnout_score >= 0.70
        )

        recommendations: list[str] = []
        focus_time_suggestion: str | None = None
        meeting_reduction_suggestion: str | None = None
        workload_redistribution: list[str] = []

        if avg_score >= 0.70:
            recommendations.append(
                "Immediate workload redistribution required for this team."
            )
            meeting_reduction_suggestion = (
                "Reduce recurring meetings by 30% for the next 2 weeks."
            )
            workload_redistribution.append(
                "Defer non-critical deliverables to reduce team pressure."
            )
        elif avg_score >= 0.40:
            recommendations.append(
                "Monitor team closely — risk trending upward."
            )
            focus_time_suggestion = (
                "Block 2-hour no-meeting windows daily for deep work."
            )
        else:
            recommendations.append(
                "Team is in healthy range. Maintain current practices."
            )

        if high_risk_count > 0:
            workload_redistribution.append(
                f"{high_risk_count} team member(s) showing high-risk signals"
                " — consider redistributing their critical tasks."
            )

        recs.append(
            RecommendationResponse(
                team_id=tid,
                recommendations=recommendations,
                focus_time_suggestion=focus_time_suggestion,
                meeting_reduction_suggestion=meeting_reduction_suggestion,
                workload_redistribution=workload_redistribution,
            )
        )

    return recs
