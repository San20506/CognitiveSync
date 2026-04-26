"""Profile updater — runs after each scoring pipeline run.

Updates EmployeeProfile for every scored node:
- Appends score snapshot to rolling trend (last 10)
- Recalculates avg_score_30d from trend
- Updates latest_score, top_features, cascade_exposure_count
- Auto-enrolls any unknown pseudo_id with a minimal Employee record (guard)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ingestion.db.models import Employee, EmployeeProfile
from intelligence.cascade import CascadeResult
from intelligence.inference import NodeScore

logger = logging.getLogger(__name__)

TREND_MAX = 10


async def update_profiles(
    run_id: UUID,
    node_scores: dict[UUID, NodeScore],
    cascade_results: dict[UUID, CascadeResult],
    db: AsyncSession,
) -> None:
    """Upsert EmployeeProfile for every node in this scoring run.

    Also auto-enrolls any pseudo_id not yet in config.employees.
    """
    now = datetime.now(UTC)

    for pseudo_id, ns in node_scores.items():
        cr = cascade_results.get(pseudo_id)
        cascade_risk = cr.cascade_risk if cr else 0.0
        has_cascade = cascade_risk > 0.0

        await _ensure_enrolled(pseudo_id, db, now)

        result = await db.execute(
            select(EmployeeProfile).where(EmployeeProfile.pseudo_id == pseudo_id)
        )
        prof = result.scalar_one_or_none()

        snapshot = {
            "run_id": str(run_id),
            "score": ns.burnout_score,
            "cascade_risk": cascade_risk,
            "ts": now.isoformat(),
        }

        if prof is None:
            prof = EmployeeProfile(
                pseudo_id=pseudo_id,
                latest_score=ns.burnout_score,
                score_trend=[snapshot],
                top_features=ns.top_features,
                cascade_exposure_count=1 if has_cascade else 0,
                run_count=1,
                updated_at=now,
                created_at=now,
            )
            db.add(prof)
        else:
            trend: list[dict] = list(prof.score_trend or [])  # type: ignore[arg-type]
            trend.append(snapshot)
            if len(trend) > TREND_MAX:
                trend = trend[-TREND_MAX:]

            cutoff = now - timedelta(days=30)
            recent = [s for s in trend if s["ts"] >= cutoff.isoformat()]
            avg_30d = sum(s["score"] for s in recent) / len(recent) if recent else ns.burnout_score

            prof.latest_score = ns.burnout_score
            prof.score_trend = trend
            prof.avg_score_30d = avg_30d
            prof.top_features = ns.top_features
            prof.cascade_exposure_count = (prof.cascade_exposure_count or 0) + (
                1 if has_cascade else 0
            )
            prof.run_count = (prof.run_count or 0) + 1
            prof.updated_at = now

    await db.flush()
    logger.info("Updated profiles for %d nodes in run %s", len(node_scores), run_id)


async def _ensure_enrolled(
    pseudo_id: UUID,
    db: AsyncSession,
    now: datetime,
) -> None:
    """Auto-enroll a pseudo_id that arrived in pipeline data but was never explicitly enrolled."""
    result = await db.execute(select(Employee).where(Employee.pseudo_id == pseudo_id))
    if result.scalar_one_or_none() is None:
        emp = Employee(
            pseudo_id=pseudo_id,
            is_active=True,
            enrolled_at=now,
            updated_at=now,
        )
        db.add(emp)
        logger.info("Auto-enrolled unknown pseudo_id=%s", pseudo_id)
