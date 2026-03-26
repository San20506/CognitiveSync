"""Cascade map endpoint — T-057."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import UserRole, score_to_risk_level
from api.schemas.response import (
    CascadeEdgeResponse,
    CascadeMapResponse,
    CascadeNodeResponse,
)
from ingestion.db.models import BurnoutScore, EdgeSignal
from ingestion.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/cascade-map", response_model=CascadeMapResponse)
async def get_cascade_map(
    run_id: UUID | None = Query(default=None),
    token: TokenPayload = require_role(UserRole.HR_ADMIN, UserRole.HR_ANALYST),
    db: AsyncSession = Depends(get_db),
) -> CascadeMapResponse:
    """Return the full cascade risk network for a scoring run.

    Nodes represent pseudonymized employees; edges represent interaction
    signals.  Only nodes and edges belonging to the requested run are
    included.

    Access: HR Admin, HR Analyst only.
    """
    if run_id is None:
        result = await db.execute(
            select(BurnoutScore.run_id)
            .order_by(BurnoutScore.window_end.desc())
            .limit(1)
        )
        run_id = result.scalar_one_or_none()

    if run_id is None:
        return CascadeMapResponse(
            nodes=[],
            edges=[],
            high_risk_sources=[],
            window_end=datetime.now(timezone.utc),
            run_id=uuid4(),
        )

    result = await db.execute(
        select(BurnoutScore).where(BurnoutScore.run_id == run_id)
    )
    scores = result.scalars().all()

    if not scores:
        return CascadeMapResponse(
            nodes=[],
            edges=[],
            high_risk_sources=[],
            window_end=datetime.now(timezone.utc),
            run_id=run_id,
        )

    window_end = scores[0].window_end

    nodes: list[CascadeNodeResponse] = []
    high_risk_sources: list[UUID] = []

    for s in scores:
        score_val = s.burnout_score or 0.0
        is_source = score_val >= 0.70
        if is_source:
            high_risk_sources.append(s.pseudo_id)

        nodes.append(
            CascadeNodeResponse(
                pseudo_id=s.pseudo_id,
                burnout_score=score_val,
                cascade_risk=s.cascade_risk or 0.0,
                risk_level=score_to_risk_level(score_val),
                is_cascade_source=is_source,
                cascade_sources=[
                    UUID(x)
                    for x in (s.cascade_sources or {}).get("sources", [])
                ],
            )
        )

    # Fetch interaction edges whose window aligns with this run.
    result = await db.execute(
        select(EdgeSignal).where(EdgeSignal.window_end == window_end)
    )
    edge_signals = result.scalars().all()

    pseudo_ids_in_run: set[UUID] = {s.pseudo_id for s in scores}
    high_risk_set: set[UUID] = set(high_risk_sources)

    edges: list[CascadeEdgeResponse] = [
        CascadeEdgeResponse(
            source=e.source_pseudo_id,
            target=e.target_pseudo_id,
            weight=e.weight,
            is_cascade_path=(
                e.source_pseudo_id in high_risk_set
                or e.target_pseudo_id in high_risk_set
            ),
        )
        for e in edge_signals
        if (
            e.source_pseudo_id in pseudo_ids_in_run
            and e.target_pseudo_id in pseudo_ids_in_run
        )
    ]

    return CascadeMapResponse(
        nodes=nodes,
        edges=edges,
        high_risk_sources=high_risk_sources,
        window_end=window_end,
        run_id=run_id,
    )
