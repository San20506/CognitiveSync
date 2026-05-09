"""GDPR Article 30 audit, right-to-erasure, and data retention endpoints."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import TokenPayload
from api.middleware.rbac import require_role
from api.schemas.common import UserRole
from ingestion.db.models import BurnoutScore, EdgeSignal, Employee
from ingestion.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AuditEventResponse(BaseModel):
    event_id: UUID
    pseudo_id: UUID
    run_id: UUID
    burnout_score: float
    window_end: datetime
    recorded_at: datetime


class ErasureResponse(BaseModel):
    pseudo_id: UUID
    deleted_scores: int
    deleted_edges: int
    deleted_employees: int
    erased_at: datetime


class RetentionPurgeResponse(BaseModel):
    purged_scores: int
    purged_edges: int
    cutoff_date: datetime
    purged_at: datetime


# ---------------------------------------------------------------------------
# A. GET /api/v1/audit/events — Article 30 processing activity log
# ---------------------------------------------------------------------------


@router.get(
    "/audit/events",
    response_model=list[AuditEventResponse],
    summary="GDPR Art. 30 processing activity log",
)
async def list_audit_events(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    token: TokenPayload = require_role(UserRole.IT_ADMIN),
    db: AsyncSession = Depends(get_db),
) -> list[AuditEventResponse]:
    """Return recent BurnoutScore records as GDPR Article 30 audit events.

    Access: IT_ADMIN only.
    """
    stmt = (
        select(BurnoutScore)
        .order_by(BurnoutScore.window_end.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        AuditEventResponse(
            event_id=row.id,
            pseudo_id=row.pseudo_id,
            run_id=row.run_id,
            burnout_score=row.burnout_score,
            window_end=row.window_end,
            recorded_at=row.window_end,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# B. DELETE /api/v1/audit/erasure/{pseudo_id} — Right to erasure (Art. 17)
# ---------------------------------------------------------------------------


@router.delete(
    "/audit/erasure/{pseudo_id}",
    response_model=ErasureResponse,
    summary="GDPR Art. 17 right-to-erasure for a single pseudo_id",
)
async def erase_subject(
    pseudo_id: UUID,
    token: TokenPayload = require_role(UserRole.IT_ADMIN),
    db: AsyncSession = Depends(get_db),
) -> ErasureResponse:
    """Delete all records for pseudo_id from BurnoutScore, Employee, and EdgeSignal.

    Access: IT_ADMIN only.
    """
    # BurnoutScore
    score_result = await db.execute(
        delete(BurnoutScore).where(BurnoutScore.pseudo_id == pseudo_id)
    )
    deleted_scores: int = score_result.rowcount  # type: ignore[assignment]

    # Employee
    emp_result = await db.execute(
        delete(Employee).where(Employee.pseudo_id == pseudo_id)
    )
    deleted_employees: int = emp_result.rowcount  # type: ignore[assignment]

    # EdgeSignal — both directions
    edge_result = await db.execute(
        delete(EdgeSignal).where(
            (EdgeSignal.source_pseudo_id == pseudo_id)
            | (EdgeSignal.target_pseudo_id == pseudo_id)
        )
    )
    deleted_edges: int = edge_result.rowcount  # type: ignore[assignment]

    await db.commit()

    logger.info(
        "GDPR erasure: pseudo_id=%s scores=%d employees=%d edges=%d",
        pseudo_id,
        deleted_scores,
        deleted_employees,
        deleted_edges,
    )

    return ErasureResponse(
        pseudo_id=pseudo_id,
        deleted_scores=deleted_scores,
        deleted_edges=deleted_edges,
        deleted_employees=deleted_employees,
        erased_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# C. DELETE /api/v1/audit/retention/purge — Data retention enforcement
# ---------------------------------------------------------------------------


@router.delete(
    "/audit/retention/purge",
    response_model=RetentionPurgeResponse,
    summary="Purge records older than retention_days (default 90)",
)
async def purge_retention(
    retention_days: int = Query(90, ge=1, le=3650),
    token: TokenPayload = require_role(UserRole.IT_ADMIN),
    db: AsyncSession = Depends(get_db),
) -> RetentionPurgeResponse:
    """Delete BurnoutScore and EdgeSignal records beyond the retention window.

    Access: IT_ADMIN only.
    """
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)

    score_result = await db.execute(
        delete(BurnoutScore).where(BurnoutScore.window_end < cutoff)
    )
    purged_scores: int = score_result.rowcount  # type: ignore[assignment]

    edge_result = await db.execute(
        delete(EdgeSignal).where(EdgeSignal.window_end < cutoff)
    )
    purged_edges: int = edge_result.rowcount  # type: ignore[assignment]

    await db.commit()

    logger.info(
        "Retention purge: cutoff=%s purged_scores=%d purged_edges=%d",
        cutoff.isoformat(),
        purged_scores,
        purged_edges,
    )

    return RetentionPurgeResponse(
        purged_scores=purged_scores,
        purged_edges=purged_edges,
        cutoff_date=cutoff,
        purged_at=datetime.now(UTC),
    )
