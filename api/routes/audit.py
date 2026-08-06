"""GDPR Article 30 audit, right-to-erasure, and data retention endpoints.

Erasure covers all 6 data stores:
  BurnoutScore, Employee, EdgeSignal, FeatureVector, EmployeeProfile, vault mapping.

Every action is logged to the tamper-evident audit trail (audit.events).
"""

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
from api.services.audit_service import log_event
from config.settings import settings
from config.vault import EncryptedMappingStore
from ingestion.db.models import (
    AuditEvent,
    BurnoutScore,
    EdgeSignal,
    Employee,
    EmployeeProfile,
    FeatureVector,
)
from ingestion.db.session import get_audit_db, get_db

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
    deleted_features: int
    deleted_edges: int
    deleted_employees: int
    deleted_profiles: int
    vault_purged: bool
    erased_at: datetime


class RetentionPurgeResponse(BaseModel):
    purged_scores: int
    purged_features: int
    purged_edges: int
    cutoff_date: datetime
    purged_at: datetime


class AuditLogEntry(BaseModel):
    event_id: UUID
    event_type: str
    actor_id: UUID
    action: str
    resource_type: str
    resource_id: UUID | None
    event_hash: str
    created_at: datetime


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
    audit_db: AsyncSession = Depends(get_audit_db),
) -> list[AuditEventResponse]:
    """Return recent BurnoutScore records as GDPR Article 30 audit events.

    Access: IT_ADMIN only.
    """
    # Log the access event
    await log_event(
        audit_db,
        event_type="gdpr.audit_access",
        actor_id=token.sub,
        action="accessed",
        resource_type="BurnoutScore",
        payload={"limit": limit, "offset": offset},
    )

    stmt = select(BurnoutScore).order_by(BurnoutScore.window_end.desc()).limit(limit).offset(offset)
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
    audit_db: AsyncSession = Depends(get_audit_db),
) -> ErasureResponse:
    """Delete all records for pseudo_id across all 6 data stores.

    Covers: BurnoutScore, FeatureVector, EdgeSignal, Employee,
    EmployeeProfile, and the encrypted vault mapping.

    Access: IT_ADMIN only.
    """
    # 1. BurnoutScore
    score_result = await db.execute(delete(BurnoutScore).where(BurnoutScore.pseudo_id == pseudo_id))
    deleted_scores: int = score_result.rowcount  # type: ignore[assignment]

    # 2. FeatureVector
    fv_result = await db.execute(delete(FeatureVector).where(FeatureVector.pseudo_id == pseudo_id))
    deleted_features: int = fv_result.rowcount  # type: ignore[assignment]

    # 3. EdgeSignal — both directions
    edge_result = await db.execute(
        delete(EdgeSignal).where(
            (EdgeSignal.source_pseudo_id == pseudo_id) | (EdgeSignal.target_pseudo_id == pseudo_id)
        )
    )
    deleted_edges: int = edge_result.rowcount  # type: ignore[assignment]

    # 4. Employee
    emp_result = await db.execute(delete(Employee).where(Employee.pseudo_id == pseudo_id))
    deleted_employees: int = emp_result.rowcount  # type: ignore[assignment]

    # 5. EmployeeProfile
    profile_result = await db.execute(
        delete(EmployeeProfile).where(EmployeeProfile.pseudo_id == pseudo_id)
    )
    deleted_profiles: int = profile_result.rowcount  # type: ignore[assignment]

    # 6. Encrypted vault mapping
    vault = EncryptedMappingStore(settings.vault_path, settings.vault_key)
    vault.purge(pseudo_id)

    await db.commit()

    # Log erasure to tamper-evident audit trail
    await log_event(
        audit_db,
        event_type="gdpr.erasure",
        actor_id=token.sub,
        action="erased",
        resource_type="Employee",
        resource_id=pseudo_id,
        payload={
            "deleted_scores": deleted_scores,
            "deleted_features": deleted_features,
            "deleted_edges": deleted_edges,
            "deleted_employees": deleted_employees,
            "deleted_profiles": deleted_profiles,
            "vault_purged": True,
        },
    )
    await audit_db.commit()

    logger.info(
        "GDPR erasure: pseudo_id=%s scores=%d features=%d employees=%d"
        " profiles=%d edges=%d vault=1",
        pseudo_id,
        deleted_scores,
        deleted_features,
        deleted_employees,
        deleted_profiles,
        deleted_edges,
    )

    return ErasureResponse(
        pseudo_id=pseudo_id,
        deleted_scores=deleted_scores,
        deleted_features=deleted_features,
        deleted_edges=deleted_edges,
        deleted_employees=deleted_employees,
        deleted_profiles=deleted_profiles,
        vault_purged=True,
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
    audit_db: AsyncSession = Depends(get_audit_db),
) -> RetentionPurgeResponse:
    """Delete BurnoutScore, FeatureVector, and EdgeSignal records beyond the retention window.

    Access: IT_ADMIN only.
    """
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)

    score_result = await db.execute(delete(BurnoutScore).where(BurnoutScore.window_end < cutoff))
    purged_scores: int = score_result.rowcount  # type: ignore[assignment]

    fv_result = await db.execute(delete(FeatureVector).where(FeatureVector.window_end < cutoff))
    purged_features: int = fv_result.rowcount  # type: ignore[assignment]

    edge_result = await db.execute(delete(EdgeSignal).where(EdgeSignal.window_end < cutoff))
    purged_edges: int = edge_result.rowcount  # type: ignore[assignment]

    await db.commit()

    # Log retention purge to audit trail
    await log_event(
        audit_db,
        event_type="gdpr.retention_purge",
        actor_id=token.sub,
        action="purged",
        resource_type="BurnoutScore",
        payload={
            "retention_days": retention_days,
            "cutoff": cutoff.isoformat(),
            "purged_scores": purged_scores,
            "purged_features": purged_features,
            "purged_edges": purged_edges,
        },
    )
    await audit_db.commit()

    logger.info(
        "Retention purge: cutoff=%s purged_scores=%d purged_features=%d purged_edges=%d",
        cutoff.isoformat(),
        purged_scores,
        purged_features,
        purged_edges,
    )

    return RetentionPurgeResponse(
        purged_scores=purged_scores,
        purged_features=purged_features,
        purged_edges=purged_edges,
        cutoff_date=cutoff,
        purged_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# D. GET /api/v1/audit/log — Tamper-evident audit log (Art. 30)
# ---------------------------------------------------------------------------


@router.get(
    "/audit/log",
    response_model=list[AuditLogEntry],
    summary="Tamper-evident audit event log with hash chain",
)
async def list_audit_log(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    token: TokenPayload = require_role(UserRole.IT_ADMIN),
    audit_db: AsyncSession = Depends(get_audit_db),
) -> list[AuditLogEntry]:
    """Return audit events with hash chain for tamper verification.

    Access: IT_ADMIN only.
    """
    stmt = select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit).offset(offset)
    result = await audit_db.execute(stmt)
    rows = result.scalars().all()

    return [
        AuditLogEntry(
            event_id=row.id,
            event_type=row.event_type,
            actor_id=row.actor_id,
            action=row.action,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            event_hash=row.event_hash,
            created_at=row.created_at,
        )
        for row in rows
    ]
