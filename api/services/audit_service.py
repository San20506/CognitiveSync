"""Tamper-evident audit event writer with hash chaining.

Every audit event is linked to its predecessor via SHA-256 hash chaining,
making retroactive tampering mathematically detectable.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ingestion.db.models import AuditEvent

logger = logging.getLogger(__name__)

_ZERO_HASH = "0" * 64


def _compute_event_hash(
    prev_hash: str | None,
    actor_id: UUID,
    action: str,
    resource_type: str,
    resource_id: UUID | None,
    created_at: datetime,
) -> str:
    """Compute SHA-256 hash chain for tamper-evidence."""
    seed = (
        f"{prev_hash or _ZERO_HASH}|{created_at.isoformat()}"
        f"|{actor_id}|{action}|{resource_type}|{resource_id}"
    )
    return hashlib.sha256(seed.encode()).hexdigest()


async def log_event(
    db: AsyncSession,
    *,
    event_type: str,
    actor_id: UUID,
    action: str,
    resource_type: str,
    resource_id: UUID | None = None,
    payload: dict,
) -> AuditEvent:
    """Write a tamper-evident audit event with hash chaining.

    Args:
        db: Async SQLAlchemy session.
        event_type: Category (e.g. "gdpr.erasure", "gdpr.retention_purge").
        actor_id: Pseudo-ID of the actor performing the action.
        action: Verb (e.g. "erased", "purged", "accessed").
        resource_type: Target model (e.g. "Employee", "BurnoutScore").
        resource_id: Pseudo-ID of the affected resource (if applicable).
        payload: Arbitrary event metadata (counts, timestamps, etc.).

    Returns:
        The created AuditEvent with computed event_hash.
    """
    # Fetch the most recent event hash for chaining
    prev_event = await db.execute(
        select(AuditEvent.event_hash).order_by(AuditEvent.created_at.desc()).limit(1)
    )
    prev_hash: str | None = prev_event.scalar()

    now = datetime.now(UTC)
    event_hash = _compute_event_hash(prev_hash, actor_id, action, resource_type, resource_id, now)

    event = AuditEvent(
        event_type=event_type,
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        payload=payload,
        prev_hash=prev_hash,
        event_hash=event_hash,
        created_at=now,
    )
    db.add(event)
    await db.flush()  # assign id without committing — caller controls transaction

    logger.info(
        "Audit event: type=%s action=%s resource=%s id=%s hash=%s",
        event_type,
        action,
        resource_type,
        resource_id,
        event_hash[:12],
    )
    return event
