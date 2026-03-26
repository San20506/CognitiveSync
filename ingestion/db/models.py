"""SQLAlchemy 2.x async ORM models — implementation in Phase 4A (T-017).

All models use pseudonymized UUIDs only. No real identifiers in any column.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, Float, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FeatureVector(Base):
    """Pseudonymized behavioral feature vector per user per 48h window.

    Schema: features.feature_vectors
    """

    __tablename__ = "feature_vectors"
    __table_args__ = (
        Index("ix_fv_pseudo_id_window", "pseudo_id", "window_start"),
        {"schema": "features"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    pseudo_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    window_start: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    # 13-dim vector: {"meeting_density": 0.3, "after_hours_meetings": 0.7, ...}
    feature_json: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    is_imputed: Mapped[bool] = mapped_column(Boolean, default=False)
    data_completeness: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )


class EdgeSignal(Base):
    """Cross-user interaction signal for graph edge construction.

    Schema: features.edge_signals
    """

    __tablename__ = "edge_signals"
    __table_args__ = (
        Index("ix_edge_window", "window_start"),
        {"schema": "features"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_pseudo_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    target_pseudo_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    window_start: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)


class BurnoutScore(Base):
    """GNN burnout score per node per scoring run.

    Schema: scores.burnout_scores
    """

    __tablename__ = "burnout_scores"
    __table_args__ = (
        Index("ix_bs_run_id", "run_id"),
        Index("ix_bs_pseudo_id", "pseudo_id"),
        {"schema": "scores"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    pseudo_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    burnout_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    cascade_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    # List of source pseudo_ids that contributed cascade risk
    cascade_sources: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    # {feature_name: attention_weight}
    top_features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    team_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    window_end: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )


class ScoringRun(Base):
    """Metadata for each pipeline run.

    Schema: scores.scoring_runs
    """

    __tablename__ = "scoring_runs"
    __table_args__ = {"schema": "scores"}

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    org_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    node_count: Mapped[int | None] = mapped_column(nullable=True)
    edge_count: Mapped[int | None] = mapped_column(nullable=True)
    sources_completed: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditEvent(Base):
    """Audit log — all sensitive events recorded here.

    Schema: audit.events  |  Retention: 12 months minimum
    """

    __tablename__ = "events"
    __table_args__ = (
        Index("ix_audit_event_type", "event_type"),
        Index("ix_audit_created_at", "created_at"),
        {"schema": "audit"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, index=True
    )


class OrgConfig(Base):
    """Organisation configuration.

    Schema: config.orgs
    """

    __tablename__ = "orgs"
    __table_args__ = {"schema": "config"}

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    org_name: Mapped[str] = mapped_column(String(200), nullable=False)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    work_hours_start: Mapped[str] = mapped_column(String(5), default="09:00")
    work_hours_end: Mapped[str] = mapped_column(String(5), default="18:00")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )
