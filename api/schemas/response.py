"""Pydantic v2 response schemas — strict mode enforced on all score/cascade models.

These are the API contracts. Implementation must conform to these types exactly.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.schemas.common import PipelineStatus, RiskLevel, score_to_risk_level


# ---------------------------------------------------------------------------
# Burnout Score Responses
# ---------------------------------------------------------------------------


class BurnoutScoreResponse(BaseModel):
    """Individual burnout score — HR Admin and HR Analyst only.

    RBAC: Never returned to manager role. Manager sees TeamSummaryResponse only.
    """

    model_config = ConfigDict(strict=True)

    pseudo_id: UUID
    burnout_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    confidence_low: float = Field(ge=0.0, le=1.0)
    confidence_high: float = Field(ge=0.0, le=1.0)
    cascade_risk: float = Field(ge=0.0, le=1.0)
    cascade_sources: list[UUID] = Field(default_factory=list)
    top_features: dict[str, float] = Field(
        description="GAT attention weights per feature — {feature_name: weight}."
    )
    team_id: UUID | None = None
    window_end: datetime
    run_id: UUID

    @classmethod
    def build(
        cls,
        pseudo_id: UUID,
        burnout_score: float,
        confidence_low: float,
        confidence_high: float,
        cascade_risk: float,
        cascade_sources: list[UUID],
        top_features: dict[str, float],
        team_id: UUID | None,
        window_end: datetime,
        run_id: UUID,
    ) -> BurnoutScoreResponse:
        return cls(
            pseudo_id=pseudo_id,
            burnout_score=burnout_score,
            risk_level=score_to_risk_level(burnout_score),
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            cascade_risk=cascade_risk,
            cascade_sources=cascade_sources,
            top_features=top_features,
            team_id=team_id,
            window_end=window_end,
            run_id=run_id,
        )


class TeamSummaryResponse(BaseModel):
    """Team-level aggregate — returned to manager role.

    No individual pseudo_ids exposed. Redistribution guidance only.
    """

    model_config = ConfigDict(strict=True)

    team_id: UUID
    avg_burnout_score: float = Field(ge=0.0, le=1.0)
    max_burnout_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    team_size: int = Field(ge=0)
    high_risk_member_count: int = Field(ge=0)
    top_contributing_signals: list[str] = Field(
        description="Signal names only — no individual attribution."
    )
    window_end: datetime


# ---------------------------------------------------------------------------
# Cascade Map Responses
# ---------------------------------------------------------------------------


class CascadeNodeResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    pseudo_id: UUID
    burnout_score: float = Field(ge=0.0, le=1.0)
    cascade_risk: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    is_cascade_source: bool
    cascade_sources: list[UUID] = Field(default_factory=list)


class CascadeEdgeResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    source: UUID
    target: UUID
    weight: float = Field(ge=0.0, le=1.0)
    is_cascade_path: bool


class CascadeMapResponse(BaseModel):
    """Full cascade network — HR Admin and HR Analyst only."""

    model_config = ConfigDict(strict=True)

    nodes: list[CascadeNodeResponse]
    edges: list[CascadeEdgeResponse]
    high_risk_sources: list[UUID]
    window_end: datetime
    run_id: UUID


# ---------------------------------------------------------------------------
# Recommendation Responses
# ---------------------------------------------------------------------------


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    team_id: UUID
    recommendations: list[str]
    focus_time_suggestion: str | None = None
    meeting_reduction_suggestion: str | None = None
    workload_redistribution: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline Responses
# ---------------------------------------------------------------------------


class ScoreDistribution(BaseModel):
    model_config = ConfigDict(strict=True)

    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    mean_score: float
    max_score: float


class PipelineRunResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    run_id: UUID
    status: PipelineStatus
    node_count: int | None = None
    edge_count: int | None = None
    duration_seconds: float | None = None
    score_distribution: ScoreDistribution | None = None
    error_message: str | None = None
    sources_completed: list[str] = Field(default_factory=list)
    sources_failed: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model Responses
# ---------------------------------------------------------------------------


class ModelMetadata(BaseModel):
    model_config = ConfigDict(strict=True)

    version: str
    training_date: datetime
    val_accuracy: float
    val_f1: float
    auc_roc: float
    graph_size: int
    feature_schema_version: str
    is_active: bool


class ModelTrainingResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    job_id: UUID
    status: PipelineStatus
    model_version: str | None = None
    metrics: ModelMetadata | None = None
    promoted: bool = False
    message: str


# ---------------------------------------------------------------------------
# Config & Audit Responses
# ---------------------------------------------------------------------------


class OrgConfigResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    org_id: UUID
    org_name: str
    timezone: str
    work_hours_start: str  # "HH:MM"
    work_hours_end: str    # "HH:MM"
    alert_threshold: float
    cascade_threshold: float


class AlertConfigResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    alert_threshold: float
    cascade_alert_threshold: float
    cascade_threshold: float
    decay_factor: float
    max_hops: int
    hr_channel_configured: bool
    manager_channels_count: int


class AuditEventResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    event_id: UUID
    event_type: str
    payload: dict[str, object]
    created_at: datetime
