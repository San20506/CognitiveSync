"""Pydantic v2 request schemas for all API endpoints."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PipelineRunRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    dry_run: bool = Field(
        default=False,
        description="If true, run pipeline but do not persist scores or trigger alerts.",
    )
    adapter_override: str | None = Field(
        default=None,
        description="Override adapter mode for this run only (e.g. 'mock' for testing).",
    )


class ModelRetrainRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    promote_on_success: bool = Field(
        default=True,
        description="Automatically promote model if val_accuracy >= 0.80.",
    )
    epochs: int = Field(default=100, ge=1, le=500)
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.1)


class OrgConfigCreateRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    org_name: str = Field(min_length=1, max_length=200)
    timezone: str = Field(default="UTC", description="IANA timezone string.")
    work_hours_start: str = Field(default="09:00", pattern=r"^\d{2}:\d{2}$")
    work_hours_end: str = Field(default="18:00", pattern=r"^\d{2}:\d{2}$")


class AlertConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    alert_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    cascade_alert_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    cascade_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    decay_factor: float | None = Field(default=None, ge=0.0, le=1.0)
    max_hops: int | None = Field(default=None, ge=1, le=5)
    hr_teams_channel_id: str | None = None
    manager_teams_channel_ids: dict[str, str] | None = None


class AuditQueryRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    event_type: str | None = None
    org_id: UUID | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
