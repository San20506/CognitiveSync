"""Enrollment and individual profile schemas — T-060, T-061, T-062."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ── Enrollment ────────────────────────────────────────────────────────────────


class InitialProfileSeed(BaseModel):
    """Optional template supplied at enrollment to pre-seed the profile.

    All fields optional — omitted fields use system defaults.
    known_stressors is free text hashed before storage; never stored raw.
    """

    model_config = ConfigDict(strict=True)

    role_risk_modifier: float = Field(
        default=0.0,
        ge=-0.3,
        le=0.3,
        description="Role-based baseline adjustment. "
        "Positive = higher-risk role (e.g. on-call). Negative = lower-risk.",
    )
    work_hours_start: str = Field(default="09:00", pattern=r"^\d{2}:\d{2}$")
    work_hours_end: str = Field(default="18:00", pattern=r"^\d{2}:\d{2}$")
    expected_after_hours_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Expected after-hours work fraction for this role (baseline).",
    )
    known_stressors: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Optional stressor tags (hashed before storage).",
    )
    notes_hash: str | None = Field(
        default=None,
        description="SHA-256 of any freeform HR notes. Never store raw text.",
    )


class EnrollRequest(BaseModel):
    """Payload for POST /api/v1/employees/enroll."""

    model_config = ConfigDict(strict=True)

    pseudo_id: UUID = Field(description="Pre-computed UUID v5 pseudonym for this employee.")
    display_name_hash: str | None = Field(
        default=None,
        max_length=64,
        description="SHA-256 of display name — for de-dup only.",
    )
    team_id: str | None = Field(default=None, max_length=100)
    role: str | None = Field(default=None, max_length=100)
    seniority: str | None = Field(default=None, max_length=50)
    timezone: str = Field(default="UTC", max_length=50)
    initial_profile: InitialProfileSeed | None = None


class EnrollResponse(BaseModel):
    pseudo_id: UUID
    enrolled_at: datetime
    profile_seeded: bool
    message: str = ""


class EmployeeResponse(BaseModel):
    pseudo_id: UUID
    team_id: str | None
    role: str | None
    seniority: str | None
    timezone: str
    work_hours_start: str
    work_hours_end: str
    is_active: bool
    enrolled_at: datetime
    updated_at: datetime


# ── Individual Profile ────────────────────────────────────────────────────────


class ScoreSnapshot(BaseModel):
    run_id: UUID
    score: float
    cascade_risk: float
    ts: datetime


class ProfileResponse(BaseModel):
    """Individual employee profile — GET /api/v1/profiles/{pseudo_id}."""

    model_config = ConfigDict(strict=True)

    pseudo_id: UUID
    team_id: str | None
    latest_score: float | None
    avg_score_30d: float | None
    score_trend: list[ScoreSnapshot] = Field(default_factory=list)
    top_features: dict[str, float] = Field(default_factory=dict)
    cascade_exposure_count: int
    run_count: int
    risk_trajectory: str = Field(
        description="'improving', 'stable', 'worsening', or 'insufficient_data'",
    )
    seed_data: dict | None = None  # type: ignore[type-arg]
    updated_at: datetime


class TeamProfileSummary(BaseModel):
    """Team-level profile summary — GET /api/v1/profiles/team/{team_id}."""

    team_id: str
    member_count: int
    avg_score: float
    high_risk_count: int
    avg_cascade_exposure: float
    top_signals: list[str]
