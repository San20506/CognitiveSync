"""Shared types and enumerations used across request/response schemas."""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"        # burnout_score < 0.40
    MEDIUM = "medium"  # 0.40 <= burnout_score < 0.70
    HIGH = "high"      # burnout_score >= 0.70


class PipelineStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some sources failed but run completed with degraded features


class UserRole(str, Enum):
    HR_ADMIN = "hr_admin"
    HR_ANALYST = "hr_analyst"
    MANAGER = "manager"
    IT_ADMIN = "it_admin"


class AdapterSource(str, Enum):
    MSGRAPH = "msgraph"
    SLACK = "slack"
    GITHUB = "github"
    MOCK = "mock"


def score_to_risk_level(score: float) -> RiskLevel:
    """Convert a float burnout score [0,1] to a RiskLevel label."""
    if score >= 0.70:
        return RiskLevel.HIGH
    if score >= 0.40:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
