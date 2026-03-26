"""Abstract base class for all connector adapters.

All adapters return raw signals keyed by real user identifiers (email/username).
PII stripping happens immediately downstream in the Anonymizer — NEVER in adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RawSignals:
    """Raw behavioral signals for a single user — PII still present.

    All fields are Optional because sources may not provide every signal.
    None → will be imputed with neutral baseline (0.5) in feature extractor.
    """

    # MS Graph signals
    meeting_density: float | None = None          # meetings per day
    after_hours_meetings: float | None = None     # count outside 09:00-18:00
    focus_blocks: float | None = None             # uninterrupted slots > 90min
    email_response_latency: float | None = None   # avg minutes to reply
    meeting_accept_rate: float | None = None      # accepted / total invites

    # Slack signals
    message_volume: float | None = None           # messages per day
    after_hours_messages: float | None = None     # messages outside work hours
    response_time_slack: float | None = None      # avg DM response time (min)
    mention_frequency: float | None = None        # incoming @mentions per day

    # GitHub signals
    commit_frequency: float | None = None         # commits per day
    after_hours_commits: float | None = None      # commits outside 09:00-18:00
    pr_review_load: float | None = None           # open PRs assigned for review
    context_switch_rate: float | None = None      # distinct repos per week

    # Cross-user interaction signals (for edge construction)
    interactions: dict[str, float] = field(default_factory=dict)
    # {other_user_identifier: interaction_weight}


class BaseAdapter(ABC):
    """Abstract interface for all data source connectors.

    Subclasses must NOT persist raw payloads to any store.
    Raw data lives in memory for the duration of fetch only.
    """

    @abstractmethod
    async def fetch_signals(
        self,
        window_start: datetime,
        window_end: datetime,
    ) -> dict[str, RawSignals]:
        """Fetch behavioral signals for all org users in the given window.

        Args:
            window_start: Start of rolling window (typically 48h ago).
            window_end: End of rolling window (now).

        Returns:
            Dict mapping real user identifier (email/username) to RawSignals.
            PII is still present — Anonymizer handles stripping downstream.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify connectivity and auth token validity."""
        ...
