"""Synthetic behavioral fixture data for integration tests.

Provides 5 users with known risk profiles:
- 2 high risk
- 2 medium risk
- 1 low risk

The schema matches connector outputs: mapping of real user identifier (email) -> RawSignals.
"""

from __future__ import annotations

from dataclasses import dataclass

from ingestion.adapters.base import RawSignals


@dataclass(frozen=True)
class SyntheticUser:
    email: str
    expected_risk_level: str  # "high" | "medium" | "low"


USERS: list[SyntheticUser] = [
    SyntheticUser(email="alice.high@cognitivesync.local", expected_risk_level="high"),
    SyntheticUser(email="bob.high@cognitivesync.local", expected_risk_level="high"),
    SyntheticUser(email="carol.medium@cognitivesync.local", expected_risk_level="medium"),
    SyntheticUser(email="dave.medium@cognitivesync.local", expected_risk_level="medium"),
    SyntheticUser(email="eve.low@cognitivesync.local", expected_risk_level="low"),
]


def expected_risk_by_email() -> dict[str, str]:
    return {u.email: u.expected_risk_level for u in USERS}


def msgraph_fixture() -> dict[str, RawSignals]:
    """MS Graph-like signals (calendar/mail) + collaboration interactions.

    Values are already in [0,1] to avoid normalization ambiguity in tests.
    """
    a, b, c, d, e = [u.email for u in USERS]
    return {
        a: RawSignals(
            meeting_density=0.95,
            after_hours_meetings=0.85,
            focus_blocks=0.10,
            email_response_latency=0.90,
            meeting_accept_rate=0.20,
            interactions={c: 0.8, d: 0.6},
        ),
        b: RawSignals(
            meeting_density=0.90,
            after_hours_meetings=0.80,
            focus_blocks=0.15,
            email_response_latency=0.88,
            meeting_accept_rate=0.25,
            interactions={a: 0.7, e: 0.4},
        ),
        c: RawSignals(
            meeting_density=0.55,
            after_hours_meetings=0.45,
            focus_blocks=0.45,
            email_response_latency=0.55,
            meeting_accept_rate=0.55,
            interactions={a: 0.4, d: 0.3},
        ),
        d: RawSignals(
            meeting_density=0.50,
            after_hours_meetings=0.40,
            focus_blocks=0.50,
            email_response_latency=0.52,
            meeting_accept_rate=0.60,
            interactions={a: 0.35, c: 0.25},
        ),
        e: RawSignals(
            meeting_density=0.20,
            after_hours_meetings=0.10,
            focus_blocks=0.80,
            email_response_latency=0.20,
            meeting_accept_rate=0.85,
            interactions={b: 0.10},
        ),
    }


def slack_fixture() -> dict[str, RawSignals]:
    """Slack-like signals (message/mention/response)."""
    a, b, c, d, e = [u.email for u in USERS]
    return {
        a: RawSignals(
            message_volume=0.95,
            after_hours_messages=0.85,
            response_time_slack=0.80,
            mention_frequency=0.90,
            interactions={b: 0.5, c: 0.5},
        ),
        b: RawSignals(
            message_volume=0.90,
            after_hours_messages=0.80,
            response_time_slack=0.78,
            mention_frequency=0.88,
            interactions={a: 0.6, d: 0.4},
        ),
        c: RawSignals(
            message_volume=0.55,
            after_hours_messages=0.40,
            response_time_slack=0.50,
            mention_frequency=0.55,
            interactions={a: 0.4, d: 0.6},
        ),
        d: RawSignals(
            message_volume=0.50,
            after_hours_messages=0.35,
            response_time_slack=0.48,
            mention_frequency=0.52,
            interactions={b: 0.3, c: 0.7},
        ),
        e: RawSignals(
            message_volume=0.25,
            after_hours_messages=0.10,
            response_time_slack=0.20,
            mention_frequency=0.20,
            interactions={c: 0.2},
        ),
    }


def github_fixture() -> dict[str, RawSignals]:
    """GitHub-like signals (commits/PRs/context switching)."""
    a, b, c, d, e = [u.email for u in USERS]
    return {
        a: RawSignals(
            commit_frequency=0.85,
            after_hours_commits=0.85,
            pr_review_load=0.90,
            context_switch_rate=0.80,
            interactions={b: 0.7, d: 0.3},
        ),
        b: RawSignals(
            commit_frequency=0.80,
            after_hours_commits=0.80,
            pr_review_load=0.88,
            context_switch_rate=0.78,
            interactions={a: 0.6, c: 0.4},
        ),
        c: RawSignals(
            commit_frequency=0.55,
            after_hours_commits=0.45,
            pr_review_load=0.55,
            context_switch_rate=0.55,
            interactions={d: 0.5},
        ),
        d: RawSignals(
            commit_frequency=0.50,
            after_hours_commits=0.40,
            pr_review_load=0.52,
            context_switch_rate=0.50,
            interactions={c: 0.5},
        ),
        e: RawSignals(
            commit_frequency=0.20,
            after_hours_commits=0.10,
            pr_review_load=0.20,
            context_switch_rate=0.15,
            interactions={b: 0.2},
        ),
    }


def adapter_payloads() -> dict[str, dict[str, RawSignals]]:
    """Convenience bundle for tests."""
    return {
        "msgraph": msgraph_fixture(),
        "slack": slack_fixture(),
        "github": github_fixture(),
    }
