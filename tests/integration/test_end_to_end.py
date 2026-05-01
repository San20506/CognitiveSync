"""End-to-end integration test scaffold.

Pipeline stages covered (in-process, no real network calls):
data_source -> feature_engineering -> gnn_inference (deterministic stub)
-> risk_cascade -> api_response

Notes:
- External APIs (MS Graph / Slack / GitHub) are mocked by patching adapter.fetch_signals.
- No real database is touched; the test runs entirely in memory.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict
from uuid import UUID, uuid4

import networkx as nx
import pytest

from api.schemas.common import RiskLevel, score_to_risk_level
from ingestion.adapters.base import RawSignals
from ingestion.adapters.github import GitHubAdapter
from ingestion.adapters.msgraph import MSGraphAdapter
from ingestion.adapters.slack import SlackAdapter
from ingestion.anonymizer import Anonymizer
from ingestion.feature_extractor import FeatureExtractor
from intelligence.cascade import CascadePropagator
from intelligence.graph_builder import GraphBuilder
from intelligence.inference import NodeScore, ScoredGraph
from tests.fixtures.synthetic_behavioral_data import adapter_payloads, expected_risk_by_email


class ApiResponse(TypedDict):
    user_id: str
    risk_score: float
    risk_level: str
    top_factors: list[str]
    recommended_actions: list[str]


def _merge_raw_signals(*sources: dict[str, RawSignals]) -> dict[str, RawSignals]:
    """Merge per-source RawSignals into a single dict keyed by real identifier.

    - For scalar fields: first non-None wins.
    - For interactions: weights are summed per target identifier.
    """
    merged: dict[str, RawSignals] = {}

    def _ensure(uid: str) -> RawSignals:
        if uid not in merged:
            merged[uid] = RawSignals()
        return merged[uid]

    scalar_fields = [
        "meeting_density",
        "after_hours_meetings",
        "focus_blocks",
        "email_response_latency",
        "meeting_accept_rate",
        "message_volume",
        "after_hours_messages",
        "response_time_slack",
        "mention_frequency",
        "commit_frequency",
        "after_hours_commits",
        "pr_review_load",
        "context_switch_rate",
    ]

    for src in sources:
        for uid, sig in src.items():
            dst = _ensure(uid)

            for field_name in scalar_fields:
                if getattr(dst, field_name) is None:
                    value = getattr(sig, field_name)
                    if value is not None:
                        setattr(dst, field_name, value)

            for other_id, w in (sig.interactions or {}).items():
                dst.interactions[other_id] = float(dst.interactions.get(other_id, 0.0)) + float(w)

    return merged


def _to_model_feature_space(raw_vec: list[float]) -> dict[str, float]:
    """Map 13-dim ingestion feature vector to the 10-dim model feature space."""
    (
        meeting_density,
        after_hours_meetings,
        focus_blocks,
        email_response_latency,
        _meeting_accept_rate,
        message_volume,
        _after_hours_messages,
        response_time_slack,
        mention_frequency,
        _commit_frequency,
        _after_hours_commits,
        _pr_review_load,
        context_switch_rate,
    ) = raw_vec

    return {
        "meeting_density": meeting_density,
        "after_hours_ratio": after_hours_meetings,
        "response_latency_avg": email_response_latency,
        "focus_time_blocks": focus_blocks,
        "msg_volume_daily": message_volume,
        "msg_response_time": response_time_slack,
        "mention_load": mention_frequency,
        "context_switch_rate": context_switch_rate,
        # Wearable signals aren't part of connector schemas yet; impute neutral.
        "hrv_avg": 0.5,
        "sleep_score": 0.5,
    }


def _deterministic_gnn_inference(
    node_features: dict[UUID, dict[str, float]],
    run_id: UUID,
) -> dict[UUID, NodeScore]:
    """Deterministic scoring stub for integration scaffolding.

    This avoids flaky assertions from random/untrained neural weights while
    keeping the pipeline shape identical to production (scores + top factors).
    """
    node_scores: dict[UUID, NodeScore] = {}
    for pseudo_id, feats in node_features.items():
        # Score is a simple average of high-signal drivers.
        drivers = [
            feats["meeting_density"],
            feats["after_hours_ratio"],
            feats["response_latency_avg"],
            feats["msg_volume_daily"],
            feats["mention_load"],
        ]
        score = float(sum(drivers) / len(drivers))
        score = max(0.0, min(1.0, score))

        # Top factors by value.
        top = sorted(feats.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_factors = {k: float(v) for k, v in top}

        low = max(0.0, score - 0.05)
        high = min(1.0, score + 0.05)

        node_scores[pseudo_id] = NodeScore(
            pseudo_id=pseudo_id,
            burnout_score=score,
            confidence_low=low,
            confidence_high=high,
            top_features=top_factors,
        )

    return node_scores


def _recommended_actions(risk_level: RiskLevel) -> list[str]:
    if risk_level == RiskLevel.HIGH:
        return [
            "Reduce recurring meetings by 30% for the next 2 weeks.",
            "Defer non-critical deliverables to reduce pressure.",
        ]
    if risk_level == RiskLevel.MEDIUM:
        return [
            "Block 2-hour no-meeting windows daily for deep work.",
            "Monitor closely — risk trending upward.",
        ]
    return ["Maintain current practices."]


@pytest.fixture
def synthetic_adapter_data() -> dict[str, dict[str, RawSignals]]:
    return adapter_payloads()


@pytest.fixture
def patch_connectors(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_adapter_data: dict[str, dict[str, RawSignals]],
)-> None:
    async def _msgraph_fetch(
        self: MSGraphAdapter,
        window_start: datetime,
        window_end: datetime,
    ) -> dict[str, RawSignals]:
        return synthetic_adapter_data["msgraph"]

    async def _slack_fetch(
        self: SlackAdapter,
        window_start: datetime,
        window_end: datetime,
    ) -> dict[str, RawSignals]:
        return synthetic_adapter_data["slack"]

    async def _github_fetch(
        self: GitHubAdapter,
        window_start: datetime,
        window_end: datetime,
    ) -> dict[str, RawSignals]:
        return synthetic_adapter_data["github"]

    monkeypatch.setattr(MSGraphAdapter, "fetch_signals", _msgraph_fetch)
    monkeypatch.setattr(SlackAdapter, "fetch_signals", _slack_fetch)
    monkeypatch.setattr(GitHubAdapter, "fetch_signals", _github_fetch)


@pytest.fixture
def time_window() -> tuple[datetime, datetime]:
    window_end = datetime.now(UTC)
    return window_end - timedelta(hours=48), window_end


async def test_end_to_end_pipeline_to_api_schema(
    tmp_path: Path,
    patch_connectors: None,
    time_window: tuple[datetime, datetime],
) -> None:
    window_start, window_end = time_window

    # ------------------------------------------------------------------
    # 1) data_source (mocked adapters)
    # ------------------------------------------------------------------
    msgraph = MSGraphAdapter("", "", "")
    slack = SlackAdapter("", "")
    github = GitHubAdapter("", "")

    raw_msgraph, raw_slack, raw_github = await asyncio.gather(
        msgraph.fetch_signals(window_start, window_end),
        slack.fetch_signals(window_start, window_end),
        github.fetch_signals(window_start, window_end),
    )
    merged_raw = _merge_raw_signals(raw_msgraph, raw_slack, raw_github)

    assert set(merged_raw) == set(expected_risk_by_email())

    # ------------------------------------------------------------------
    # 2) feature_engineering (anonymize + extract)
    # ------------------------------------------------------------------
    vault_path = tmp_path / "vault" / "id_mapping.enc"
    anonymizer = Anonymizer(org_salt="test-salt", vault_path=vault_path, vault_key="test-vault-key")
    anon_signals = anonymizer.anonymize_batch(merged_raw)

    extractor = FeatureExtractor()
    features, edge_batch = extractor.extract_batch(anon_signals, window_start, window_end)

    pseudo_by_email = {email: anonymizer.pseudonymize(email) for email in merged_raw}
    expected_risk = expected_risk_by_email()
    assert len(features) == 5

    # ------------------------------------------------------------------
    # 3) graph_building (in memory)
    # ------------------------------------------------------------------
    # Build node feature dict in the 10-dim model space.
    node_features: dict[UUID, dict[str, float]] = {}
    for fv in features:
        mapped = _to_model_feature_space([float(x) for x in fv.feature_vector.tolist()])
        node_features[fv.pseudo_id] = mapped

    node_ids = sorted(node_features)
    graph: nx.DiGraph = nx.DiGraph()
    for pid in node_ids:
        graph.add_node(pid, **node_features[pid])

    for src, dst, w in edge_batch.edges:
        if src in graph and dst in graph and src != dst:
            graph.add_edge(src, dst, weight=float(w))

    built = GraphBuilder()
    pyg = built.to_pyg(graph, node_ids)
    assert pyg.x.shape[0] == 5

    # ------------------------------------------------------------------
    # 4) gnn_inference (deterministic stub) + 5) risk_cascade
    # ------------------------------------------------------------------
    run_id = uuid4()
    node_scores = _deterministic_gnn_inference(node_features, run_id)

    propagator = CascadePropagator(threshold=0.70, decay_factor=0.60, max_hops=2)
    cascade_results = propagator.propagate(
        graph,
        {pid: ns.burnout_score for pid, ns in node_scores.items()},
    )

    scored = ScoredGraph(node_scores=node_scores, nx_graph=graph, run_id=run_id)
    assert scored.run_id == run_id

    # ------------------------------------------------------------------
    # 6) api_response (schema contract required by task)
    # ------------------------------------------------------------------
    # Expected schema:
    # {user_id, risk_score, risk_level, top_factors, recommended_actions}
    responses: list[ApiResponse] = []

    # invert pseudo_by_email for stable assertions
    email_by_pseudo = {pid: email for email, pid in pseudo_by_email.items()}

    for pid, ns in scored.node_scores.items():
        risk_level = score_to_risk_level(ns.burnout_score)
        responses.append(
            {
                "user_id": str(pid),
                "risk_score": ns.burnout_score,
                "risk_level": risk_level.value,
                "top_factors": list(ns.top_features.keys()),
                "recommended_actions": _recommended_actions(risk_level),
            }
        )

        # Cascade result exists (sources present or absent) and is in [0,1]
        cr = cascade_results.get(pid)
        if cr is not None:
            assert 0.0 <= cr.cascade_risk <= 1.0

    # Schema assertions
    for r in responses:
        assert set(r) == {
            "user_id",
            "risk_score",
            "risk_level",
            "top_factors",
            "recommended_actions",
        }
        assert isinstance(r["user_id"], str)
        assert isinstance(r["risk_score"], float)
        assert r["risk_level"] in {"low", "medium", "high"}
        assert isinstance(r["top_factors"], list)
        assert len(r["top_factors"]) == 3
        assert isinstance(r["recommended_actions"], list)
        assert len(r["recommended_actions"]) >= 1

    # Risk-profile assertions (based on deterministic scoring)
    by_email = {email_by_pseudo[UUID(r["user_id"])]: r for r in responses}
    assert set(by_email) == set(expected_risk)
    for email, exp_level in expected_risk.items():
        assert by_email[email]["risk_level"] == exp_level
