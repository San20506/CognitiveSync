"""Unit tests for cascade propagation module — T-055."""

from __future__ import annotations

from uuid import UUID, uuid4

import networkx as nx
import pytest

from intelligence.cascade import CascadeResult, CascadePropagator


def _graph(*edges: tuple[UUID, UUID, float]) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    for src, dst, w in edges:
        G.add_edge(src, dst, weight=w)
    return G


class TestCascadeOnHop:
    """T-055: 1-hop propagation verification."""

    def test_one_hop_single_neighbor(self) -> None:
        source = uuid4()
        neighbor = uuid4()
        G = _graph((source, neighbor, 1.0))
        scores = {source: 0.9, neighbor: 0.1}

        result = CascadePropagator(threshold=0.70).propagate(G, scores)

        assert neighbor in result
        # contribution = 0.9 * 1.0 * 0.6^1 = 0.54; max_risk = 0.54 → normalized = 1.0
        assert abs(result[neighbor].cascade_risk - 1.0) < 1e-6

    def test_one_hop_weight_scales_contribution(self) -> None:
        source = uuid4()
        n1 = uuid4()
        n2 = uuid4()
        G = _graph((source, n1, 0.5), (source, n2, 1.0))
        scores = {source: 0.9, n1: 0.0, n2: 0.0}

        result = CascadePropagator(threshold=0.70).propagate(G, scores)

        # n2 should have higher risk than n1 (weight 1.0 > 0.5)
        assert result[n2].cascade_risk > result[n1].cascade_risk

    def test_node_below_threshold_not_a_source(self) -> None:
        low_risk = uuid4()
        neighbor = uuid4()
        G = _graph((low_risk, neighbor, 1.0))
        scores = {low_risk: 0.5, neighbor: 0.1}  # 0.5 < threshold 0.70

        result = CascadePropagator(threshold=0.70).propagate(G, scores)

        # No sources → no cascade
        assert neighbor not in result


class TestCascadeTwoHop:
    """T-055: 2-hop propagation verification."""

    def test_two_hop_reaches_second_neighbor(self) -> None:
        source = uuid4()
        mid = uuid4()
        far = uuid4()
        # source → mid → far
        G = _graph((source, mid, 1.0), (mid, far, 1.0))
        scores = {source: 0.9, mid: 0.0, far: 0.0}

        result = CascadePropagator(threshold=0.70, decay_factor=0.6, max_hops=2).propagate(G, scores)

        assert mid in result
        assert far in result

    def test_two_hop_decay_applied(self) -> None:
        source = uuid4()
        hop1 = uuid4()
        hop2 = uuid4()
        G = _graph((source, hop1, 1.0), (hop1, hop2, 1.0))
        scores = {source: 1.0, hop1: 0.0, hop2: 0.0}

        propagator = CascadePropagator(threshold=0.70, decay_factor=0.6, max_hops=2)
        result = propagator.propagate(G, scores)

        # hop1 raw = 1.0 * 1.0 * 0.6^1 = 0.6
        # hop2 raw = 1.0 * 1.0 * 0.6^2 = 0.36
        # max = 0.6 → hop1 normalized=1.0, hop2 normalized=0.6
        assert result[hop1].cascade_risk > result[hop2].cascade_risk

    def test_beyond_max_hops_not_reached(self) -> None:
        source = uuid4()
        hop1 = uuid4()
        hop2 = uuid4()
        hop3 = uuid4()
        G = _graph((source, hop1, 1.0), (hop1, hop2, 1.0), (hop2, hop3, 1.0))
        scores = {source: 0.9, hop1: 0.0, hop2: 0.0, hop3: 0.0}

        result = CascadePropagator(threshold=0.70, max_hops=2).propagate(G, scores)

        assert hop1 in result
        assert hop2 in result
        # hop3 is 3 hops away — should NOT appear
        assert hop3 not in result


class TestDecayFactor:
    """T-055: Decay factor application."""

    def test_higher_decay_produces_higher_far_risk(self) -> None:
        source = uuid4()
        hop1 = uuid4()
        hop2 = uuid4()
        G = _graph((source, hop1, 1.0), (hop1, hop2, 1.0))
        scores = {source: 1.0, hop1: 0.0, hop2: 0.0}

        low_decay = CascadePropagator(threshold=0.70, decay_factor=0.3, max_hops=2)
        high_decay = CascadePropagator(threshold=0.70, decay_factor=0.9, max_hops=2)

        low_result = low_decay.propagate(G, scores)
        high_result = high_decay.propagate(G, scores)

        # With high decay, hop2 relative risk is preserved better
        # hop1 raw same for both; hop2 raw higher with decay=0.9
        # Compare normalized hop2 scores
        low_hop2 = low_result[hop2].cascade_risk
        high_hop2 = high_result[hop2].cascade_risk
        assert high_hop2 >= low_hop2

    def test_zero_decay_no_second_hop(self) -> None:
        source = uuid4()
        hop1 = uuid4()
        hop2 = uuid4()
        G = _graph((source, hop1, 1.0), (hop1, hop2, 1.0))
        scores = {source: 1.0, hop1: 0.0, hop2: 0.0}

        result = CascadePropagator(threshold=0.70, decay_factor=0.0, max_hops=2).propagate(G, scores)

        # hop2 contribution = 1.0 * 1.0 * 0.0^2 = 0.0
        assert result.get(hop2) is None or result[hop2].cascade_risk == pytest.approx(0.0)


class TestSourceAttribution:
    """T-055: Source attribution per node."""

    def test_attribution_lists_source(self) -> None:
        source = uuid4()
        neighbor = uuid4()
        G = _graph((source, neighbor, 1.0))
        scores = {source: 0.9, neighbor: 0.0}

        result = CascadePropagator(threshold=0.70).propagate(G, scores)

        assert source in result[neighbor].cascade_sources

    def test_attribution_multiple_sources(self) -> None:
        s1 = uuid4()
        s2 = uuid4()
        shared = uuid4()
        G = _graph((s1, shared, 1.0), (s2, shared, 1.0))
        scores = {s1: 0.9, s2: 0.8, shared: 0.0}

        result = CascadePropagator(threshold=0.70).propagate(G, scores)

        assert s1 in result[shared].cascade_sources
        assert s2 in result[shared].cascade_sources

    def test_source_node_has_empty_attribution(self) -> None:
        source = uuid4()
        G: nx.DiGraph = nx.DiGraph()
        G.add_node(source)
        scores = {source: 0.9}

        result = CascadePropagator(threshold=0.70).propagate(G, scores)

        assert result[source].cascade_sources == []


class TestNormalization:
    """T-055: Normalization stays in [0, 1]."""

    def test_all_risks_in_unit_interval(self) -> None:
        nodes = [uuid4() for _ in range(5)]
        G: nx.DiGraph = nx.DiGraph()
        # source → all others, plus a chain
        scores = {n: 0.0 for n in nodes}
        scores[nodes[0]] = 0.95  # high-risk source
        for i in range(1, 5):
            G.add_edge(nodes[0], nodes[i], weight=0.7)
        G.add_edge(nodes[1], nodes[2], weight=0.5)

        result = CascadePropagator(threshold=0.70).propagate(G, scores)

        for res in result.values():
            assert 0.0 <= res.cascade_risk <= 1.0, (
                f"cascade_risk {res.cascade_risk} out of [0, 1]"
            )

    def test_max_risk_is_one(self) -> None:
        source = uuid4()
        neighbor = uuid4()
        G = _graph((source, neighbor, 1.0))
        scores = {source: 0.9, neighbor: 0.0}

        result = CascadePropagator(threshold=0.70).propagate(G, scores)
        risks = [r.cascade_risk for r in result.values()]
        assert max(risks) == pytest.approx(1.0)

    def test_empty_graph_returns_source_only(self) -> None:
        source = uuid4()
        G: nx.DiGraph = nx.DiGraph()
        G.add_node(source)
        scores = {source: 0.9}

        result = CascadePropagator(threshold=0.70).propagate(G, scores)

        assert source in result
        assert result[source].cascade_risk == 0.0
        assert len(result) == 1
