"""Cascade propagation module — implementation in Phase 4B (T-053, T-054).

Algorithm (per ARCHITECTURE.md §4.3 and WORKFLOWS.md WF-06):

    for each node v where burnout_score(v) > THRESHOLD:
        for each neighbor u of v (up to MAX_HOPS):
            cascade_risk(u) += burnout_score(v) * edge_weight(v, u) * DECAY_FACTOR^hop
    normalize cascade_risk to [0, 1]

Defaults: THRESHOLD=0.70, DECAY_FACTOR=0.60, MAX_HOPS=2
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from uuid import UUID

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class CascadeResult:
    """Cascade risk score and source attribution for a single node."""

    pseudo_id: UUID
    cascade_risk: float          # [0, 1] — normalized cascade contribution received
    cascade_sources: list[UUID] = field(default_factory=list)
    # Which high-risk nodes contributed to this node's cascade risk


class CascadePropagator:
    """2-hop cascade risk propagation over the org collaboration graph.

    Phase 4B implementation target: T-053, T-054.
    """

    def __init__(
        self,
        threshold: float = 0.70,
        decay_factor: float = 0.60,
        max_hops: int = 2,
    ) -> None:
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.max_hops = max_hops

    def propagate(
        self,
        nx_graph: object,           # networkx.DiGraph
        scores: dict[UUID, float],  # burnout_score per node
    ) -> dict[UUID, CascadeResult]:
        """Compute cascade risk for all neighbors of high-risk nodes.

        Returns mapping of pseudo_id → CascadeResult for all affected nodes.
        High-risk source nodes also included with their own cascade_risk=0.0
        (they ARE the source, not the receiver).

        T-053, T-054: Cascade propagation implementation.
        """
        graph: nx.DiGraph = nx_graph  # type: ignore[assignment]

        cascade_risk: dict[UUID, float] = {}
        cascade_sources: dict[UUID, list[UUID]] = {}

        high_risk_nodes: set[UUID] = {v for v, s in scores.items() if s > self.threshold}

        for source in high_risk_nodes:
            if source not in graph:
                continue

            # BFS from source up to max_hops
            visited: dict[UUID, int] = {source: 0}
            queue: deque[tuple[UUID, int]] = deque([(source, 0)])

            while queue:
                current, hop = queue.popleft()
                if hop >= self.max_hops:
                    continue
                for neighbor in graph.successors(current):
                    next_hop = hop + 1
                    if neighbor not in visited:
                        visited[neighbor] = next_hop
                        queue.append((neighbor, next_hop))

                        edge_w: float = graph[current][neighbor].get("weight", 1.0)
                        contribution = scores[source] * edge_w * (self.decay_factor ** next_hop)
                        cascade_risk[neighbor] = cascade_risk.get(neighbor, 0.0) + contribution

                        if neighbor not in cascade_sources:
                            cascade_sources[neighbor] = []
                        if source not in cascade_sources[neighbor]:
                            cascade_sources[neighbor].append(source)

        # Normalize cascade_risk to [0, 1]
        max_risk: float = max(cascade_risk.values(), default=1.0)
        if max_risk > 0.0:
            cascade_risk = {node: risk / max_risk for node, risk in cascade_risk.items()}

        # Build result dict — include high-risk sources with cascade_risk=0.0
        result: dict[UUID, CascadeResult] = {}

        for source in high_risk_nodes:
            result[source] = CascadeResult(
                pseudo_id=source,
                cascade_risk=0.0,
                cascade_sources=[],
            )

        for node, risk in cascade_risk.items():
            result[node] = CascadeResult(
                pseudo_id=node,
                cascade_risk=risk,
                cascade_sources=cascade_sources.get(node, []),
            )

        logger.debug(
            "Cascade propagation complete: %d source nodes, %d affected nodes",
            len(high_risk_nodes),
            len(cascade_risk),
        )

        return result
