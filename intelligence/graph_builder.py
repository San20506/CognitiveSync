"""NetworkX graph builder + PyG converter — implementation in Phase 4B (T-042, T-043).

Constructs a weighted directed graph from the feature store after each pipeline run.
Nodes: pseudonymous employees (UUID). Edges: collaboration relationships.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import networkx as nx
import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from torch_geometric.data import Data

from ingestion.db.models import EdgeSignal, FeatureVector

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
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


@dataclass
class BuiltGraph:
    """Container for the constructed graph and its PyG representation."""

    nx_graph: nx.DiGraph
    node_ids: list[UUID]  # Index-aligned with PyG node tensors
    pyg_data: object  # torch_geometric.data.Data — imported lazily to avoid torch dep at import


class GraphBuilder:
    """Builds org collaboration graph from feature store.

    Phase 4B implementation target: T-042, T-043.

    Architecture (per ARCHITECTURE.md §4.1):
    - Nodes: one per pseudo_id, attrs = 13-dim feature vector
    - Edges: directed (source → target), attr = interaction weight [0,1]
    - Orphan nodes (no interactions): retained as isolated nodes for GNN
    - PyG conversion: manual tensor construction preserving FEATURE_NAMES order
    """

    async def build_from_store(
        self,
        window_end: datetime,
        db: AsyncSession,
    ) -> BuiltGraph:
        """Load feature vectors and edge signals from DB, build graph.

        Implements T-042.
        """
        # Query FeatureVectors for the given window_end
        fv_result = await db.execute(
            select(FeatureVector).where(FeatureVector.window_end == window_end)
        )
        feature_vectors: list[FeatureVector] = list(fv_result.scalars().all())
        logger.debug(
            "Loaded %d feature vectors for window_end=%s",
            len(feature_vectors),
            window_end,
        )

        # Query EdgeSignals for the given window_end
        es_result = await db.execute(
            select(EdgeSignal).where(EdgeSignal.window_end == window_end)
        )
        edge_signals: list[EdgeSignal] = list(es_result.scalars().all())
        logger.debug(
            "Loaded %d edge signals for window_end=%s",
            len(edge_signals),
            window_end,
        )

        G: nx.DiGraph = nx.DiGraph()

        # Add nodes with feature attrs; collect node_ids in insertion order
        node_ids: list[UUID] = []
        for fv in feature_vectors:
            attrs: dict[str, float] = fv.feature_json if fv.feature_json else {}
            G.add_node(fv.pseudo_id, **attrs)
            node_ids.append(fv.pseudo_id)

        # Add edges
        for es in edge_signals:
            G.add_edge(es.source_pseudo_id, es.target_pseudo_id, weight=es.weight)

        logger.debug(
            "Built graph: %d nodes, %d edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )

        pyg_data = self.to_pyg(G, node_ids)

        return BuiltGraph(nx_graph=G, node_ids=node_ids, pyg_data=pyg_data)

    def to_pyg(self, G: nx.DiGraph, node_ids: list[UUID]) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object.

        Returns Data(x: Tensor[N×13], edge_index: Tensor[2×E], edge_attr: Tensor[E×1])

        Implements T-043.
        """
        node_index: dict[UUID, int] = {nid: i for i, nid in enumerate(node_ids)}

        # Build node feature matrix x: shape [N, 13] float32
        rows: list[list[float]] = []
        for nid in node_ids:
            attrs = G.nodes[nid]
            row = [float(attrs.get(feat, 0.5)) for feat in FEATURE_NAMES]
            rows.append(row)

        x = torch.tensor(rows, dtype=torch.float32)  # [N, 13]

        # Build edge_index [2, E] and edge_attr [E, 1]
        src_indices: list[int] = []
        dst_indices: list[int] = []
        weights: list[float] = []

        for src, dst, data in G.edges(data=True):
            src_indices.append(node_index[src])
            dst_indices.append(node_index[dst])
            weights.append(float(data.get("weight", 1.0)))

        if src_indices:
            edge_index = torch.tensor(
                [src_indices, dst_indices], dtype=torch.long
            )  # [2, E]
            edge_attr = torch.tensor(
                [[w] for w in weights], dtype=torch.float32
            )  # [E, 1]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)

        logger.debug(
            "PyG tensors: x=%s, edge_index=%s, edge_attr=%s",
            tuple(x.shape),
            tuple(edge_index.shape),
            tuple(edge_attr.shape),
        )

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
