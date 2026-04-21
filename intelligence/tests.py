"""
CognitiveSync — Unit Tests
T-041: Validate feature extractor (all 13 features, shape, values)
T-044: Validate graph builder (node count, edge weights, PyG shape)
"""

from __future__ import annotations

import numpy as np
import torch
import networkx as nx
import pandas as pd
from torch_geometric.data import Data

from features import FEATURE_COLS


def test_features(features_df: pd.DataFrame) -> None:
    """
    T-041 — Validate the feature matrix.

    Checks:
        - All 13 feature columns are present
        - All values are in [0, 1]
        - No NaN or Inf values
        - No feature column has zero variance
    """
    print("[T-041] Feature extractor tests")

    # All 13 columns present
    for col in FEATURE_COLS:
        assert col in features_df.columns, f"Missing feature column: {col}"
    print("  ✓ All 13 feature columns present")

    feat_matrix = features_df[FEATURE_COLS].values

    # Shape check
    assert feat_matrix.shape[1] == 13, (
        f"Expected 13 features, got {feat_matrix.shape[1]}"
    )
    print(f"  ✓ Feature matrix shape: {feat_matrix.shape}")

    # Value range
    assert (feat_matrix >= 0).all(), "Feature values below 0 found"
    assert (feat_matrix <= 1).all(), "Feature values above 1 found"
    print("  ✓ All values in [0, 1]")

    # No NaN / Inf
    assert not np.isnan(feat_matrix).any(), "NaN found in feature matrix"
    assert not np.isinf(feat_matrix).any(), "Inf found in feature matrix"
    print("  ✓ No NaN or Inf values")

    # Non-zero variance per feature
    for col in FEATURE_COLS:
        assert features_df[col].std() > 0, (
            f"Feature '{col}' has zero variance — check computation"
        )
    print("  ✓ All features have non-zero variance")


def test_graph(
    G: nx.Graph,
    employees_df: pd.DataFrame,
) -> None:
    """
    T-044 — Validate the NetworkX graph structure.

    Checks:
        - Node count equals number of employees
        - All nodes have a feature vector of length 13
        - All edge weights are in [0, 1]
        - Edge count is plausible
    """
    print("[T-044] Graph builder tests")

    N = len(employees_df)

    # Node count
    assert G.number_of_nodes() == N, (
        f"Node count mismatch: graph={G.number_of_nodes()}, expected={N}"
    )
    print(f"  ✓ Node count = {N}")

    # Node feature vectors
    for node_id, attrs in G.nodes(data=True):
        assert "x" in attrs, f"Node {node_id} missing feature vector 'x'"
        assert len(attrs["x"]) == 13, (
            f"Node {node_id} feature vector has length {len(attrs['x'])}, expected 13"
        )
    print("  ✓ All nodes have feature vectors of shape (13,)")

    # Edge weights
    for u, v, attrs in G.edges(data=True):
        w = attrs.get("weight")
        assert w is not None, f"Edge ({u}, {v}) missing 'weight' attribute"
        assert 0 <= w <= 1, f"Edge ({u}, {v}) weight {w} out of [0, 1]"
    print(f"  ✓ All {G.number_of_edges()} edge weights in [0, 1]")

    # Plausibility check
    assert G.number_of_edges() >= 10, (
        f"Only {G.number_of_edges()} edges — suspiciously low, check interaction data"
    )
    print(f"  ✓ Edge count {G.number_of_edges()} is plausible")


def test_pyg(data: Data, employees_df: pd.DataFrame) -> None:
    """
    T-043 — Validate the PyG Data object Sandy's GNN will receive.

    Checks:
        - data.x shape is (N, 13)
        - data.x values are in [0, 1]
        - data.edge_index is COO format shape (2, E)
        - data.edge_attr shape is (E, 1)
        - No NaN in any tensor
        - Node count preserved through conversion
    """
    print("[T-043] PyG Data object tests")

    N = len(employees_df)
    E = data.edge_index.shape[1]

    # data.x shape
    assert data.x.shape == (N, 13), (
        f"data.x shape {tuple(data.x.shape)}, expected ({N}, 13)"
    )
    print(f"  ✓ data.x shape: {tuple(data.x.shape)}")

    # data.x value range
    assert float(data.x.min()) >= 0 and float(data.x.max()) <= 1, (
        "data.x values outside [0, 1]"
    )
    print("  ✓ data.x values in [0, 1]")

    # edge_index shape
    assert data.edge_index.shape[0] == 2, (
        "edge_index must have 2 rows (COO format)"
    )
    print(f"  ✓ data.edge_index shape: {tuple(data.edge_index.shape)}")

    # edge_attr shape
    assert data.edge_attr.shape == (E, 1), (
        f"data.edge_attr shape {tuple(data.edge_attr.shape)}, expected ({E}, 1)"
    )
    print(f"  ✓ data.edge_attr shape: {tuple(data.edge_attr.shape)}")

    # No NaN
    assert not torch.isnan(data.x).any(), "NaN in data.x"
    assert not torch.isnan(data.edge_attr).any(), "NaN in data.edge_attr"
    print("  ✓ No NaN in any tensor")

    # Node count preserved
    assert data.x.shape[0] == N, "Node count changed during PyG conversion"
    print(f"  ✓ Node count preserved through conversion ({N} nodes)")


def run_all_tests(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    employees_df: pd.DataFrame,
    G: nx.Graph,
    data: Data,
) -> None:
    """
    Run all unit tests for T-041, T-044, and T-043.
    Raises AssertionError on the first failure.

    Args:
        features_df:  Output of features.load_features()
        edges_df:     Output of edges.load_edges()
        employees_df: Raw employees DataFrame
        G:            Output of graph_builder.build_graph()
        data:         Output of pyg_converter.convert_to_pyg()
    """
    print("=" * 55)
    print("  CognitiveSync — Unit Tests")
    print("=" * 55)

    test_features(features_df)
    print()
    test_graph(G, employees_df)
    print()
    test_pyg(data, employees_df)

    print()
    print("=" * 55)
    print("  ALL TESTS PASSED ✓")
    print("=" * 55)