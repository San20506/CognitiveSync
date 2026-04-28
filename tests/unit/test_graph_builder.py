"""Unit tests for intelligence/graph_builder.py — T-042, T-043."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from intelligence.graph_builder import GraphBuilder, FEATURE_NAMES


def _write_csvs(tmp_path: pytest.TempPathFactory, n_employees: int = 10) -> tuple[str, str]:  # type: ignore[type-arg]
    rng = np.random.default_rng(42)
    from intelligence.features import FEATURE_COLS
    ids = [f"{'a' * 8}-{'b' * 4}-{'c' * 4}-{'d' * 4}-{i:012x}" for i in range(n_employees)]

    feat_rows = []
    for pid in ids:
        row = {"pseudo_id": pid, "window_start": "2026-04-01T00:00:00Z", "window_end": "2026-04-15T00:00:00Z"}
        for col in FEATURE_COLS:
            row[col] = float(rng.uniform(0.1, 0.9))
        feat_rows.append(row)
    features_path = str(tmp_path / "features.csv")
    pd.DataFrame(feat_rows).to_csv(features_path, index=False)

    interaction_rows = []
    for i in range(n_employees - 1):
        interaction_rows.append({
            "source_pseudo_id": ids[i],
            "target_pseudo_id": ids[i + 1],
            "interaction_type": "MEETING",
            "weight": float(rng.uniform(0.5, 1.5)),
        })
    interactions_path = str(tmp_path / "interactions.csv")
    pd.DataFrame(interaction_rows).to_csv(interactions_path, index=False)

    return features_path, interactions_path


class TestBuildFromCSV:
    def test_node_count_equals_employee_count(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 15)
        result = GraphBuilder().build_from_csv(fp, ip)
        assert result.nx_graph.number_of_nodes() == 15

    def test_node_ids_length_matches_graph(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 10)
        result = GraphBuilder().build_from_csv(fp, ip)
        assert len(result.node_ids) == result.nx_graph.number_of_nodes()

    def test_edges_present(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 10)
        result = GraphBuilder().build_from_csv(fp, ip)
        assert result.nx_graph.number_of_edges() >= 1

    def test_edge_weights_in_unit_interval(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 10)
        result = GraphBuilder().build_from_csv(fp, ip)
        for _, _, attrs in result.nx_graph.edges(data=True):
            assert 0.0 <= attrs["weight"] <= 1.0


class TestToPyG:
    def test_x_shape(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 12)
        result = GraphBuilder().build_from_csv(fp, ip)
        assert result.pyg_data.x.shape == (12, len(FEATURE_NAMES))

    def test_x_dtype_float32(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 8)
        result = GraphBuilder().build_from_csv(fp, ip)
        assert result.pyg_data.x.dtype == torch.float32

    def test_x_values_in_unit_interval(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 8)
        result = GraphBuilder().build_from_csv(fp, ip)
        assert float(result.pyg_data.x.min()) >= 0.0
        assert float(result.pyg_data.x.max()) <= 1.0

    def test_edge_index_shape(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 10)
        result = GraphBuilder().build_from_csv(fp, ip)
        assert result.pyg_data.edge_index.shape[0] == 2

    def test_edge_attr_shape(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 10)
        result = GraphBuilder().build_from_csv(fp, ip)
        E = result.pyg_data.edge_index.shape[1]
        assert result.pyg_data.edge_attr.shape == (E, 1)

    def test_no_nan_in_tensors(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, ip = _write_csvs(tmp_path, 10)
        result = GraphBuilder().build_from_csv(fp, ip)
        assert not torch.isnan(result.pyg_data.x).any()
        assert not torch.isnan(result.pyg_data.edge_attr).any()

    def test_empty_edges_produces_zero_edge_index(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        fp, _ = _write_csvs(tmp_path, 5)
        # Write empty interactions
        ip = str(tmp_path / "empty.csv")
        pd.DataFrame(columns=["source_pseudo_id", "target_pseudo_id", "interaction_type", "weight"]).to_csv(ip, index=False)
        result = GraphBuilder().build_from_csv(fp, ip)
        assert result.pyg_data.edge_index.shape == (2, 0)
