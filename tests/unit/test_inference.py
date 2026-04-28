"""Unit tests for intelligence/inference.py — T-051, T-052."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from intelligence.gnn_model import SmallBurnoutGAT
from intelligence.inference import InferencePipeline, NodeScore, ScoredGraph


def _make_pyg(n: int = 10) -> Data:
    x = torch.rand(n, 10)
    src = torch.arange(n - 1)
    dst = torch.arange(1, n)
    edge_index = torch.stack([src, dst])
    edge_attr = torch.rand(n - 1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _make_pipeline(tmp_path: Path) -> tuple[InferencePipeline, list[UUID]]:
    """Save a SmallBurnoutGAT checkpoint and build a ready InferencePipeline."""
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    model = SmallBurnoutGAT()
    torch.save(model.state_dict(), version_dir / "model.pt")
    meta = {"architecture": "SmallBurnoutGAT (10→64→16→1)", "n_features": 10}
    (version_dir / "metrics.json").write_text(json.dumps(meta))
    latest = tmp_path / "latest"
    latest.symlink_to(version_dir)

    pipe = InferencePipeline(tmp_path, device="cpu")
    pipe.load_model("latest")
    node_ids = [uuid4() for _ in range(10)]
    return pipe, node_ids


class TestInferencePipeline:
    def test_score_returns_scored_graph(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        pipe, ids = _make_pipeline(tmp_path)
        data = _make_pyg(10)
        result = pipe.score(data, ids, uuid4())
        assert isinstance(result, ScoredGraph)

    def test_score_count_equals_node_count(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        pipe, ids = _make_pipeline(tmp_path)
        result = pipe.score(_make_pyg(10), ids, uuid4())
        assert len(result.node_scores) == 10

    def test_burnout_scores_in_unit_interval(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        pipe, ids = _make_pipeline(tmp_path)
        result = pipe.score(_make_pyg(10), ids, uuid4())
        for ns in result.node_scores.values():
            assert 0.0 <= ns.burnout_score <= 1.0

    def test_confidence_low_le_score_le_high(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        pipe, ids = _make_pipeline(tmp_path)
        result = pipe.score(_make_pyg(10), ids, uuid4())
        for ns in result.node_scores.values():
            assert ns.confidence_low <= ns.burnout_score + 1e-5
            assert ns.burnout_score <= ns.confidence_high + 1e-5

    def test_top_features_has_three_entries(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        pipe, ids = _make_pipeline(tmp_path)
        result = pipe.score(_make_pyg(10), ids, uuid4())
        for ns in result.node_scores.values():
            assert len(ns.top_features) == 3

    def test_top_feature_keys_are_valid_names(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        from intelligence.inference import FEATURE_NAMES
        pipe, ids = _make_pipeline(tmp_path)
        result = pipe.score(_make_pyg(10), ids, uuid4())
        for ns in result.node_scores.values():
            for k in ns.top_features:
                assert k in FEATURE_NAMES

    def test_run_id_preserved(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        pipe, ids = _make_pipeline(tmp_path)
        run_id = uuid4()
        result = pipe.score(_make_pyg(10), ids, run_id)
        assert result.run_id == run_id

    def test_raises_if_model_not_loaded(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        pipe = InferencePipeline(tmp_path, device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            pipe.score(_make_pyg(5), [uuid4() for _ in range(5)], uuid4())

    def test_load_model_raises_if_no_checkpoint(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        pipe = InferencePipeline(tmp_path, device="cpu")
        with pytest.raises(FileNotFoundError):
            pipe.load_model("latest")
