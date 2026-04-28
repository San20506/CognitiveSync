"""Unit tests for intelligence/edges.py — T-039."""

from __future__ import annotations

import pandas as pd
import pytest

from intelligence.edges import load_edges


def _make_interactions(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _csv(tmp_path: pytest.TempPathFactory, rows: list[dict]) -> str:  # type: ignore[type-arg]
    df = _make_interactions(rows)
    path = str(tmp_path / "interactions.csv")
    df.to_csv(path, index=False)
    return path


_BASE_ROW = dict(
    source_pseudo_id="aaa",
    target_pseudo_id="bbb",
    interaction_type="MEETING",
    weight=1.0,
)


class TestLoadEdges:
    def test_returns_source_target_weight_columns(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        path = _csv(tmp_path, [_BASE_ROW])
        df = load_edges(path)
        assert set(df.columns) == {"source", "target", "weight"}

    def test_weights_in_unit_interval(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        rows = [
            dict(source_pseudo_id="a", target_pseudo_id="b", interaction_type="MEETING", weight=5.0),
            dict(source_pseudo_id="c", target_pseudo_id="d", interaction_type="MEETING", weight=10.0),
        ]
        df = load_edges(_csv(tmp_path, rows))
        assert (df["weight"] >= 0).all() and (df["weight"] <= 1).all()

    def test_undirected_merge_keeps_max_weight(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        rows = [
            dict(source_pseudo_id="a", target_pseudo_id="b", interaction_type="SLACK_DM", weight=3.0),
            dict(source_pseudo_id="b", target_pseudo_id="a", interaction_type="SLACK_DM", weight=1.0),
        ]
        df = load_edges(_csv(tmp_path, rows))
        # Should collapse to one undirected edge
        assert len(df) == 1

    def test_multiple_interaction_types_normalised_independently(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        rows = [
            dict(source_pseudo_id="a", target_pseudo_id="b", interaction_type="MEETING", weight=10.0),
            dict(source_pseudo_id="c", target_pseudo_id="d", interaction_type="GITHUB_PR", weight=2.0),
        ]
        df = load_edges(_csv(tmp_path, rows))
        # Both edges are the max for their type — should normalise to ≈1.0
        assert (df["weight"] > 0.999).all()

    def test_no_self_loops_after_undirected_merge(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        rows = [_BASE_ROW]
        df = load_edges(_csv(tmp_path, rows))
        for _, row in df.iterrows():
            assert row["source"] != row["target"]

    def test_empty_interactions_returns_empty_df(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        path = str(tmp_path / "interactions.csv")
        pd.DataFrame(columns=["source_pseudo_id", "target_pseudo_id", "interaction_type", "weight"]).to_csv(path, index=False)
        df = load_edges(path)
        assert len(df) == 0
