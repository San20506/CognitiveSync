"""Unit tests for intelligence/shap_explainer.py."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
import torch

from intelligence.gnn_model import SmallBurnoutGAT
from intelligence.inference import FEATURE_NAMES
from intelligence.shap_explainer import (
    ShapExplanation,
    ShapExplainer,
    _NodeWrapper,
)

_N = len(FEATURE_NAMES)  # 10


def _make_model() -> SmallBurnoutGAT:
    m = SmallBurnoutGAT()
    m.eval()
    return m


def _make_background(n: int = 20) -> torch.Tensor:
    return torch.rand(n, _N) * 0.5 + 0.25


def _make_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """Single-node self-loop graph — minimal valid graph for GAT."""
    node_features = torch.rand(1, _N)
    edge_index = torch.zeros(2, 1, dtype=torch.long)
    return node_features, edge_index


def _mock_shap_module(shap_value: float = 0.05) -> MagicMock:
    """Return a mock shap module whose DeepExplainer returns predictable values."""
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = (
        np.full((1, _N), shap_value, dtype=np.float64)
    )
    mock_explainer.expected_value = np.array([0.42])

    mock_shap = MagicMock()
    mock_shap.DeepExplainer.return_value = mock_explainer
    return mock_shap


# ---------------------------------------------------------------------------
# _NodeWrapper tests
# ---------------------------------------------------------------------------


class TestNodeWrapper:
    def test_forward_returns_sigmoid_tensor(self) -> None:
        model = _make_model()
        edge_index = torch.zeros(2, 1, dtype=torch.long)
        wrapper = _NodeWrapper(model, edge_index, None)
        x = torch.rand(1, _N)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 1
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_forward_unwraps_tuple_output(self) -> None:
        """Wrapper must handle models that return (logits, attention) tuples."""
        model = _make_model()
        edge_index = torch.zeros(2, 1, dtype=torch.long)
        wrapper = _NodeWrapper(model, edge_index, None)

        with patch.object(
            model,
            "forward",
            return_value=(torch.zeros(1, 1), torch.zeros(1, 1)),
        ):
            out = wrapper(torch.rand(1, _N))
        assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# ShapExplainer._explain_sync tests (sync, with mocked shap)
# ---------------------------------------------------------------------------


class TestShapExplainerSync:
    def _run(
        self,
        shap_value: float = 0.05,
        shap_shape: tuple[int, ...] | None = None,
    ) -> ShapExplanation:
        mock_shap = _mock_shap_module(shap_value)
        if shap_shape is not None:
            mock_shap.DeepExplainer.return_value.shap_values.return_value = (
                np.full(shap_shape, shap_value, dtype=np.float64)
            )

        model = _make_model()
        background = _make_background()
        explainer = ShapExplainer(model=model, background=background)
        node_features, edge_index = _make_graph()
        pseudo_id = uuid4()

        with patch.dict("sys.modules", {"shap": mock_shap}):
            return explainer._explain_sync(pseudo_id, node_features, edge_index, None)

    def test_returns_shap_explanation(self) -> None:
        result = self._run()
        assert isinstance(result, ShapExplanation)

    def test_feature_names_match_canonical(self) -> None:
        result = self._run()
        assert result.feature_names == FEATURE_NAMES[:_N]

    def test_shap_values_length_matches_features(self) -> None:
        result = self._run()
        assert len(result.shap_values) == _N
        assert len(result.confidence_low) == _N
        assert len(result.confidence_high) == _N

    def test_confidence_bounds_contain_mean(self) -> None:
        result = self._run(shap_value=0.1)
        for sv, lo, hi in zip(result.shap_values, result.confidence_low, result.confidence_high):
            assert lo <= sv + 1e-9
            assert sv <= hi + 1e-9

    def test_base_value_from_expected_value(self) -> None:
        result = self._run()
        # Expected value set to 0.42 in the mock
        assert abs(result.base_value - 0.42) < 1e-6

    def test_model_output_in_unit_interval(self) -> None:
        result = self._run()
        assert 0.0 <= result.model_output <= 1.0

    def test_pseudo_id_preserved(self) -> None:
        pseudo_id = uuid4()
        mock_shap = _mock_shap_module()
        model = _make_model()
        explainer = ShapExplainer(model=model, background=_make_background())
        node_features, edge_index = _make_graph()

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = explainer._explain_sync(pseudo_id, node_features, edge_index, None)

        assert result.pseudo_id == pseudo_id

    def test_1d_node_features_accepted(self) -> None:
        """explain_sync should handle unsqueezed [F] input."""
        mock_shap = _mock_shap_module()
        model = _make_model()
        explainer = ShapExplainer(model=model, background=_make_background())
        node_features = torch.rand(_N)  # [F], not [1, F]
        edge_index = torch.zeros(2, 1, dtype=torch.long)

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = explainer._explain_sync(uuid4(), node_features, edge_index, None)

        assert len(result.shap_values) == _N

    def test_list_shap_values_unwrapped(self) -> None:
        """DeepExplainer sometimes returns list[ndarray] — must be unwrapped."""
        mock_shap = _mock_shap_module()
        inner_arr = np.full((1, _N), 0.07, dtype=np.float64)
        mock_shap.DeepExplainer.return_value.shap_values.return_value = [inner_arr]

        model = _make_model()
        explainer = ShapExplainer(model=model, background=_make_background())
        node_features, edge_index = _make_graph()

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = explainer._explain_sync(uuid4(), node_features, edge_index, None)

        assert len(result.shap_values) == _N


# ---------------------------------------------------------------------------
# ShapExplainer.explain async test
# ---------------------------------------------------------------------------


class TestShapExplainerAsync:
    def test_explain_returns_shap_explanation(self) -> None:
        mock_shap = _mock_shap_module(0.03)
        model = _make_model()
        explainer = ShapExplainer(model=model, background=_make_background())
        node_features, edge_index = _make_graph()

        async def _run() -> ShapExplanation:
            with patch.dict("sys.modules", {"shap": mock_shap}):
                return await explainer.explain(uuid4(), node_features, edge_index)

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert isinstance(result, ShapExplanation)
        assert len(result.shap_values) == _N
