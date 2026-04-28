"""Unit tests for intelligence/gnn_model.py — T-045, T-051."""

from __future__ import annotations

import torch
import pytest

from intelligence.gnn_model import BurnoutGAT, SmallBurnoutGAT


def _minimal_graph(n: int = 8, in_channels: int = 13) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, in_channels)
    src = torch.arange(n - 1)
    dst = torch.arange(1, n)
    edge_index = torch.stack([src, dst])
    return x, edge_index


class TestBurnoutGAT:
    def test_forward_output_shape(self) -> None:
        model = BurnoutGAT()
        model.eval()
        x, ei = _minimal_graph(8, 13)
        out = model(x, ei)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (8, 1)

    def test_forward_return_attention(self) -> None:
        model = BurnoutGAT()
        model.eval()
        x, ei = _minimal_graph(8, 13)
        out = model(x, ei, return_attention=True)
        assert isinstance(out, tuple)
        scores, alpha = out
        assert scores.shape == (8, 1)
        assert alpha.shape[1] == 2  # HEADS_2

    def test_mc_dropout_returns_three_tensors(self) -> None:
        model = BurnoutGAT()
        model.eval()
        x, ei = _minimal_graph(8, 13)
        mean, low, high = model.mc_dropout_predict(x, ei, n_passes=3)
        assert mean.shape == low.shape == high.shape == (8, 1)

    def test_mc_dropout_mean_in_unit_interval(self) -> None:
        model = BurnoutGAT()
        x, ei = _minimal_graph(8, 13)
        mean, _, _ = model.mc_dropout_predict(x, ei, n_passes=3)
        assert float(mean.min()) >= 0.0
        assert float(mean.max()) <= 1.0

    def test_mc_dropout_low_le_mean_le_high(self) -> None:
        model = BurnoutGAT()
        x, ei = _minimal_graph(10, 13)
        mean, low, high = model.mc_dropout_predict(x, ei, n_passes=5)
        assert (low <= mean + 1e-5).all()
        assert (mean <= high + 1e-5).all()

    def test_no_nan_in_output(self) -> None:
        model = BurnoutGAT()
        model.eval()
        x, ei = _minimal_graph(8, 13)
        out = model(x, ei)
        assert not torch.isnan(out).any()


class TestSmallBurnoutGAT:
    def test_forward_output_shape(self) -> None:
        model = SmallBurnoutGAT()
        model.eval()
        x, ei = _minimal_graph(8, 10)
        out = model(x, ei)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (8, 1)

    def test_forward_return_attention(self) -> None:
        model = SmallBurnoutGAT()
        model.eval()
        x, ei = _minimal_graph(8, 10)
        out = model(x, ei, return_attention=True)
        assert isinstance(out, tuple)
        scores, alpha = out
        assert scores.shape == (8, 1)

    def test_mc_dropout_output_shapes(self) -> None:
        model = SmallBurnoutGAT()
        x, ei = _minimal_graph(8, 10)
        mean, low, high = model.mc_dropout_predict(x, ei, n_passes=3)
        assert mean.shape == (8, 1)

    def test_mc_dropout_restores_eval_mode(self) -> None:
        model = SmallBurnoutGAT()
        model.eval()
        x, ei = _minimal_graph(8, 10)
        model.mc_dropout_predict(x, ei, n_passes=3)
        assert not model.training

    def test_loads_checkpoint(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        model = SmallBurnoutGAT()
        ckpt = tmp_path / "model.pt"
        torch.save(model.state_dict(), ckpt)
        loaded = SmallBurnoutGAT()
        loaded.load_state_dict(torch.load(ckpt, weights_only=True))
        loaded.eval()
        x, ei = _minimal_graph(5, 10)
        out = loaded(x, ei)
        assert out.shape == (5, 1)

    def test_no_nan_output(self) -> None:
        model = SmallBurnoutGAT()
        model.eval()
        x, ei = _minimal_graph(8, 10)
        out = model(x, ei)
        assert not torch.isnan(out).any()
