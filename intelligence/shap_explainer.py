"""SHAP DeepExplainer for per-node burnout feature attribution."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from uuid import UUID

import numpy as np
import torch
import torch.nn as nn

from intelligence.inference import FEATURE_NAMES

logger = logging.getLogger(__name__)

_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="shap"
)
_BOOTSTRAP_RUNS: int = 5


@dataclass
class ShapExplanation:
    """Per-node SHAP feature attribution output."""

    pseudo_id: UUID
    feature_names: list[str]
    shap_values: list[float]       # mean SHAP per feature; negative=protective, positive=risk
    confidence_low: list[float]    # mean - 2σ across bootstrap runs
    confidence_high: list[float]   # mean + 2σ across bootstrap runs
    base_value: float              # expected model output (background mean)
    model_output: float            # actual model output for this node


class _NodeWrapper(nn.Module):
    """Wrap a GAT so SHAP sees a single-tensor (x → score) interface.

    Edge structure is fixed as context; SHAP attributes output to node features only.
    """

    def __init__(
        self,
        model: nn.Module,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self._model = model
        self._edge_index = edge_index
        self._edge_attr = edge_attr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._model(x, self._edge_index, self._edge_attr)
        if isinstance(out, tuple):
            out = out[0]
        return torch.sigmoid(out)  # [N, 1]


class ShapExplainer:
    """Async SHAP DeepExplainer for per-node burnout feature attribution.

    Uses bootstrap sampling over background subsets to produce per-feature
    SHAP values with 95% confidence intervals.
    """

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,  # [n_bg, n_features]
    ) -> None:
        self._model = model
        self._background = background

    async def explain(
        self,
        pseudo_id: UUID,
        node_features: torch.Tensor,    # [1, n_features] or [n_features]
        edge_index: torch.Tensor,        # [2, E]
        edge_attr: torch.Tensor | None = None,
    ) -> ShapExplanation:
        """Compute per-feature SHAP attributions for one node, non-blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _EXECUTOR,
            self._explain_sync,
            pseudo_id,
            node_features,
            edge_index,
            edge_attr,
        )

    def _explain_sync(
        self,
        pseudo_id: UUID,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> ShapExplanation:
        import shap  # noqa: PLC0415

        if node_features.dim() == 1:
            node_features = node_features.unsqueeze(0)  # [1, F]

        n_features: int = node_features.shape[1]
        wrapped = _NodeWrapper(self._model, edge_index, edge_attr)
        wrapped.eval()

        background = self._background
        bg_size = max(1, len(background) // _BOOTSTRAP_RUNS)

        all_shap: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
        base_val = 0.0

        for i in range(_BOOTSTRAP_RUNS):
            start = (i * bg_size) % len(background)
            end = min(start + bg_size, len(background))
            bg_subset = background[start:end]

            explainer = shap.DeepExplainer(wrapped, bg_subset)
            sv = explainer.shap_values(node_features)
            base_val = float(
                np.array(explainer.expected_value, dtype=np.float64).squeeze()
            )

            if isinstance(sv, list):
                sv = sv[0]
            sv_arr: np.ndarray[tuple[int], np.dtype[np.float64]] = (
                np.array(sv, dtype=np.float64).squeeze()
            )
            if sv_arr.ndim == 0:
                sv_arr = sv_arr.reshape(1)
            all_shap.append(sv_arr[:n_features])

        stacked: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.stack(all_shap)
        mean_sv = stacked.mean(axis=0)
        std_sv = stacked.std(axis=0)

        with torch.no_grad():
            model_out = wrapped(node_features)
        model_output = float(model_out.squeeze().cpu().item())

        logger.debug(
            "SHAP explanation complete: pseudo_id=%s output=%.3f base=%.3f",
            pseudo_id,
            model_output,
            base_val,
        )

        return ShapExplanation(
            pseudo_id=pseudo_id,
            feature_names=FEATURE_NAMES[:n_features],
            shap_values=mean_sv.tolist(),
            confidence_low=(mean_sv - 2.0 * std_sv).tolist(),
            confidence_high=(mean_sv + 2.0 * std_sv).tolist(),
            base_value=base_val,
            model_output=model_output,
        )
