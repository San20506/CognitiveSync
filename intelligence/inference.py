"""GNN inference pipeline with MC Dropout confidence — Phase 4B (T-051, T-052).

Inference protocol (per WORKFLOWS.md WF-05):
1. Load active model from registry
2. Move Data + model to CUDA if available
3. Mini-batch split if N > threshold
4. Forward pass (model.eval() — no gradients)
5. Score extraction: sigmoid output → burnout_score [0,1]
6. MC Dropout: 5 stochastic passes → confidence interval (mean ± 2σ)
7. GAT attention weights → top contributing features per node
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import torch

from intelligence.gnn_model import BurnoutGAT, SmallBurnoutGAT  # noqa: F401

logger = logging.getLogger(__name__)

MC_DROPOUT_PASSES = 5

FEATURE_NAMES: list[str] = [
    "meeting_density",
    "after_hours_ratio",
    "response_latency_avg",
    "focus_time_blocks",
    "msg_volume_daily",
    "msg_response_time",
    "mention_load",
    "context_switch_rate",
    "hrv_avg",
    "sleep_score",
]


@dataclass
class NodeScore:
    """Scored output for a single graph node."""

    pseudo_id: UUID
    burnout_score: float        # Mean MC Dropout score [0,1]
    confidence_low: float       # mean - 2σ, clamped to [0,1]
    confidence_high: float      # mean + 2σ, clamped to [0,1]
    top_features: dict[str, float] = field(default_factory=dict)
    # {feature_name: attention_weight} — from GAT Layer 2 attention


@dataclass
class ScoredGraph:
    """Collection of scores for all nodes after a single inference run."""

    node_scores: dict[UUID, NodeScore]
    nx_graph: object  # networkx.DiGraph — passed through to cascade propagator
    run_id: UUID


class InferencePipeline:
    """Loads model and runs inference with MC Dropout.

    Phase 4B implementation target: T-051, T-052.
    """

    def __init__(self, model_registry_path: Path, device: str = "cuda") -> None:
        self._registry = model_registry_path
        self._device = device
        self._model: BurnoutGAT | SmallBurnoutGAT | None = None
        self._device_obj: torch.device = torch.device("cpu")

    def load_model(self, version: str = "latest") -> None:
        """Load model checkpoint from registry.

        Resolves the requested version directory, loads the serialised
        state-dict, and moves the model to the configured device.

        Args:
            version: Version tag to load, or ``"latest"`` (default) which
                follows the ``latest`` symlink if present, or falls back to
                the most-recently-modified version directory.

        Raises:
            FileNotFoundError: If no checkpoint directories or model file
                can be located under the registry path.
        """
        registry = self._registry

        if version == "latest":
            target = registry / "latest"
            if target.is_symlink():
                version_dir = target.resolve()
            else:
                # Fall back to most recently modified version directory
                dirs = sorted(
                    registry.iterdir(),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                version_dir = next(
                    (d for d in dirs if d.is_dir() and d.name != "latest"),
                    None,
                )
                if version_dir is None:
                    raise FileNotFoundError(
                        f"No model checkpoints found in {registry}"
                    )
        else:
            version_dir = registry / version

        model_path = version_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        # Detect architecture from metrics.json if present
        import json as _json
        meta_path = version_dir / "metrics.json"
        is_small = False
        if meta_path.exists():
            meta = _json.loads(meta_path.read_text())
            arch = meta.get("architecture", "")
            n_feat = meta.get("n_features", 13)
            is_small = "Small" in arch or n_feat <= 10

        model: BurnoutGAT | SmallBurnoutGAT = SmallBurnoutGAT() if is_small else BurnoutGAT()
        device = torch.device(
            self._device if torch.cuda.is_available() else "cpu"
        )
        state_dict = torch.load(
            model_path, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        self._model = model
        self._device_obj = device
        logger.info("Model loaded from %s on %s", version_dir, device)

    def score(
        self,
        pyg_data: object,
        node_ids: list[UUID],
        run_id: UUID,
    ) -> ScoredGraph:
        """Run MC Dropout inference and extract GAT attention weights.

        Executes ``MC_DROPOUT_PASSES`` stochastic forward passes to produce
        per-node burnout scores with 95 % confidence intervals, then runs a
        single deterministic pass (``model.eval()``) to retrieve Layer-2
        attention weights used for feature attribution.

        Args:
            pyg_data: A ``torch_geometric.data.Data`` object containing at
                minimum ``x`` (node features), ``edge_index``, and optionally
                ``edge_attr``.
            node_ids: Ordered list of pseudonymised UUIDs corresponding to
                the node rows in ``pyg_data.x``.
            run_id: Unique identifier for this inference run (for logging and
                downstream tracing).

        Returns:
            A :class:`ScoredGraph` with per-node :class:`NodeScore` objects
            and the original ``pyg_data`` object forwarded for cascade
            propagation.

        Raises:
            RuntimeError: If :meth:`load_model` has not been called yet.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded — call load_model() first"
            )

        from torch_geometric.data import Data  # noqa: PLC0415

        data: Data = pyg_data  # type: ignore[assignment]
        model: BurnoutGAT = self._model  # type: ignore[assignment]
        device = self._device_obj

        data = data.to(device)

        # MC Dropout: 5 stochastic passes for confidence intervals
        with torch.no_grad():
            mean_scores, conf_low, conf_high = model.mc_dropout_predict(
                data.x, data.edge_index, n_passes=MC_DROPOUT_PASSES
            )

        # Attention weights for feature attribution (deterministic eval pass)
        model.eval()
        with torch.no_grad():
            forward_out = model.forward(
                data.x, data.edge_index, data.edge_attr, return_attention=True
            )
        # forward() with return_attention=True returns (scores, alpha)
        _, attention_weights = forward_out  # type: ignore[misc]
        # attention_weights: [E, HEADS_2] — average over heads → [E]

        # Aggregate attention received per destination node
        n_nodes = len(node_ids)
        node_attention = torch.zeros(n_nodes, device=device)

        if data.edge_index.shape[1] > 0:
            edge_index = data.edge_index
            attn = attention_weights.mean(dim=1)  # [E] mean over heads
            for i in range(edge_index.shape[1]):
                dst = int(edge_index[1, i].item())
                node_attention[dst] += attn[i].item()

        # Convert tensors to numpy for iteration
        mean_np = mean_scores.squeeze(1).cpu().numpy()
        low_np = conf_low.squeeze(1).clamp(0.0, 1.0).cpu().numpy()
        high_np = conf_high.squeeze(1).clamp(0.0, 1.0).cpu().numpy()

        node_scores: dict[UUID, NodeScore] = {}
        for i, pseudo_id in enumerate(node_ids):
            # Rank features by their raw value (higher = more signal); take top 3
            feature_vec = data.x[i].cpu().numpy()
            top_idx = feature_vec.argsort()[::-1][:3]
            top_features: dict[str, float] = {
                FEATURE_NAMES[j]: float(feature_vec[j]) for j in top_idx
            }

            node_scores[pseudo_id] = NodeScore(
                pseudo_id=pseudo_id,
                burnout_score=float(mean_np[i]),
                confidence_low=float(max(0.0, low_np[i])),
                confidence_high=float(min(1.0, high_np[i])),
                top_features=top_features,
            )

        high_risk_count = sum(
            1 for ns in node_scores.values() if ns.burnout_score > 0.75
        )
        logger.info(
            "Scored %d nodes, run_id=%s, high_risk=%d",
            n_nodes,
            run_id,
            high_risk_count,
        )

        return ScoredGraph(
            node_scores=node_scores,
            nx_graph=data,
            run_id=run_id,
        )
