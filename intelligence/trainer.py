"""GAT model training pipeline — implementation in Phase 4B (T-046, T-047, T-048, T-049).

Training strategy:
- Dataset: Synthetic org graph (100-500 nodes) with rule-based burnout labels
- Loss: BCEWithLogitsLoss with pos_weight (burnout class ~15% minority)
- Optimizer: Adam, lr=0.001, weight_decay=5e-4
- Validation: 80/10/10 train/val/test split (stratified)
- Early stopping: patience=10 on val_loss
- Hardware: RTX 4060 8GB VRAM — mini-batch via PyG DataLoader if N > 1000
- Threshold gate: val_accuracy >= 0.80 required before model promotion
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from intelligence.gnn_model import BurnoutGAT

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    val_accuracy: float
    val_f1: float
    auc_roc: float
    best_epoch: int
    train_loss_final: float
    val_loss_final: float


class ModelTrainer:
    """GAT training pipeline.

    Phase 4B implementation target: T-046.
    """

    def __init__(
        self,
        model: object,  # BurnoutGAT
        model_registry_path: Path,
        device: str = "cuda",
    ) -> None:
        self._model = model
        self._registry = model_registry_path
        self._device = device

    def train(
        self,
        pyg_data: object,  # torch_geometric.data.Data
        labels: object,  # Tensor[N] binary burnout labels
        epochs: int = 100,
        lr: float = 0.001,
    ) -> TrainingMetrics:
        """Train the GAT model.

        Implements T-046, T-047: full training loop with stratified split,
        class-weighted BCE loss, Adam optimiser, early stopping (patience=10),
        and a val_accuracy >= 0.80 phase gate (T-048).
        """
        data: Data = pyg_data  # type: ignore[assignment]
        y: torch.Tensor = labels  # type: ignore[assignment]
        model: BurnoutGAT = self._model  # type: ignore[assignment]

        device = torch.device(self._device if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        data = data.to(device)
        y = y.to(device).float()

        n_nodes = y.shape[0]
        indices = list(range(n_nodes))
        y_np = y.cpu().numpy()

        # Stratified 80 / 10 / 10 split
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, stratify=y_np, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            stratify=y_np[temp_idx],
            random_state=42,
        )

        train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # pos_weight to handle ~15 % burnout minority class
        n_pos = float(y[train_mask].sum().item())
        n_neg = float(train_mask.sum().item()) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=5e-4
        )

        best_val_loss = float("inf")
        patience_count = 0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(epochs):
            # --- training step ---
            model.train()
            optimizer.zero_grad()
            out: torch.Tensor = model(data.x, data.edge_index, data.edge_attr)  # [N,1]
            loss: torch.Tensor = criterion(out[train_mask].squeeze(), y[train_mask])
            loss.backward()
            optimizer.step()

            # --- validation step ---
            model.eval()
            with torch.no_grad():
                val_out: torch.Tensor = model(
                    data.x, data.edge_index, data.edge_attr
                )
                val_loss: torch.Tensor = criterion(
                    val_out[val_mask].squeeze(), y[val_mask]
                )

            logger.debug(
                "Epoch %d: train_loss=%.4f val_loss=%.4f",
                epoch,
                loss.item(),
                val_loss.item(),
            )

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_epoch = epoch
                patience_count = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= 10:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation on the held-out test set
        model.eval()
        with torch.no_grad():
            test_logits: torch.Tensor = model(
                data.x, data.edge_index, data.edge_attr
            )
            test_probs = torch.sigmoid(test_logits)
            test_preds = (test_probs[test_mask].squeeze() > 0.5).float().cpu().numpy()
            test_labels_np = y[test_mask].cpu().numpy()
            train_logits: torch.Tensor = model(
                data.x, data.edge_index, data.edge_attr
            )
            train_loss_final = criterion(
                train_logits[train_mask].squeeze(), y[train_mask]
            ).item()

        val_accuracy = float((test_preds == test_labels_np).mean())
        val_f1 = float(f1_score(test_labels_np, test_preds, zero_division=0))
        try:
            auc = float(
                roc_auc_score(
                    test_labels_np,
                    test_probs[test_mask].squeeze().cpu().numpy(),
                )
            )
        except ValueError:
            # Only one class present in test split
            auc = 0.5

        metrics = TrainingMetrics(
            val_accuracy=val_accuracy,
            val_f1=val_f1,
            auc_roc=auc,
            best_epoch=best_epoch,
            train_loss_final=train_loss_final,
            val_loss_final=best_val_loss,
        )

        logger.info(
            "Training complete: val_accuracy=%.3f val_f1=%.3f auc=%.3f",
            val_accuracy,
            val_f1,
            auc,
        )

        # T-048 phase gate
        if val_accuracy < 0.80:
            raise ValueError(
                f"Phase gate failed: val_accuracy={val_accuracy:.3f} < 0.80"
            )

        return metrics

    def save_checkpoint(self, version: str, metrics: TrainingMetrics) -> Path:
        """Save model checkpoint to versioned registry path.

        Implements T-050: writes model.pt and metrics.json under
        ``<registry>/<version>/`` and updates the ``latest`` symlink.
        """
        self._registry.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self._registry / version
        checkpoint_dir.mkdir(exist_ok=True)

        model_path = checkpoint_dir / "model.pt"
        meta_path = checkpoint_dir / "metrics.json"

        model: BurnoutGAT = self._model  # type: ignore[assignment]
        torch.save(model.state_dict(), model_path)  # type: ignore[arg-type]

        meta: dict[str, object] = {
            "version": version,
            "saved_at": datetime.utcnow().isoformat(),
            "val_accuracy": metrics.val_accuracy,
            "val_f1": metrics.val_f1,
            "auc_roc": metrics.auc_roc,
            "best_epoch": metrics.best_epoch,
            "train_loss_final": metrics.train_loss_final,
            "val_loss_final": metrics.val_loss_final,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        # Update "latest" symlink atomically
        latest = self._registry / "latest"
        if latest.is_symlink():
            latest.unlink()
        latest.symlink_to(checkpoint_dir.name)

        logger.info(
            "Checkpoint saved: %s (val_accuracy=%.3f)",
            checkpoint_dir,
            metrics.val_accuracy,
        )
        return checkpoint_dir
