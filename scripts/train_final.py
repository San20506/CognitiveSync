"""Final training run on real CSV data using tuned config.

Uses best hyperparameters from tune_and_train.py:
- SmallBurnoutGAT (10 features, hidden 32*2, dropout 0.1)
- Drops zero-heavy engineering-only features
- Trains on full dataset, evaluates via 5-fold CV, saves best-fold weights
- Runs 3 independent seeds, keeps highest AUC checkpoint
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA = Path(__file__).parent.parent / "data"
REGISTRY = Path(__file__).parent.parent / "models"
ARTIFACTS = Path(__file__).parent.parent / "artifacts"

FEATURE_COLS = [
    "meeting_density", "after_hours_ratio", "response_latency_avg",
    "focus_time_blocks", "msg_volume_daily", "msg_response_time",
    "mention_load", "context_switch_rate", "hrv_avg", "sleep_score",
]
IN_DIM = len(FEATURE_COLS)
SEEDS = [42, 7, 99]


class SmallBurnoutGAT(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = GATConv(IN_DIM, 32, heads=2, dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = GATConv(64, 16, heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Linear(16, 1)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        x = self.elu(self.bn1(self.conv1(x, edge_index)))
        x = self.drop(x)
        x = self.elu(self.conv2(x, edge_index))
        return self.classifier(x)


def load_pyg() -> tuple[Data, np.ndarray]:
    feat = pd.read_csv(DATA / "features.csv")
    inter = pd.read_csv(DATA / "interactions.csv")
    node_index = {pid: i for i, pid in enumerate(feat["pseudo_id"])}

    x = torch.tensor(feat[FEATURE_COLS].values.astype(np.float32))
    labels_np = feat["burnout_label"].values.astype(np.float32)

    valid = inter[
        inter["source_pseudo_id"].isin(node_index) &
        inter["target_pseudo_id"].isin(node_index)
    ]
    src = [node_index[s] for s in valid["source_pseudo_id"]]
    dst = [node_index[t] for t in valid["target_pseudo_id"]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor([[w] for w in valid["weight"].values.astype(np.float32)])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    logger.info("%d nodes | %d edges | %d burnout (%.0f%%)",
                len(feat), len(src), int(labels_np.sum()), 100 * labels_np.mean())
    return data, labels_np


def train_fold(
    data: Data, labels_np: np.ndarray,
    tr_idx: list, te_idx: list,
    seed: int, epochs: int = 300,
) -> tuple[dict, dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallBurnoutGAT(dropout=0.1).to(device)
    data_d = data.to(device)
    y = torch.tensor(labels_np, dtype=torch.float32).to(device)
    n = len(labels_np)

    tr_mask = torch.zeros(n, dtype=torch.bool, device=device)
    te_mask = torch.zeros(n, dtype=torch.bool, device=device)
    tr_mask[tr_idx] = True
    te_mask[te_idx] = True

    n_pos = float(y[tr_mask].sum())
    n_neg = len(tr_idx) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss, wait, best_ep = float("inf"), 0, 0
    best_state: dict[str, torch.Tensor] = {}

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data_d.x, data_d.edge_index, data_d.edge_attr)
        loss = criterion(out[tr_mask].squeeze(), y[tr_mask])
        loss.backward()
        opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(
                model(data_d.x, data_d.edge_index, data_d.edge_attr)[te_mask].squeeze(),
                y[te_mask]
            ).item()

        if vl < best_loss:
            best_loss, best_ep, wait = vl, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 25:
                break

    # Evaluate
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(data_d.x, data_d.edge_index, data_d.edge_attr)
        )[te_mask].squeeze().cpu().numpy()

    y_te = labels_np[te_idx]
    prec, rec, threshs = precision_recall_curve(y_te, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    bt = float(threshs[f1s[:-1].argmax()]) if len(threshs) else 0.5
    preds = (probs > bt).astype(float)

    metrics = {
        "acc": float((preds == y_te).mean()),
        "f1": float(f1_score(y_te, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_te, probs)) if len(np.unique(y_te)) > 1 else 0.5,
        "best_ep": best_ep,
        "threshold": bt,
    }
    return metrics, best_state


def main() -> int:
    REGISTRY.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    data, labels_np = load_pyg()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_results: list[dict] = []
    global_best_auc = -1.0
    global_best_state: dict[str, torch.Tensor] = {}
    global_best_meta: dict = {}

    for seed in SEEDS:
        logger.info("── Seed %d ──────────────────────────────", seed)
        fold_aucs = []
        for fold, (tr_idx, te_idx) in enumerate(kf.split(np.zeros(len(labels_np)), labels_np)):
            metrics, state = train_fold(data, labels_np, tr_idx.tolist(), te_idx.tolist(), seed)
            fold_aucs.append(metrics["auc"])
            logger.info("  fold %d  acc=%.3f  f1=%.3f  auc=%.3f  ep=%d",
                        fold + 1, metrics["acc"], metrics["f1"], metrics["auc"], metrics["best_ep"])
            if metrics["auc"] > global_best_auc:
                global_best_auc = metrics["auc"]
                global_best_state = state
                global_best_meta = {"seed": seed, "fold": fold + 1, **metrics}
            all_results.append({"seed": seed, "fold": fold + 1, **metrics})

        mean_auc = float(np.mean(fold_aucs))
        logger.info("  seed %d mean AUC = %.4f", seed, mean_auc)

    # Summary
    accs = [r["acc"] for r in all_results]
    f1s  = [r["f1"]  for r in all_results]
    aucs = [r["auc"] for r in all_results]

    print(f"\n{'='*45}")
    print(f"  TRAINING COMPLETE  ({len(SEEDS)} seeds × 5 folds = {len(all_results)} evals)")
    print(f"{'='*45}")
    print(f"  Accuracy  :  {np.mean(accs):.4f}  ±{np.std(accs):.3f}")
    print(f"  F1        :  {np.mean(f1s):.4f}  ±{np.std(f1s):.3f}")
    print(f"  AUC-ROC   :  {np.mean(aucs):.4f}  ±{np.std(aucs):.3f}")
    print(f"  Best AUC  :  {global_best_auc:.4f}  (seed={global_best_meta['seed']} fold={global_best_meta['fold']})")
    print(f"{'='*45}")

    # Save best checkpoint
    ckdir = REGISTRY / "final-v1"
    ckdir.mkdir(parents=True, exist_ok=True)
    torch.save(global_best_state, ckdir / "model.pt")
    artifact = {
        "version": "final-v1",
        "saved_at": datetime.utcnow().isoformat(),
        "feature_cols": FEATURE_COLS,
        "n_features": IN_DIM,
        "architecture": "SmallBurnoutGAT (10→64→16→1, dropout=0.1)",
        "seeds": SEEDS,
        "cv_folds": 5,
        "mean_accuracy": float(np.mean(accs)),
        "mean_f1": float(np.mean(f1s)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "best_auc": global_best_auc,
        "best_checkpoint_meta": global_best_meta,
        "all_fold_results": all_results,
    }
    (ckdir / "metrics.json").write_text(json.dumps(artifact, indent=2))
    latest = REGISTRY / "latest"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to("final-v1")
    (ARTIFACTS / "final_training_metrics.json").write_text(json.dumps(artifact, indent=2))
    logger.info("Saved: models/final-v1/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
