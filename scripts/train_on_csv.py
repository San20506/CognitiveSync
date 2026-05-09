"""Train BurnoutGAT on real CSV data (data/employees.csv, features.csv, interactions.csv)."""

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
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from intelligence.gnn_model import BurnoutGAT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA = Path(__file__).parent.parent / "data"
REGISTRY = Path(__file__).parent.parent / "models"
ARTIFACTS = Path(__file__).parent.parent / "artifacts"

FEATURE_COLS = [
    "meeting_density", "after_hours_ratio", "response_latency_avg",
    "focus_time_blocks", "msg_volume_daily", "msg_response_time",
    "mention_load", "commit_frequency", "pr_review_load",
    "context_switch_rate", "after_hours_commits", "hrv_avg", "sleep_score",
]


def load_pyg() -> tuple[Data, torch.Tensor]:
    feat = pd.read_csv(DATA / "features.csv")
    inter = pd.read_csv(DATA / "interactions.csv")
    node_index = {pid: i for i, pid in enumerate(feat["pseudo_id"])}

    x = torch.tensor(feat[FEATURE_COLS].values.astype(np.float32))
    labels = torch.tensor(feat["burnout_label"].values.astype(np.float32))

    valid = inter[inter["source_pseudo_id"].isin(node_index) & inter["target_pseudo_id"].isin(node_index)]
    src = [node_index[s] for s in valid["source_pseudo_id"]]
    dst = [node_index[t] for t in valid["target_pseudo_id"]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor([[w] for w in valid["weight"].values.astype(np.float32)])

    logger.info("%d nodes | %d edges | %d burnout (%.0f%%)",
                len(feat), len(src), int(labels.sum()), 100 * labels.mean())
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), labels


def train(data: Data, labels: torch.Tensor, epochs: int = 300, lr: float = 0.001, patience: int = 20) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BurnoutGAT().to(device)
    data = data.to(device)
    y = labels.to(device).float()
    n = y.shape[0]
    y_np = y.cpu().numpy()

    tr_idx, te_idx = train_test_split(range(n), test_size=0.2, stratify=y_np, random_state=42)
    tr_mask = torch.zeros(n, dtype=torch.bool, device=device)
    te_mask = torch.zeros(n, dtype=torch.bool, device=device)
    tr_mask[list(tr_idx)] = True
    te_mask[list(te_idx)] = True

    n_pos = float(y[tr_mask].sum())
    n_neg = len(tr_idx) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_loss, best_ep, wait = float("inf"), 0, 0
    best_state = None

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[tr_mask].squeeze(), y[tr_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(data.x, data.edge_index, data.edge_attr)[te_mask].squeeze(), y[te_mask]).item()
        if vl < best_loss:
            best_loss, best_ep, wait = vl, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stop ep %d", ep)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr))[te_mask].squeeze().cpu().numpy()
    y_te = y[te_mask].cpu().numpy()

    prec, rec, threshs = precision_recall_curve(y_te, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    bt = float(threshs[f1s[:-1].argmax()]) if len(threshs) else 0.5
    preds = (probs > bt).astype(float)
    acc = float((preds == y_te).mean())
    f1 = float(f1_score(y_te, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(y_te, probs))
    except Exception:
        auc = 0.5

    logger.info("acc=%.3f f1=%.3f auc=%.3f (best ep %d)", acc, f1, auc, best_ep)

    # Save checkpoint
    ckdir = REGISTRY / "csv-v1"
    ckdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckdir / "model.pt")
    meta = {"version": "csv-v1", "saved_at": datetime.utcnow().isoformat(),
            "val_accuracy": acc, "val_f1": f1, "auc_roc": auc, "best_epoch": best_ep}
    (ckdir / "metrics.json").write_text(json.dumps(meta, indent=2))
    latest = REGISTRY / "latest"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to("csv-v1")

    return meta


def main() -> int:
    REGISTRY.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    data, labels = load_pyg()
    meta = train(data, labels)
    (ARTIFACTS / "csv_training_metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nAccuracy : {meta['val_accuracy']:.4f}")
    print(f"F1       : {meta['val_f1']:.4f}")
    print(f"AUC-ROC  : {meta['auc_roc']:.4f}")
    print(f"Checkpoint: models/csv-v1/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
