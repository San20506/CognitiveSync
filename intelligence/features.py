"""
CognitiveSync — Feature Extractor & Normaliser
T-037: Compute all 13 features per pseudo_id
T-038: Min-max normalisation with org-level statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.features import FEATURE_NAMES, MODEL_FEATURE_NAMES

# CSV column names (old naming from trained model)
CSV_COLUMN_MAPPING = {
    "after_hours_ratio": "after_hours_meetings",
    "response_latency_avg": "email_response_latency",
    "focus_time_blocks": "focus_blocks",
    "msg_volume_daily": "message_volume",
    "msg_response_time": "response_time_slack",
    "mention_load": "mention_frequency",
}

# Canonical feature columns for internal use
FEATURE_COLS = FEATURE_NAMES

# Model feature columns (10-dim for GNN input)
MODEL_FEATURE_COLS = MODEL_FEATURE_NAMES

EPSILON = 1e-8  # prevents division by zero in normalisation


# ─────────────────────────────────────────────
# T-038 — Org-level min-max normalisation
# ─────────────────────────────────────────────

def normalise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply org-level min-max normalisation to all 13 feature columns.

    Normalisation is computed across the full employee population,
    not per individual — this preserves relative risk differences
    between employees.

    Formula: normalised = (value - org_min) / (org_max - org_min + epsilon)

    Args:
        df: DataFrame containing FEATURE_COLS (raw or partially normalised)

    Returns:
        pd.DataFrame with same columns, all feature values in [0, 1]
    """
    df = df.copy()
    for col in FEATURE_COLS:
        col_min = df[col].min()
        col_max = df[col].max()
        df[col] = (df[col] - col_min) / (col_max - col_min + EPSILON)
    df[FEATURE_COLS] = df[FEATURE_COLS].clip(0.0, 1.0)
    return df


# ─────────────────────────────────────────────
# T-037 — Feature loader and validator
# ─────────────────────────────────────────────

def load_features(features_path: str) -> pd.DataFrame:
    """
    Load features CSV, impute missing wearable values, and normalise.

    If multiple time windows exist per employee, only the most recent
    window is kept.

    Imputation rules (per spec):
        hrv_avg     — fill missing with 0.5
        sleep_score — fill missing with 0.5

    Args:
        features_path: Path to features.csv

    Returns:
        pd.DataFrame with columns [pseudo_id] + FEATURE_COLS (canonical names)
        All values guaranteed in [0, 1], no NaNs.
    """
    df = pd.read_csv(features_path)
    df = df.rename(columns=CSV_COLUMN_MAPPING)

    # Keep only the most recent window per employee
    if "window_start" in df.columns:
        df = df.sort_values("window_start").groupby("pseudo_id").last().reset_index()

    # Impute missing wearable data
    for col in ("hrv_avg", "sleep_score"):
        if col in df.columns:
            df[col] = df[col].fillna(0.5)

    # Impute any remaining NaNs with column median
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Apply org-level normalisation
    df = normalise_features(df)

    wearable_cols = [c for c in ("hrv_avg", "sleep_score") if c in df.columns]
    return df[["pseudo_id"] + FEATURE_COLS + wearable_cols]


def get_feature_matrix(
    features_df: pd.DataFrame,
    ordered_ids: list[str],
) -> np.ndarray:
    """
    Build an ordered feature matrix aligned to a list of pseudo_ids.

    Employees with no feature data are imputed with 0.5 (neutral).

    Args:
        features_df:  Output of load_features()
        ordered_ids:  List of pseudo_ids in the desired node order

    Returns:
        np.ndarray shape (N, 13), dtype float32
    """
    feat_lookup = features_df.set_index("pseudo_id")[FEATURE_COLS]
    N = len(ordered_ids)
    X = np.full((N, len(FEATURE_COLS)), 0.5, dtype=np.float32)

    for i, pid in enumerate(ordered_ids):
        if pid in feat_lookup.index:
            X[i] = feat_lookup.loc[pid].values.astype(np.float32)

    return X