"""Unit tests for intelligence/features.py — T-037, T-038."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intelligence.features import (
    FEATURE_COLS,
    get_feature_matrix,
    load_features,
    normalise_features,
)


def _make_df(n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(0.1, 10.0, size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    df.insert(0, "pseudo_id", [f"user-{i:04d}" for i in range(n)])
    df.insert(1, "window_start", "2026-04-01T00:00:00Z")
    return df


class TestNormaliseFeatures:
    def test_output_in_unit_interval(self) -> None:
        df = _make_df(20)
        out = normalise_features(df)
        mat = out[FEATURE_COLS].values
        assert (mat >= 0).all() and (mat <= 1).all()

    def test_max_is_one_min_is_zero(self) -> None:
        df = _make_df(20)
        out = normalise_features(df)
        for col in FEATURE_COLS:
            assert abs(out[col].max() - 1.0) < 1e-4, f"{col} max not 1"
            assert out[col].min() >= 0.0

    def test_does_not_mutate_input(self) -> None:
        df = _make_df(10)
        original = df[FEATURE_COLS].values.copy()
        normalise_features(df)
        assert np.allclose(df[FEATURE_COLS].values, original)

    def test_constant_column_does_not_raise(self) -> None:
        df = _make_df(10)
        df["meeting_density"] = 5.0
        out = normalise_features(df)
        assert not out["meeting_density"].isna().any()

    def test_no_nan_in_output(self) -> None:
        df = _make_df(15)
        out = normalise_features(df)
        assert not out[FEATURE_COLS].isna().any().any()


class TestLoadFeatures:
    def test_loads_csv_and_returns_feature_cols(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        df = _make_df(10)
        path = str(tmp_path / "features.csv")
        df.to_csv(path, index=False)
        result = load_features(path)
        assert "pseudo_id" in result.columns
        for col in FEATURE_COLS:
            assert col in result.columns

    def test_deduplicates_to_most_recent_window(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        df = _make_df(5)
        df2 = df.copy()
        df["window_start"] = "2026-03-01T00:00:00Z"
        df2["window_start"] = "2026-04-01T00:00:00Z"
        combined = pd.concat([df, df2])
        path = str(tmp_path / "features.csv")
        combined.to_csv(path, index=False)
        result = load_features(path)
        assert len(result) == 5

    def test_imputes_missing_hrv_sleep(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        df = _make_df(5)
        df["hrv_avg"] = np.nan
        df["sleep_score"] = np.nan
        path = str(tmp_path / "features.csv")
        df.to_csv(path, index=False)
        result = load_features(path)
        assert not result["hrv_avg"].isna().any()
        assert not result["sleep_score"].isna().any()

    def test_all_values_in_unit_interval(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
        df = _make_df(10)
        path = str(tmp_path / "features.csv")
        df.to_csv(path, index=False)
        result = load_features(path)
        mat = result[FEATURE_COLS].values
        assert (mat >= 0).all() and (mat <= 1).all()


class TestGetFeatureMatrix:
    def test_shape_matches_ordered_ids(self) -> None:
        df = _make_df(8)
        ids = df["pseudo_id"].tolist()[:5]
        mat = get_feature_matrix(load_features_from_df(df), ids)
        assert mat.shape == (5, len(FEATURE_COLS))

    def test_unknown_id_gets_neutral_imputation(self) -> None:
        df = _make_df(5)
        mat = get_feature_matrix(load_features_from_df(df), ["unknown-id"])
        assert mat.shape == (1, len(FEATURE_COLS))
        assert np.allclose(mat[0], 0.5)

    def test_dtype_is_float32(self) -> None:
        df = _make_df(5)
        ids = df["pseudo_id"].tolist()
        mat = get_feature_matrix(load_features_from_df(df), ids)
        assert mat.dtype == np.float32


def load_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Helper: normalise a raw df into the shape load_features() returns."""
    from intelligence.features import normalise_features
    df2 = df.copy()
    if "window_start" in df2.columns:
        df2 = df2.sort_values("window_start").groupby("pseudo_id").last().reset_index()
    for col in ("hrv_avg", "sleep_score"):
        if col in df2.columns:
            df2[col] = df2[col].fillna(0.5)
    for col in FEATURE_COLS:
        if col in df2.columns:
            df2[col] = df2[col].fillna(df2[col].median())
    return normalise_features(df2)[["pseudo_id"] + FEATURE_COLS]
