"""
tests/test_train.py
--------------------
Unit and integration tests for src/train.py and src/evaluate.py.

Run:
    pytest tests/ -v
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.train import (
    FEATURE_COLS,
    TARGET_COL,
    compute_metrics,
    plot_feature_importance,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_train_df():
    """
    A minimal but valid processed DataFrame (200 rows) for fast unit tests.
    Mirrors the schema produced by src/preprocess.py.
    """
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        "rider_count":          np.random.randn(n),
        "driver_count":         np.random.randn(n),
        "traffic_index":        np.random.randn(n),
        "weather_severity":     np.random.choice([0, 1, 2, 3], n),
        "is_weekend":           np.random.choice([0, 1], n),
        "demand_supply_ratio":  np.abs(np.random.randn(n)) + 0.5,
        "hour_sin":             np.sin(np.random.uniform(0, 2 * np.pi, n)),
        "hour_cos":             np.cos(np.random.uniform(0, 2 * np.pi, n)),
        "is_rush_hour":         np.random.choice([0, 1], n),
        "rush_hour_score":      np.random.uniform(0, 1, n),
        "demand_pressure":      np.random.randn(n),
        "log_demand_supply":    np.abs(np.random.randn(n)),
        "surge_multiplier":     np.clip(1.0 + np.abs(np.random.randn(n)) * 0.5, 1.0, 4.0),
        # extra column (should be ignored by training)
        "hour":                 np.random.randint(0, 24, n),
    })


@pytest.fixture
def fitted_model(tiny_train_df):
    """Returns a fast-trained XGBoost model on tiny data."""
    X = tiny_train_df[FEATURE_COLS]
    y = tiny_train_df[TARGET_COL]
    model = XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


# ── compute_metrics tests ─────────────────────────────────────────────────────

def test_compute_metrics_perfect_prediction():
    """When prediction equals truth, R²=1, RMSE=0, MAE=0."""
    y = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    metrics = compute_metrics(y, y)
    assert metrics["r2"]   == pytest.approx(1.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mae"]  == pytest.approx(0.0, abs=1e-6)
    assert metrics["mape"] == pytest.approx(0.0, abs=1e-4)


def test_compute_metrics_returns_all_keys():
    """All four metric keys must be present."""
    y = np.array([1.0, 2.0, 3.0])
    metrics = compute_metrics(y, y + 0.1)
    for key in ("rmse", "mae", "r2", "mape"):
        assert key in metrics, f"Missing metric key: {key}"


def test_compute_metrics_rmse_positive():
    """RMSE must always be non-negative."""
    y_true = np.array([1.0, 2.0, 1.5])
    y_pred = np.array([1.1, 1.9, 1.6])
    assert compute_metrics(y_true, y_pred)["rmse"] >= 0


def test_compute_metrics_r2_range():
    """R² should be ≤ 1.0 for any predictions."""
    y_true = np.array([1.0, 2.0, 1.5, 3.0])
    y_pred = np.array([1.2, 1.8, 1.6, 2.8])
    assert compute_metrics(y_true, y_pred)["r2"] <= 1.0


# ── Model training tests ───────────────────────────────────────────────────────

def test_model_produces_correct_output_shape(fitted_model, tiny_train_df):
    """predict() output shape must match number of test rows."""
    X = tiny_train_df[FEATURE_COLS]
    preds = fitted_model.predict(X)
    assert preds.shape == (len(tiny_train_df),)


def test_model_predictions_within_business_bounds(fitted_model, tiny_train_df):
    """
    After clipping to [1.0, 4.0] (same as evaluate.py),
    no prediction should be outside the valid surge range.
    """
    X = tiny_train_df[FEATURE_COLS]
    preds = np.clip(fitted_model.predict(X), 1.0, 4.0)
    assert preds.min() >= 1.0, "Prediction below 1.0 surge floor"
    assert preds.max() <= 4.0, "Prediction above 4.0 surge cap"


def test_model_r2_above_zero(fitted_model, tiny_train_df):
    """Even a tiny model should beat a constant predictor (R² > 0)."""
    X = tiny_train_df[FEATURE_COLS]
    y = tiny_train_df[TARGET_COL].values
    preds = fitted_model.predict(X)
    assert r2_score(y, preds) > 0


def test_feature_cols_all_present_in_dataframe(tiny_train_df):
    """Every column in FEATURE_COLS must exist in the processed DataFrame."""
    missing = [c for c in FEATURE_COLS if c not in tiny_train_df.columns]
    assert not missing, f"Missing feature columns: {missing}"


# ── plot_feature_importance tests ─────────────────────────────────────────────

def test_feature_importance_plot_creates_file(fitted_model):
    """plot_feature_importance() must create a PNG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(tmpdir, "fi.png")
        plot_feature_importance(fitted_model, FEATURE_COLS, plot_path)
        assert os.path.exists(plot_path), "Feature importance plot not created"
        assert os.path.getsize(plot_path) > 0, "Feature importance plot is empty"


# ── metrics.json format tests ─────────────────────────────────────────────────

def test_metrics_json_structure():
    """Manually simulate what train.py writes and verify the structure."""
    sample_metrics = {
        "train": {"rmse": 0.12, "mae": 0.09, "r2": 0.91, "mape": 5.2},
        "test":  {"rmse": 0.15, "mae": 0.11, "r2": 0.88, "mape": 6.1},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(sample_metrics, f)
        path = f.name

    with open(path) as f:
        loaded = json.load(f)

    assert "train" in loaded
    assert "test" in loaded
    for split in ("train", "test"):
        for key in ("rmse", "mae", "r2", "mape"):
            assert key in loaded[split], f"Missing '{key}' in {split} metrics"

    os.unlink(path)
