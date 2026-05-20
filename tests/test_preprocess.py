"""
tests/test_preprocess.py
-------------------------
Unit tests for src/preprocess.py.
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

# Import functions we want to test
from src.preprocess import engineer_features, make_surge_bucket, validate_schema

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal valid DataFrame that mirrors the raw dataset schema."""
    return pd.DataFrame(
        {
            "rider_count": [50, 20, 80, 10],
            "driver_count": [30, 25, 20, 40],
            "traffic_index": [2.5, 1.2, 4.0, 1.0],
            "weather_severity": [0, 1, 2, 0],
            "hour": [8, 14, 18, 3],
            "is_weekend": [0, 1, 0, 1],
            "demand_supply_ratio": [1.67, 0.8, 4.0, 0.25],
            "surge_multiplier": [1.5, 1.0, 2.8, 1.0],
        }
    )


# ── Schema Validation Tests ───────────────────────────────────────────────────

def test_validate_schema_passes(sample_df):
    """Should pass silently with a correct DataFrame."""
    validate_schema(sample_df)  # no exception expected


def test_validate_schema_missing_column(sample_df):
    """Should raise ValueError if a required column is missing."""
    df_bad = sample_df.drop(columns=["surge_multiplier"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df_bad)


def test_validate_schema_rejects_surge_below_one(sample_df):
    """surge_multiplier must never be < 1.0."""
    df_bad = sample_df.copy()
    df_bad.loc[0, "surge_multiplier"] = 0.5
    with pytest.raises(ValueError, match="surge_multiplier contains values < 1.0"):
        validate_schema(df_bad)


def test_validate_schema_rejects_nulls(sample_df):
    """Should raise ValueError when null values are present."""
    df_bad = sample_df.copy()
    df_bad.loc[0, "rider_count"] = np.nan
    with pytest.raises(ValueError, match="Null values found"):
        validate_schema(df_bad)


# ── Feature Engineering Tests ─────────────────────────────────────────────────

def test_engineer_features_adds_new_columns(sample_df):
    """All expected engineered columns must be present."""
    df_out = engineer_features(sample_df)
    expected_new_cols = [
        "hour_sin", "hour_cos",
        "is_rush_hour", "rush_hour_score",
        "demand_pressure", "log_demand_supply",
    ]
    for col in expected_new_cols:
        assert col in df_out.columns, f"Missing engineered column: {col}"


def test_hour_sin_cos_range(sample_df):
    """Cyclical features must be in [-1, 1]."""
    df_out = engineer_features(sample_df)
    assert df_out["hour_sin"].between(-1, 1).all()
    assert df_out["hour_cos"].between(-1, 1).all()


def test_rush_hour_flag(sample_df):
    """
    hour=8 (morning peak) and hour=18 (evening peak) → is_rush_hour=1.
    hour=3 and hour=14 → is_rush_hour=0.
    """
    df_out = engineer_features(sample_df)
    # Row 0: hour=8 (morning peak) → 1
    assert df_out.iloc[0]["is_rush_hour"] == 1
    # Row 2: hour=18 (evening peak) → 1
    assert df_out.iloc[2]["is_rush_hour"] == 1
    # Row 1: hour=14 (midday) → 0
    assert df_out.iloc[1]["is_rush_hour"] == 0
    # Row 3: hour=3 (night) → 0
    assert df_out.iloc[3]["is_rush_hour"] == 0


def test_demand_pressure_values(sample_df):
    """demand_pressure = rider_count - driver_count."""
    df_out = engineer_features(sample_df)
    expected = sample_df["rider_count"] - sample_df["driver_count"]
    pd.testing.assert_series_equal(
        df_out["demand_pressure"].reset_index(drop=True),
        expected.astype(float).reset_index(drop=True),
        check_names=False,
    )


def test_log_demand_supply_non_negative(sample_df):
    """log1p transform of a positive ratio must always be >= 0."""
    df_out = engineer_features(sample_df)
    assert (df_out["log_demand_supply"] >= 0).all()


def test_engineer_features_does_not_mutate_input(sample_df):
    """engineer_features must return a copy, not modify the original."""
    original_cols = list(sample_df.columns)
    _ = engineer_features(sample_df)
    assert list(sample_df.columns) == original_cols


# ── Surge Bucket Tests ────────────────────────────────────────────────────────

def test_make_surge_bucket_returns_four_categories(sample_df):
    """Surge buckets should map to one of labels [0, 1, 2, 3]."""
    buckets = make_surge_bucket(sample_df["surge_multiplier"])
    valid_labels = {0, 1, 2, 3}
    result_labels = set(buckets.dropna().astype(int).unique())
    assert result_labels.issubset(valid_labels)
