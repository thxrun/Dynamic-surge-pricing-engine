"""
src/preprocess.py
------------------
Reads raw surge pricing data, applies feature engineering and
preprocessing, then writes train/test splits to data/processed/.

Steps:
  1. Load raw CSV
  2. Validate schema and data quality
  3. Engineer time-based and interaction features
  4. Split into train / test sets (stratified on surge bucket)
  5. Scale numeric features with StandardScaler
  6. Serialize scaler for use at inference time
  7. Save processed CSVs

Run:
    python src/preprocess.py
"""

import os
import sys
import logging

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_FILE = os.path.join(ROOT_DIR, "params.yaml")
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "surge_pricing.csv")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.joblib")

# ── Expected schema ───────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "rider_count",
    "driver_count",
    "traffic_index",
    "weather_severity",
    "hour",
    "is_weekend",
    "demand_supply_ratio",
    "surge_multiplier",
]

# Columns that will be scaled (excludes binary and the target)
NUMERIC_FEATURES = [
    "rider_count",
    "driver_count",
    "traffic_index",
    "demand_supply_ratio",
    "hour_sin",
    "hour_cos",
    "rush_hour_score",
    "demand_pressure",
]


def load_params() -> dict:
    """Load experiment parameters from params.yaml."""
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    log.info("Params loaded from %s", PARAMS_FILE)
    return params


def validate_schema(df: pd.DataFrame) -> None:
    """Raise an error if required columns are missing or dtypes look wrong."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")

    if df["surge_multiplier"].lt(1.0).any():
        raise ValueError("surge_multiplier contains values < 1.0 — check generator.")

    if df.isnull().sum().sum() > 0:
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Null values found in columns: {null_cols}")

    log.info("Schema validation passed. Shape: %s", df.shape)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that improve model signal.

    New features
    ------------
    hour_sin / hour_cos  : Cyclical encoding of the 24-h clock so the model
                           understands that hour 23 and hour 0 are adjacent.
    is_rush_hour         : Binary flag for the two daily peak windows.
    rush_hour_score      : Soft version — 0 / 0.5 / 1.0 for off / morning / evening peak.
    demand_pressure      : rider_count minus driver_count (signed gap).
    log_demand_supply    : Log-transform of demand_supply_ratio to compress skew.
    """
    df = df.copy()

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).round(6)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).round(6)

    # Rush-hour features
    morning_peak = (df["hour"] >= 7) & (df["hour"] <= 9)
    evening_peak = (df["hour"] >= 17) & (df["hour"] <= 20)
    df["is_rush_hour"] = (morning_peak | evening_peak).astype(int)
    df["rush_hour_score"] = (
        morning_peak.astype(float) * 0.5 + evening_peak.astype(float) * 1.0
    ).clip(0, 1)

    # Supply-demand gap
    df["demand_pressure"] = (df["rider_count"] - df["driver_count"]).astype(float)

    # Log-transform skewed ratio (add 1 to avoid log(0))
    df["log_demand_supply"] = np.log1p(df["demand_supply_ratio"]).round(6)

    log.info("Feature engineering done. New shape: %s", df.shape)
    return df


def make_surge_bucket(series: pd.Series) -> pd.Series:
    """
    Bin surge_multiplier into 4 coarse buckets for stratified splitting.
    Avoids data leakage — only used to guide the split, not as a feature.
    """
    return pd.cut(series, bins=[0.9, 1.5, 2.0, 2.5, 4.1], labels=[0, 1, 2, 3])


def preprocess(params: dict) -> None:
    """End-to-end preprocessing pipeline."""

    test_size = params["preprocess"]["test_size"]
    random_state = params["preprocess"]["random_state"]

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("Loading raw data from %s", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)

    # ── 2. Validate ───────────────────────────────────────────────────────────
    validate_schema(df)

    # ── 3. Feature engineering ────────────────────────────────────────────────
    df = engineer_features(df)

    # ── 4. Split (stratified by surge bucket) ─────────────────────────────────
    surge_bucket = make_surge_bucket(df["surge_multiplier"])
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=surge_bucket,
    )
    log.info(
        "Train size: %d | Test size: %d (%.0f%% test split)",
        len(df_train),
        len(df_test),
        test_size * 100,
    )

    # ── 5. Scale numeric features ─────────────────────────────────────────────
    scaler = StandardScaler()

    # Fit ONLY on training data to prevent data leakage
    df_train[NUMERIC_FEATURES] = scaler.fit_transform(df_train[NUMERIC_FEATURES])
    df_test[NUMERIC_FEATURES] = scaler.transform(df_test[NUMERIC_FEATURES])

    log.info("StandardScaler fitted on training set.")

    # ── 6. Save scaler ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    log.info("Scaler saved → %s", SCALER_PATH)

    # ── 7. Save processed data ────────────────────────────────────────────────
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    log.info("Processed train → %s", train_path)
    log.info("Processed test  → %s", test_path)
    log.info("Preprocessing complete ✓")

    # ── Summary report ────────────────────────────────────────────────────────
    log.info("\n--- Feature summary (train set) ---\n%s", df_train.describe().round(3))


if __name__ == "__main__":
    params = load_params()
    preprocess(params)
