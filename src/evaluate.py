"""
src/evaluate.py
----------------
Standalone evaluation script — loads the saved model, runs it on the
test set, and prints a rich evaluation report.

Used by:
  - DVC 'evaluate' stage (produces reports/metrics.json)
  - GitHub Actions CI gate (Phase 3) to enforce quality thresholds
  - Developers wanting a quick sanity check after re-training

Run:
    python src/evaluate.py
"""

import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_FILE  = os.path.join(ROOT_DIR, "params.yaml")
TEST_PATH    = os.path.join(ROOT_DIR, "data", "processed", "test.csv")
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "surge_model.joblib")
METRICS_PATH = os.path.join(ROOT_DIR, "reports", "metrics.json")

TARGET_COL = "surge_multiplier"

FEATURE_COLS = [
    "rider_count", "driver_count", "traffic_index", "weather_severity",
    "is_weekend", "demand_supply_ratio", "hour_sin", "hour_cos",
    "is_rush_hour", "rush_hour_score", "demand_pressure", "log_demand_supply",
]


def load_params() -> dict:
    with open(PARAMS_FILE, "r") as f:
        return yaml.safe_load(f)


def load_metrics() -> dict:
    """Load the metrics JSON that was written by train.py."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"metrics.json not found at {METRICS_PATH}. Run train.py first."
        )
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def evaluate_model(params: dict) -> dict:
    """
    Load the saved model + test set, compute metrics, enforce thresholds.

    Returns
    -------
    dict of test metrics
    Exits with code 1 if quality gates fail (used by CI).
    """
    # ── Load artifacts ────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        log.error("Model not found at %s. Run train.py first.", MODEL_PATH)
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL].values

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1.0, 4.0)   # enforce business rule: 1x–4x only

    # ── Metrics ───────────────────────────────────────────────────────────────
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    mape = float(
        np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-6, None))) * 100
    )

    metrics = {
        "rmse": round(rmse, 6),
        "mae":  round(mae, 6),
        "r2":   round(r2, 6),
        "mape": round(mape, 4),
    }

    # ── Pretty report ─────────────────────────────────────────────────────────
    separator = "─" * 45
    print(f"\n{separator}")
    print("  SURGE PRICING MODEL — EVALUATION REPORT")
    print(separator)
    print(f"  Test samples   : {len(y_test):,}")
    print(f"  RMSE           : {rmse:.4f}  (lower is better)")
    print(f"  MAE            : {mae:.4f}  (lower is better)")
    print(f"  R²             : {r2:.4f}  (higher is better, max 1.0)")
    print(f"  MAPE           : {mape:.2f}%   (lower is better)")
    print(separator)

    # ── Residual bucket breakdown ─────────────────────────────────────────────
    residuals = np.abs(y_test - y_pred)
    print("  Absolute error distribution:")
    for threshold, label in [(0.1, "≤ 0.1x"), (0.2, "≤ 0.2x"), (0.5, "≤ 0.5x")]:
        pct = (residuals <= threshold).mean() * 100
        print(f"    {label} : {pct:.1f}% of predictions")
    print(separator)

    # ── Quality gate check ────────────────────────────────────────────────────
    min_r2   = params["evaluate"]["min_r2"]
    max_rmse = params["evaluate"]["max_rmse"]

    gate_r2_ok   = r2   >= min_r2
    gate_rmse_ok = rmse <= max_rmse

    print("  Quality Gates:")
    print(f"    R²   {r2:.4f} {'✅' if gate_r2_ok   else '❌'} (threshold ≥ {min_r2})")
    print(f"    RMSE {rmse:.4f} {'✅' if gate_rmse_ok else '❌'} (threshold ≤ {max_rmse})")
    print(separator + "\n")

    if not (gate_r2_ok and gate_rmse_ok):
        log.error("Quality gate FAILED. Retrain or tune hyperparameters.")
        sys.exit(1)   # non-zero exit → CI pipeline will fail

    log.info("Quality gate PASSED. Model is production-ready.")
    return metrics


if __name__ == "__main__":
    params = load_params()
    evaluate_model(params)
