"""
src/train.py
-------------
Trains an XGBoost regression model on the processed surge pricing dataset
and logs every detail to MLflow:
  - Parameters (from params.yaml)
  - All evaluation metrics (RMSE, MAE, R², MAPE)
  - Feature importance plot
  - The trained model artifact
  - Registers the model in the MLflow Model Registry

Run:
    python src/train.py
"""

import json
import logging
import os

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works on any machine
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_FILE     = os.path.join(ROOT_DIR, "params.yaml")
TRAIN_PATH      = os.path.join(ROOT_DIR, "data", "processed", "train.csv")
TEST_PATH       = os.path.join(ROOT_DIR, "data", "processed", "test.csv")
MODEL_DIR       = os.path.join(ROOT_DIR, "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "surge_model.joblib")
METRICS_PATH    = os.path.join(ROOT_DIR, "reports", "metrics.json")
FI_PLOT_PATH    = os.path.join(ROOT_DIR, "reports", "feature_importance.png")

# MLflow experiment name — groups all training runs under one roof
EXPERIMENT_NAME = "surge-pricing-xgboost"

# Target column — what we're predicting
TARGET_COL = "surge_multiplier"

# All feature columns fed to the model (original + engineered)
FEATURE_COLS = [
    "rider_count",
    "driver_count",
    "traffic_index",
    "weather_severity",
    "is_weekend",
    "demand_supply_ratio",
    "hour_sin",
    "hour_cos",
    "is_rush_hour",
    "rush_hour_score",
    "demand_pressure",
    "log_demand_supply",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params() -> dict:
    """Load experiment parameters from params.yaml."""
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params


def load_data(train_path: str, test_path: str):
    """Load train/test CSVs and return feature matrices + target vectors."""
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_test  = df_test[FEATURE_COLS]
    y_test  = df_test[TARGET_COL]

    log.info("Train: %s | Test: %s", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics.

    Returns
    -------
    dict with keys: rmse, mae, r2, mape
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    # Mean Absolute Percentage Error (clip to avoid divide-by-zero)
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    )
    return {"rmse": round(rmse, 6), "mae": round(mae, 6),
            "r2": round(r2, 6), "mape": round(mape, 4)}


def plot_feature_importance(model: XGBRegressor, feature_names: list,
                             save_path: str) -> None:
    """Save a horizontal bar chart of XGBoost feature importances."""
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#4C72B0",
        edgecolor="white",
    )
    ax.set_xlabel("Importance Score (weight)")
    ax.set_title("XGBoost — Feature Importance\nDynamic Surge Pricing Engine")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info("Feature importance plot saved → %s", save_path)


def plot_predictions(y_true, y_pred, save_path: str) -> None:
    """Save an actual vs predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Surge Multiplier")
    ax.set_ylabel("Predicted Surge Multiplier")
    ax.set_title("Actual vs Predicted — Test Set")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Main training pipeline ────────────────────────────────────────────────────

def train(params: dict) -> None:
    """Full train → evaluate → log → register pipeline."""

    tp = params["train"]    # training hyperparameters
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "reports"), exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_data(TRAIN_PATH, TEST_PATH)

    # ── 2. Configure MLflow ───────────────────────────────────────────────────
    # Local tracking URI — stores all run data in ./mlruns folder
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info("MLflow experiment: '%s'", EXPERIMENT_NAME)

    # ── 3. Start MLflow run ───────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"xgb-{tp['n_estimators']}trees") as run:
        run_id = run.info.run_id
        log.info("MLflow run started: %s", run_id)

        # ── 4. Log parameters ─────────────────────────────────────────────────
        mlflow.log_params({
            "model":          tp["model"],
            "n_estimators":   tp["n_estimators"],
            "max_depth":      tp["max_depth"],
            "learning_rate":  tp["learning_rate"],
            "subsample":      tp["subsample"],
            "random_state":   tp["random_state"],
            "n_features":     len(FEATURE_COLS),
            "train_size":     len(X_train),
            "test_size":      len(X_test),
        })

        # ── 5. Build and train the model ──────────────────────────────────────
        model = XGBRegressor(
            n_estimators   = tp["n_estimators"],
            max_depth       = tp["max_depth"],
            learning_rate   = tp["learning_rate"],
            subsample       = tp["subsample"],
            random_state    = tp["random_state"],
            n_jobs          = -1,                   # use all CPU cores
            objective       = "reg:squarederror",
            eval_metric     = "rmse",
            early_stopping_rounds = 20,             # stop if no improvement
        )

        log.info("Training XGBoost … (n_estimators=%d)", tp["n_estimators"])
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,                          # suppress XGB output clutter
        )
        log.info("Training complete. Best iteration: %d", model.best_iteration)

        # ── 6. Evaluate ───────────────────────────────────────────────────────
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        train_metrics = compute_metrics(y_train.values, y_pred_train)
        test_metrics  = compute_metrics(y_test.values,  y_pred_test)

        log.info("── Train metrics ──")
        for k, v in train_metrics.items():
            log.info("  %-6s = %.4f", k, v)

        log.info("── Test metrics ──")
        for k, v in test_metrics.items():
            log.info("  %-6s = %.4f", k, v)

        # Log both sets to MLflow with train_/test_ prefix
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}":  v for k, v in test_metrics.items()})
        mlflow.log_metric("best_iteration", model.best_iteration)

        # ── 7. Save metrics JSON (for DVC metrics tracking) ───────────────────
        all_metrics = {"train": train_metrics, "test": test_metrics}
        with open(METRICS_PATH, "w") as f:
            json.dump(all_metrics, f, indent=2)
        log.info("Metrics JSON saved → %s", METRICS_PATH)

        # ── 8. Save plots ─────────────────────────────────────────────────────
        plot_feature_importance(model, FEATURE_COLS, FI_PLOT_PATH)
        pred_plot_path = os.path.join(ROOT_DIR, "reports", "actual_vs_predicted.png")
        plot_predictions(y_test, y_pred_test, pred_plot_path)

        # Log artifacts to MLflow
        mlflow.log_artifact(FI_PLOT_PATH,   artifact_path="plots")
        mlflow.log_artifact(pred_plot_path, artifact_path="plots")
        mlflow.log_artifact(METRICS_PATH,   artifact_path="metrics")

        # ── 9. Save model to disk (for DVC tracking) ──────────────────────────
        joblib.dump(model, MODEL_PATH)
        log.info("Model saved → %s", MODEL_PATH)

        # ── 10. Log model to MLflow model registry ────────────────────────────
        model_uri = mlflow.xgboost.log_model(
            xgb_model   = model,
            artifact_path = "surge_model",
            registered_model_name = "SurgePricingModel",   # registry name
            input_example = X_test.head(3),
        ).model_uri
        log.info("Model logged to MLflow registry as 'SurgePricingModel'")
        log.info("Model URI: %s", model_uri)

        # ── 11. Tag the run with quality status ───────────────────────────────
        r2_threshold   = params["evaluate"]["min_r2"]
        rmse_threshold = params["evaluate"]["max_rmse"]

        passed = (
            test_metrics["r2"]   >= r2_threshold and
            test_metrics["rmse"] <= rmse_threshold
        )
        mlflow.set_tag("quality_gate", "PASSED" if passed else "FAILED")
        mlflow.set_tag("model_type",   "xgboost_regression")
        mlflow.set_tag("phase",        "2")

        if passed:
            log.info("Quality gate ✅ PASSED (R²=%.4f ≥ %.2f | RMSE=%.4f ≤ %.2f)",
                     test_metrics["r2"], r2_threshold,
                     test_metrics["rmse"], rmse_threshold)
        else:
            log.warning("Quality gate ❌ FAILED — check metrics!")

        log.info("MLflow run complete. Run ID: %s", run_id)

    return test_metrics


if __name__ == "__main__":
    params = load_params()
    train(params)
