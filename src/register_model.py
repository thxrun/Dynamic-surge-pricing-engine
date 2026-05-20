"""
src/register_model.py
----------------------
Queries the MLflow Model Registry, finds the latest model version that
passed quality gates, and transitions it to the 'Staging' stage.

In a real team workflow this script would run after CI passes, promoting
the winning model so the serving layer (Phase 4) can load it from
the registry rather than from disk.

Run:
    python src/register_model.py
"""

import logging
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_NAME = "surge-pricing-xgboost"
REGISTRY_NAME   = "SurgePricingModel"


def promote_best_model() -> None:
    """
    Find the most recent run tagged quality_gate=PASSED and
    transition that model version to 'Staging'.
    """
    mlflow.set_tracking_uri(f"file://{os.path.join(ROOT_DIR, 'mlruns')}")
    client = MlflowClient()

    # ── 1. Get the experiment ─────────────────────────────────────────────────
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        log.error("Experiment '%s' not found. Run train.py first.", EXPERIMENT_NAME)
        sys.exit(1)

    # ── 2. Find runs that passed the quality gate ─────────────────────────────
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.quality_gate = 'PASSED'",
        order_by=["metrics.test_r2 DESC"],   # best R² first
        max_results=1,
    )

    if not runs:
        log.error("No PASSED runs found. Retrain the model first.")
        sys.exit(1)

    best_run = runs[0]
    run_id   = best_run.info.run_id
    r2       = best_run.data.metrics.get("test_r2", "N/A")
    rmse     = best_run.data.metrics.get("test_rmse", "N/A")

    log.info("Best qualifying run found:")
    log.info("  Run ID : %s", run_id)
    log.info("  R²     : %s", r2)
    log.info("  RMSE   : %s", rmse)

    # ── 3. Get the model version linked to this run ────────────────────────────
    versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    matching = [v for v in versions if v.run_id == run_id]

    if not matching:
        log.error("No model version found for run %s in registry.", run_id)
        sys.exit(1)

    model_version = matching[0].version
    log.info("Model version: %s", model_version)

    # ── 4. Transition to Staging ───────────────────────────────────────────────
    client.transition_model_version_stage(
        name    = REGISTRY_NAME,
        version = model_version,
        stage   = "Staging",
        archive_existing_versions = True,   # demote any previous Staging model
    )

    log.info(
        "Model '%s' v%s → promoted to 'Staging' ✅",
        REGISTRY_NAME, model_version
    )
    log.info(
        "To load this model:\n"
        "  model = mlflow.xgboost.load_model('models:/%s/Staging')",
        REGISTRY_NAME,
    )


if __name__ == "__main__":
    promote_best_model()
