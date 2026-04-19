"""
evaluate_crowd.py — Crowd Estimation Model Evaluation (Sivas)
==============================================================
Loads the trained crowd model and the out-of-time test set, runs predictions,
and reports RMSE, MAE, and the top-5 feature importances.

Usage (from ai-engine/):
  python -m pipelines.crowd_pipeline.evaluate_crowd
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Make sure the ai-engine root is importable regardless of CWD
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CROWD_MODEL_PATH,
    CROWD_TARGET,
    CROWD_X_TEST_PARQUET,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _separator(char: str = "─", width: int = 60) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Step 1: Load model
# ---------------------------------------------------------------------------
def load_model() -> xgb.XGBRegressor:
    """Load the serialised crowd model from disk."""
    if not CROWD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Crowd model not found at {CROWD_MODEL_PATH}. "
            "Run train_crowd.py first."
        )
    model = xgb.XGBRegressor()
    model.load_model(str(CROWD_MODEL_PATH))
    logger.info(f"Model loaded from {CROWD_MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Step 2: Load test data
# ---------------------------------------------------------------------------
def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the out-of-time test Parquet and split X / y."""
    if not CROWD_X_TEST_PARQUET.exists():
        raise FileNotFoundError(
            f"Test data not found at {CROWD_X_TEST_PARQUET}. "
            "Run preprocess_crowd.py first."
        )

    logger.info(f"Loading test data from {CROWD_X_TEST_PARQUET}...")
    df = pd.read_parquet(CROWD_X_TEST_PARQUET)
    logger.info(f"  Test shape : {df.shape}")

    if CROWD_TARGET not in df.columns:
        raise KeyError(f"Target column '{CROWD_TARGET}' not found in test data.")

    X_test = df.drop(columns=[CROWD_TARGET])
    y_test = df[CROWD_TARGET]
    return X_test, y_test


# ---------------------------------------------------------------------------
# Step 3: Predict
# ---------------------------------------------------------------------------
def predict(model: xgb.XGBRegressor, X_test: pd.DataFrame) -> np.ndarray:
    """Run inference on the test set."""
    logger.info("Running inference on test set...")
    y_pred = model.predict(X_test)
    logger.info("  Inference complete.")
    return y_pred


# ---------------------------------------------------------------------------
# Step 4: Calculate and report metrics
# ---------------------------------------------------------------------------
def report_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculate RMSE and MAE, print a formatted report, return values."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    # Mean actual value — used for context
    mean_actual = float(y_true.mean())

    print()
    print(_separator("═"))
    print("  CROWD ESTIMATION — OUT-OF-TIME EVALUATION RESULTS")
    print(f"  Test set  : {len(y_true):,} observations (March 24–30, 2025)")
    print(f"  Target    : '{CROWD_TARGET}'  (passengers waiting at a stop)")
    print(_separator("═"))
    print(f"  {'Metric':<30}  {'Value':>12}")
    print(_separator())
    print(f"  {'RMSE (primary metric)':<30}  {rmse:>11.3f} passengers")
    print(f"  {'MAE':<30}  {mae:>11.3f} passengers")
    print(f"  {'Mean actual passengers':<30}  {mean_actual:>11.2f} passengers")
    print(f"  {'RMSE / Mean actual':<30}  {rmse / mean_actual * 100:>10.1f} %")
    print(_separator("═"))

    return {"rmse": rmse, "mae": mae, "mean_actual": mean_actual}


# ---------------------------------------------------------------------------
# Step 5: Feature importances
# ---------------------------------------------------------------------------
def report_feature_importances(model: xgb.XGBRegressor, feature_names: list[str]) -> None:
    """Print top-5 feature importances ranked by gain."""
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print()
    print(_separator("─"))
    print("  TOP-5 FEATURE IMPORTANCES  (by XGBoost internal gain)")
    print(_separator("─"))
    for rank, (feat, score) in enumerate(feat_imp.head(5).items(), start=1):
        bar = "█" * int(score * 40)
        print(f"  {rank}. {feat:<30}  {score:.4f}  {bar}")
    print(_separator("─"))
    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("Crowd Estimation — Model Evaluation (Sivas)")
    logger.info("=" * 60)

    try:
        model            = load_model()
        X_test, y_test   = load_test_data()
        y_pred           = predict(model, X_test)
        report_metrics(y_test, y_pred)
        report_feature_importances(model, list(X_test.columns))
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}", exc_info=True)
        sys.exit(1)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
