"""
train_crowd.py — Crowd Estimation Model Training (Sivas)
=========================================================
Trains an XGBoost regressor to predict `passengers_waiting` at a stop.

The model is trained on crowd_X_train.parquet, which was produced by
preprocess_crowd.py using a strict temporal split (train < 2025-03-24).

Usage (from ai-engine/):
  python -m pipelines.crowd_pipeline.train_crowd
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Make sure the ai-engine root is importable regardless of CWD
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CROWD_CATEGORICAL_FEATURES,
    CROWD_MODEL_PATH,
    CROWD_TARGET,
    CROWD_X_TRAIN_PARQUET,
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
# XGBoost Hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "objective":          "reg:squarederror",
    "enable_categorical": True,   # Native categorical support — no one-hot encoding needed
    "tree_method":        "hist", # Fast histogram-based tree builder
    "n_estimators":       200,
    "max_depth":          6,
    "learning_rate":      0.05,
    "subsample":          0.85,
    "colsample_bytree":   0.85,
    "min_child_weight":   5,
    "random_state":       42,
    "n_jobs":             -1,
}


# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------
def load_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the preprocessed training Parquet and split X / y."""
    if not CROWD_X_TRAIN_PARQUET.exists():
        raise FileNotFoundError(
            f"Training data not found at {CROWD_X_TRAIN_PARQUET}. "
            "Run preprocess_crowd.py first."
        )

    logger.info(f"Loading training data from {CROWD_X_TRAIN_PARQUET}...")
    df = pd.read_parquet(CROWD_X_TRAIN_PARQUET)
    logger.info(f"  Shape: {df.shape}")

    if CROWD_TARGET not in df.columns:
        raise KeyError(f"Target column '{CROWD_TARGET}' not found in training data.")

    X = df.drop(columns=[CROWD_TARGET])
    y = df[CROWD_TARGET]
    return X, y


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """Initialise and fit the XGBoost regressor."""
    model = xgb.XGBRegressor(**XGB_PARAMS)

    logger.info("Training XGBoost Crowd Estimation model...")
    logger.info(f"  Features  : {list(X.columns)}")
    logger.info(f"  Target    : '{CROWD_TARGET}'")
    logger.info(f"  Samples   : {len(X):,}")

    model.fit(X, y)
    logger.info("Training complete.")
    return model


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------
def save_model(model: xgb.XGBRegressor) -> None:
    """Persist the trained model to disk in JSON format."""
    CROWD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(CROWD_MODEL_PATH))
    logger.info(f"Model saved → {CROWD_MODEL_PATH}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("Crowd Estimation — Model Training (Sivas)")
    logger.info("=" * 60)

    try:
        X, y  = load_train_data()
        model = train(X, y)
        save_model(model)
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Training pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
