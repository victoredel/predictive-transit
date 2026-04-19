"""
preprocess_crowd.py — Crowd Estimation Data Pipeline (Sivas)
=============================================================
Produces two Parquet files used by train_crowd.py:
  - crowd_X_train.parquet  (arrivals < 2025-03-24)
  - crowd_X_test.parquet   (arrivals >= 2025-03-24)

Usage (from ai-engine/):
  python -m pipelines.crowd_pipeline.preprocess_crowd

Data sources  (datasets/sivas/raw/):
  - stop_arrivals.csv    — one row per observed bus arrival; contains target
  - passenger_flow.csv   — aggregated historical baseline per stop/hour/day
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the ai-engine root is importable regardless of CWD
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CROWD_CATEGORICAL_FEATURES,
    CROWD_TARGET,
    CROWD_X_TEST_PARQUET,
    CROWD_X_TRAIN_PARQUET,
    RAW_DATA_DIR,
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
# Constants
# ---------------------------------------------------------------------------
# Temporal split boundary (ISO date string)
TRAIN_TEST_CUTOFF = "2025-03-24"

# Features to retain after the merge — target is extracted separately
FEATURE_COLS = [
    # From passenger_flow (historical baseline)
    "avg_passengers_waiting",
    # Shared context
    "weather_condition",
    "traffic_level",
    "hour_of_day",
    "day_of_week",
    "stop_type",
    "line_id",
    "stop_id",
    # Model chaining: ETA model output → Crowd model input
    # A delayed bus accumulates more waiting passengers at each stop.
    "cumulative_delay_min",
]

# Columns that would cause data leakage if left in the dataset
LEAKAGE_COLS = [
    "passengers_boarding",
    "passengers_alighting",
    "crowding_level",
    # Other post-hoc observations that are unknown at prediction time
    "dwell_time_min",
    "is_delayed",
    "speed_factor",
    "minutes_to_next_bus",
]


# ---------------------------------------------------------------------------
# Step 1: Load raw data
# ---------------------------------------------------------------------------
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load stop_arrivals.csv and passenger_flow.csv from the raw data dir."""
    arrivals_path = RAW_DATA_DIR / "stop_arrivals.csv"
    flow_path     = RAW_DATA_DIR / "passenger_flow.csv"

    logger.info("Loading stop_arrivals.csv...")
    arrivals = pd.read_csv(arrivals_path, parse_dates=["actual_arrival", "planned_arrival"])

    logger.info("Loading passenger_flow.csv...")
    flow = pd.read_csv(flow_path)

    logger.info(f"  stop_arrivals  : {len(arrivals):,} rows x {arrivals.shape[1]} cols")
    logger.info(f"  passenger_flow : {len(flow):,} rows x {flow.shape[1]} cols")
    return arrivals, flow


# ---------------------------------------------------------------------------
# Step 2: Merge on the shared contextual keys
# ---------------------------------------------------------------------------
def merge_datasets(arrivals: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join stop_arrivals with passenger_flow on the contextual keys.
    Using a left join guarantees every arrival observation is retained even
    if the passenger_flow baseline is missing for that combination.
    """
    join_keys = ["stop_id", "line_id", "hour_of_day", "day_of_week", "weather_condition"]

    # passenger_flow may have overlapping columns; only bring in what is needed
    flow_cols = join_keys + ["avg_passengers_waiting"]
    flow_slim = flow[flow_cols].drop_duplicates(subset=join_keys)

    logger.info(f"Merging on keys: {join_keys}")
    merged = arrivals.merge(flow_slim, on=join_keys, how="left")

    # Fill missing avg_passengers_waiting with 0 (conservative baseline)
    merged["avg_passengers_waiting"] = merged["avg_passengers_waiting"].fillna(0.0)
    logger.info(f"  Merged shape: {merged.shape}")

    return merged


# ---------------------------------------------------------------------------
# Step 3: Feature selection + data leakage protection
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop leakage columns, select model features and target, enforce dtypes.
    """
    logger.info("Dropping leakage columns...")
    existing_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=existing_leakage)

    # Keep only feature columns + target + the arrival timestamp for splitting
    keep = FEATURE_COLS + [CROWD_TARGET, "actual_arrival"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns missing from dataset: {missing}")

    df = df[keep].copy()

    # Drop rows with a null target
    null_target = df[CROWD_TARGET].isna().sum()
    if null_target:
        logger.warning(f"Dropping {null_target:,} rows with null target '{CROWD_TARGET}'.")
        df = df.dropna(subset=[CROWD_TARGET])

    # Impute cumulative_delay_min: a missing value means no delay was recorded → 0.0
    if "cumulative_delay_min" in df.columns:
        nan_delay = df["cumulative_delay_min"].isna().sum()
        if nan_delay:
            logger.warning(f"Filling {nan_delay:,} NaN values in 'cumulative_delay_min' with 0.0.")
        df["cumulative_delay_min"] = df["cumulative_delay_min"].fillna(0.0)

    # Cast categoricals
    for col in CROWD_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Cast numeric features to float32 for memory efficiency
    numeric_cols = [c for c in FEATURE_COLS if c not in CROWD_CATEGORICAL_FEATURES]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    df[CROWD_TARGET] = df[CROWD_TARGET].astype("float32")

    logger.info(f"  Final feature set: {FEATURE_COLS}")
    logger.info(f"  Target           : '{CROWD_TARGET}'")
    return df



# ---------------------------------------------------------------------------
# Step 4: Temporal split
# ---------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict temporal split on actual_arrival date.
      Train : actual_arrival < TRAIN_TEST_CUTOFF
      Test  : actual_arrival >= TRAIN_TEST_CUTOFF
    The timestamp column is dropped afterwards (not a model feature).
    """
    cutoff = pd.Timestamp(TRAIN_TEST_CUTOFF)
    train_df = df[df["actual_arrival"] < cutoff].drop(columns=["actual_arrival"]).reset_index(drop=True)
    test_df  = df[df["actual_arrival"] >= cutoff].drop(columns=["actual_arrival"]).reset_index(drop=True)

    logger.info(f"Temporal split at {TRAIN_TEST_CUTOFF}:")
    logger.info(f"  Train : {len(train_df):,} rows ({train_df['actual_arrival'].min() if 'actual_arrival' in train_df else 'N/A'} – ...)")
    logger.info(f"  Test  : {len(test_df):,} rows")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Step 5: Save Parquet
# ---------------------------------------------------------------------------
def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Persist train and test splits as Parquet files."""
    logger.info(f"Saving train split → {CROWD_X_TRAIN_PARQUET}")
    train_df.to_parquet(CROWD_X_TRAIN_PARQUET, index=False)

    logger.info(f"Saving test split  → {CROWD_X_TEST_PARQUET}")
    test_df.to_parquet(CROWD_X_TEST_PARQUET, index=False)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("Crowd Estimation — Preprocessing Pipeline (Sivas)")
    logger.info("=" * 60)

    try:
        arrivals, flow       = load_raw_data()
        merged               = merge_datasets(arrivals, flow)
        features_df          = prepare_features(merged)
        train_df, test_df    = temporal_split(features_df)
        save_splits(train_df, test_df)
    except Exception as exc:
        logger.error(f"Preprocessing failed: {exc}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Preprocessing complete. Train: {len(train_df):,} | Test: {len(test_df):,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
