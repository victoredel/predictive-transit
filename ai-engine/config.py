import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).parent.absolute()
ACTIVE_DATASET = os.getenv("ACTIVE_DATASET", "boston").lower()

# Dynamic path configuration
DATASETS_DIR = BASE_DIR / "datasets" / ACTIVE_DATASET
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / ACTIVE_DATASET

# Ensure output directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset-specific metadata
if ACTIVE_DATASET == "boston":
    TARGET = "tiempo_viaje_segundos"
    CATEGORICAL_FEATURES = ["route_id", "direction_id", "stop_id_origen", "stop_id_destino"]
    MODEL_NAME = "transit_xgboost_boston.json"
elif ACTIVE_DATASET == "sivas":
    TARGET = "delay_min"
    CATEGORICAL_FEATURES = ["line_id", "stop_id", "traffic_level", "weather_condition"]
    MODEL_NAME = "transit_xgboost_sivas.json"
else:
    raise ValueError(f"Unsupported ACTIVE_DATASET: {ACTIVE_DATASET}")

MODEL_PATH = MODELS_DIR / MODEL_NAME

# Parquet file paths (Processed)
X_TRAIN_PARQUET = PROCESSED_DATA_DIR / "X_train_pro.parquet"
X_TEST_PARQUET  = PROCESSED_DATA_DIR / "X_test_pro.parquet"

# Global Constants from preprocess (can be moved here if needed)
MAX_LOOKAHEAD_STOPS = 10
MAX_TRAVEL_TIME_S   = 7200

# ===========================================================================
# Crowd Estimation Pipeline — Sivas dataset
# ===========================================================================
# These paths are always relative to the Sivas dataset, regardless of
# ACTIVE_DATASET, because the crowd model is Sivas-specific.

_SIVAS_PROCESSED  = BASE_DIR / "datasets" / "sivas" / "processed"
_SIVAS_MODELS_DIR = BASE_DIR / "models"   / "sivas"

# Ensure crowd output dirs exist
_SIVAS_PROCESSED.mkdir(parents=True, exist_ok=True)
_SIVAS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Target variable for crowd model
CROWD_TARGET = "passengers_waiting"

# Parquet paths produced by preprocess_crowd.py
CROWD_X_TRAIN_PARQUET = _SIVAS_PROCESSED / "crowd_X_train.parquet"
CROWD_X_TEST_PARQUET  = _SIVAS_PROCESSED / "crowd_X_test.parquet"

# Serialised model produced by train_crowd.py
CROWD_MODEL_PATH = _SIVAS_MODELS_DIR / "transit_xgboost_crowd_sivas.json"

# Categorical feature columns for the crowd XGBoost model
CROWD_CATEGORICAL_FEATURES = [
    "weather_condition",
    "traffic_level",
    "stop_type",
    "line_id",
    "stop_id",
]
