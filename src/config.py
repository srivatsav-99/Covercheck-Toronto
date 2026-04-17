import yaml
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

# Load YAML Configuration
with open(CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

FORECAST_HORIZONS = _cfg["data"]["forecast_horizons"]
SPLIT_DATE = _cfg["data"]["split_date"]
SURGE_PCT = _cfg["data"]["surge_threshold_pct"]
MODEL_CONFIG = _cfg["models"]

# Constants
NBHD_TOP_K_DEFAULT = 10
RISK_BANDS = {"high": 0.70, "medium": 0.40}
PIPELINE_RANDOM_STATE = 42

# MLflow Tracking
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
MLFLOW_EXPERIMENT_NAME = "covercheck-toronto-v1"