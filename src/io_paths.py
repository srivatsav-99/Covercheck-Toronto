from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = REPO_ROOT / "docs"
ASSETS_DIR = REPO_ROOT / "assets"

DIM_PATH = INTERIM_DIR / "dim_neighbourhoods.parquet"

GOLD_PATH = PROCESSED_DIR / "gold_nbhd_day_weather_511.parquet"
if not GOLD_PATH.exists():
    fallback = PROCESSED_DIR / "gold_nbhd_day_weather.parquet"
    if fallback.exists():
        GOLD_PATH = fallback

FEATURES_PATH = PROCESSED_DIR / "features_v2.parquet"
SURGE_PRED_PATH = PROCESSED_DIR / "surge_predictions.parquet"
NBHD_PRED_PATH = PROCESSED_DIR / "nbhd_predictions.parquet"

SURGE_METRICS_PATH = DOCS_DIR / "surge_metrics.json"
NBHD_METRICS_PATH = DOCS_DIR / "nbhd_metrics.json"

COLLISION_FI_PATH = DOCS_DIR / "collision_feature_importance.csv"
KSI_FI_PATH = DOCS_DIR / "ksi_feature_importance.csv"

DASHBOARD_FEATURES_PATH = PROCESSED_DIR / "dashboard_features_slim.parquet"
DASHBOARD_NBHD_LATEST_PATH = PROCESSED_DIR / "dashboard_nbhd_latest.parquet"