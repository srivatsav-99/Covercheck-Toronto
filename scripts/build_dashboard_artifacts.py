from src.io_paths import PROCESSED_DIR, FEATURES_PATH, NBHD_PRED_PATH
import pandas as pd

features = pd.read_parquet(FEATURES_PATH)

needed_cols = [
    "date", "nbhd_id", "collisions", "ksi_collisions",
    "collisions_lag1", "collisions_roll30_mean", "collision_momentum",
    "tavg", "prcp", "snow", "wspd",
    "freezing_rain", "heavy_snow", "feels_bad", "wind_rain",
    "is_holiday", "pre_holiday", "post_holiday", "is_weekend",
    "road_construction_count_lag1", "is_zone_disrupted",
]

keep = [c for c in needed_cols if c in features.columns]
features_slim = features[keep].copy()
features_slim.to_parquet(PROCESSED_DIR / "dashboard_features_slim.parquet", index=False)

nbhd = pd.read_parquet(NBHD_PRED_PATH)
nbhd["date"] = pd.to_datetime(nbhd["date"])
latest_date = nbhd["date"].max()

nbhd_latest = nbhd[nbhd["date"] == latest_date].copy()
nbhd_latest.to_parquet(PROCESSED_DIR / "dashboard_nbhd_latest.parquet", index=False)

print("Created dashboard_features_slim.parquet", features_slim.shape)
print("Created dashboard_nbhd_latest.parquet", nbhd_latest.shape)