from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.io_paths import FEATURES_PATH, SURGE_PRED_PATH, NBHD_PRED_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]

CITY_MODEL_PATH = REPO_ROOT / "models" / "citywide_model.pkl"
NBHD_MODEL_PATH = REPO_ROOT / "models" / "nbhd_collision_model.pkl"

SURGE_THRESHOLD = 0.50


def _clean_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:20]}")

    return (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def _positive_probability(model, x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model does not support predict_proba: {type(model)}")

    proba = model.predict_proba(x)

    if isinstance(proba, list):
        proba = proba[0]

    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]

    if proba.ndim == 1:
        return proba

    raise ValueError(f"Unexpected probability output shape: {getattr(proba, 'shape', None)}")


def score_latest(
    features_path: Path = FEATURES_PATH,
    surge_pred_path: Path = SURGE_PRED_PATH,
    nbhd_pred_path: Path = NBHD_PRED_PATH,
    city_model_path: Path = CITY_MODEL_PATH,
    nbhd_model_path: Path = NBHD_MODEL_PATH,
) -> dict:
    print("=== Stage D: Score Latest Forecasts ===")

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    if not city_model_path.exists():
        raise FileNotFoundError(f"Missing citywide model: {city_model_path}")

    if not nbhd_model_path.exists():
        raise FileNotFoundError(f"Missing neighbourhood model: {nbhd_model_path}")

    print(f"Loading features: {features_path}")
    features = pd.read_parquet(features_path).copy()
    features["date"] = pd.to_datetime(features["date"])

    print(f"Features shape: {features.shape}")

    print(f"Loading citywide model: {city_model_path}")
    city_model = joblib.load(city_model_path)
    city_feature_cols = [str(c) for c in city_model.feature_names_in_]

    city_daily = (
        features.sort_values(["date", "nbhd_id"])
        .groupby("date", as_index=False)
        .tail(1)
        .sort_values("date")
        .copy()
    )

    x_city = _clean_features(city_daily, city_feature_cols)
    surge_base = _positive_probability(city_model, x_city)

    surge_df = pd.DataFrame(
        {
            "date": city_daily["date"].values,
            "surge_proba_t1": surge_base,
        }
    )

    # Practical T+2 proxy:
    # use next-row shifted citywide probability when possible.
    surge_df["surge_proba_t2"] = surge_df["surge_proba_t1"].shift(-1)

    surge_df["surge_label_t1"] = (surge_df["surge_proba_t1"] >= SURGE_THRESHOLD).astype(int)
    surge_df["surge_label_t2"] = np.where(
        surge_df["surge_proba_t2"].notna(),
        (surge_df["surge_proba_t2"] >= SURGE_THRESHOLD).astype(int),
        np.nan,
    )

    surge_df["threshold"] = SURGE_THRESHOLD
    surge_df["created_at"] = datetime.now(timezone.utc).isoformat()
    surge_df["model_version"] = city_model_path.name

    surge_df = surge_df[
        [
            "date",
            "surge_proba_t1",
            "surge_label_t1",
            "threshold",
            "surge_proba_t2",
            "surge_label_t2",
            "created_at",
            "model_version",
        ]
    ]

    print(f"Writing surge predictions: {surge_pred_path}")
    surge_pred_path.parent.mkdir(parents=True, exist_ok=True)
    surge_df.to_parquet(surge_pred_path, index=False)

    print(f"Loading neighbourhood model: {nbhd_model_path}")
    nbhd_model = joblib.load(nbhd_model_path)
    nbhd_feature_cols = [str(c) for c in nbhd_model.feature_names_in_]

    features_plus = features.merge(
        surge_df[["date", "surge_proba_t1", "surge_proba_t2"]],
        on="date",
        how="left",
    )

    x_nbhd = _clean_features(features_plus, nbhd_feature_cols)
    collision_t1 = _positive_probability(nbhd_model, x_nbhd)

    # Practical T+2 proxy:
    # shift each neighbourhood's T+1 probability backward one date.
    nbhd_scored = features_plus[["date", "nbhd_id"]].copy()
    nbhd_scored["proba_collision_t1"] = collision_t1
    nbhd_scored = nbhd_scored.sort_values(["nbhd_id", "date"])
    nbhd_scored["proba_collision_t2"] = nbhd_scored.groupby("nbhd_id")["proba_collision_t1"].shift(-1)

    if "target_collision_t1" in features_plus.columns:
        nbhd_scored["target_collision_t1"] = features_plus["target_collision_t1"].values
    else:
        nbhd_scored["target_collision_t1"] = np.nan

    if "target_collision_t2" in features_plus.columns:
        nbhd_scored["target_collision_t2"] = features_plus["target_collision_t2"].values
    else:
        nbhd_scored["target_collision_t2"] = np.nan

    # KSI is kept as operationally unavailable / experimental.
    nbhd_scored["target_ksi_t1"] = features_plus.get("target_ksi_t1", pd.Series(np.nan, index=features_plus.index)).values
    nbhd_scored["target_ksi_t2"] = features_plus.get("target_ksi_t2", pd.Series(np.nan, index=features_plus.index)).values
    nbhd_scored["proba_ksi_t1"] = 0.0
    nbhd_scored["proba_ksi_t2"] = 0.0

    # Keep the same risk-score concept as dashboard-friendly blended score.
    nbhd_scored["risk_score_t1"] = (0.60 * nbhd_scored["proba_collision_t1"]) + (0.40 * nbhd_scored["proba_ksi_t1"])
    nbhd_scored["risk_score_t2"] = (0.60 * nbhd_scored["proba_collision_t2"]) + (0.40 * nbhd_scored["proba_ksi_t2"])

    nbhd_scored["created_at"] = datetime.now(timezone.utc).isoformat()
    nbhd_scored["model_version"] = nbhd_model_path.name

    nbhd_scored = nbhd_scored[
        [
            "date",
            "nbhd_id",
            "target_collision_t1",
            "target_ksi_t1",
            "proba_collision_t1",
            "proba_ksi_t1",
            "risk_score_t1",
            "target_collision_t2",
            "target_ksi_t2",
            "proba_collision_t2",
            "proba_ksi_t2",
            "risk_score_t2",
            "created_at",
            "model_version",
        ]
    ].sort_values(["date", "nbhd_id"])

    print(f"Writing neighbourhood predictions: {nbhd_pred_path}")
    nbhd_pred_path.parent.mkdir(parents=True, exist_ok=True)
    nbhd_scored.to_parquet(nbhd_pred_path, index=False)

    summary = {
        "features_rows": int(len(features)),
        "surge_rows": int(len(surge_df)),
        "nbhd_rows": int(len(nbhd_scored)),
        "latest_surge_date": str(surge_df["date"].max().date()),
        "latest_nbhd_date": str(nbhd_scored["date"].max().date()),
    }

    print("=== Scoring Complete ===")
    print(summary)
    return summary


if __name__ == "__main__":
    score_latest()