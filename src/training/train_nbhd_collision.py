from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.io_paths import FEATURES_PATH


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "models" / "registry" / "neighbourhood_collision"
CANDIDATES_DIR = REGISTRY_DIR / "candidates"
CHAMPION_METADATA_PATH = REGISTRY_DIR / "champion" / "metadata.json"

TARGET_COL = "target_collision_t1"

EXCLUDE_COLS = {
    "date",
    "nbhd_id",
    "area_name",
    "target_collision_t1",
    "target_collision_t2",
    "target_ksi_t1",
    "target_ksi_t2",
    "collisions",
    "ksi_collisions",
}


def precision_at_k_by_day(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    k: int,
) -> float:
    scores = []

    for _, day_df in df.groupby("date"):
        top_k = day_df.sort_values(score_col, ascending=False).head(k)
        if len(top_k) == 0:
            continue
        scores.append(top_k[target_col].mean())

    return float(np.mean(scores)) if scores else float("nan")


def load_champion_metric() -> float:
    if not CHAMPION_METADATA_PATH.exists():
        return float("-inf")

    with open(CHAMPION_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return float(metadata.get("metrics", {}).get("precision_at_10", float("-inf")))

def main() -> dict:
    print("=== Stage E Step 3: Train Neighbourhood Collision Candidate ===")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "nbhd_id"])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    LEAKAGE_PATTERNS = [
        "target",
        "proba",
        "risk",
        "surge",
        "_t1",
        "_t2",
    ]

    def is_safe_feature(col: str) -> bool:
        col_lower = col.lower()

        # Remove obvious leakage
        if any(p in col_lower for p in LEAKAGE_PATTERNS):
            return False

        # Remove identifiers
        if col_lower in ["nbhd_id"]:
            return False

        return True

    def build_feature_columns(df: pd.DataFrame) -> list[str]:
        numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

        feature_cols = [c for c in numeric_cols if is_safe_feature(c)]

        return feature_cols

    feature_cols = build_feature_columns(df)

    if not feature_cols:
        raise ValueError("No usable feature columns found.")

    split_date = pd.Timestamp("2024-01-01")
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced empty train or test set.")

    x_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df[TARGET_COL]

    x_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test_df[TARGET_COL]

    base_model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Calibrated model gives better probability reliability for risk scoring.
    model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=3,
    )

    mlflow.set_experiment("covercheck_neighbourhood_collision")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate_dir = CANDIDATES_DIR / run_timestamp
    candidate_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"nbhd_collision_candidate_{run_timestamp}") as run:
        print("Training model...")
        model.fit(x_train, y_train)

        print("Evaluating model...")
        proba = model.predict_proba(x_test)[:, 1]

        roc_auc = float(roc_auc_score(y_test, proba))
        pr_auc = float(average_precision_score(y_test, proba))
        brier = float(brier_score_loss(y_test, proba))

        eval_df = test_df[["date", "nbhd_id", TARGET_COL]].copy()
        eval_df["score"] = proba

        p_at_5 = precision_at_k_by_day(eval_df, "score", TARGET_COL, 5)
        p_at_10 = precision_at_k_by_day(eval_df, "score", TARGET_COL, 10)
        p_at_15 = precision_at_k_by_day(eval_df, "score", TARGET_COL, 15)
        p_at_20 = precision_at_k_by_day(eval_df, "score", TARGET_COL, 20)

        champion_p10 = load_champion_metric()
        beats_champion = bool(p_at_10 >= champion_p10)

        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier": brier,
            "precision_at_5": p_at_5,
            "precision_at_10": p_at_10,
            "precision_at_15": p_at_15,
            "precision_at_20": p_at_20,
            "champion_precision_at_10": champion_p10,
            "beats_champion": int(beats_champion),
        }

        params = {
            "model_type": "LightGBM + CalibratedClassifierCV",
            "target_col": TARGET_COL,
            "split_date": str(split_date.date()),
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "class_weight": "balanced",
            "feature_count": len(feature_cols),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_text("\n".join(feature_cols), "feature_columns.txt")

        model_path = candidate_dir / "model.pkl"
        metadata_path = candidate_dir / "metadata.json"

        joblib.dump(model, model_path)

        metadata = {
            "model_name": "nbhd_collision_model",
            "model_type": "LightGBM + CalibratedClassifierCV",
            "version": run_timestamp,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_data": str(FEATURES_PATH),
            "target_col": TARGET_COL,
            "split_date": str(split_date.date()),
            "feature_count": len(feature_cols),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "mlflow_run_id": run.info.run_id,
            "metrics": metrics,
            "status": "candidate",
            "beats_current_champion": beats_champion,
            "promotion_rule": "precision_at_10 >= champion precision_at_10",
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metadata_path), artifact_path="metadata")

    summary = {
        "candidate_dir": str(candidate_dir),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "precision_at_10": p_at_10,
        "champion_precision_at_10": champion_p10,
        "beats_champion": beats_champion,
    }

    print("=== Candidate Training Complete ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()