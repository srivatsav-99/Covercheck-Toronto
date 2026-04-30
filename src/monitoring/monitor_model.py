from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.io_paths import (
    FEATURES_PATH,
    SURGE_PRED_PATH,
    NBHD_PRED_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "docs" / "mlops" / "monitoring_report.json"

ALERT_REPORT_PATH = REPO_ROOT / "docs" / "mlops" / "monitoring_alerts.json"

FRESHNESS_THRESHOLD_DAYS = 7
SURGE_AVG_MIN = 0.00
SURGE_AVG_MAX = 0.80
NBHD_AVG_MIN = 0.05
NBHD_AVG_MAX = 0.95


def load_data():
    features = pd.read_parquet(FEATURES_PATH)
    surge = pd.read_parquet(SURGE_PRED_PATH)
    nbhd = pd.read_parquet(NBHD_PRED_PATH)

    return features, surge, nbhd


# -------------------------
# DATA DRIFT CHECK
# -------------------------
def check_data_drift(features: pd.DataFrame):
    exclude_cols = {"nbhd_id"}
    numeric_cols = [
        c for c in features.select_dtypes(include="number").columns
        if c not in exclude_cols and not c.startswith("target_")
    ]

    stats = []

    for col in numeric_cols[:10]:  # limit for simplicity
        mean_val = features[col].mean()
        std_val = features[col].std()

        stats.append({
            "feature": col,
            "mean": float(mean_val),
            "std": float(std_val)
        })

    return stats


# -------------------------
# PREDICTION DRIFT
# -------------------------
def check_prediction_drift(surge: pd.DataFrame, nbhd: pd.DataFrame):
    latest_surge = surge.tail(30)
    latest_nbhd = nbhd.tail(1000)

    nbhd_score_col = (
        "proba_collision_t1"
        if "proba_collision_t1" in latest_nbhd.columns
        else "risk_score_t1"
    )

    return {
        "surge_avg": float(latest_surge["surge_proba_t1"].mean()),
        "nbhd_avg": float(latest_nbhd[nbhd_score_col].mean()),
        "nbhd_score_column_used": nbhd_score_col,
    }


# -------------------------
# FRESHNESS CHECK
# -------------------------
def check_freshness(surge: pd.DataFrame, nbhd: pd.DataFrame):
    latest_surge_date = pd.to_datetime(surge["date"]).max()
    latest_nbhd_date = pd.to_datetime(nbhd["date"]).max()

    today = pd.Timestamp.today()

    return {
        "surge_last_date": str(latest_surge_date),
        "nbhd_last_date": str(latest_nbhd_date),
        "days_since_update": int((today - latest_surge_date).days),
    }


# -------------------------
# MAIN MONITOR
# -------------------------
def generate_alerts(report: dict) -> dict:
    alerts = []

    freshness = report.get("freshness", {})
    prediction_drift = report.get("prediction_drift", {})

    days_since_update = freshness.get("days_since_update")
    surge_avg = prediction_drift.get("surge_avg")
    nbhd_avg = prediction_drift.get("nbhd_avg")

    if days_since_update is not None and days_since_update > FRESHNESS_THRESHOLD_DAYS:
        alerts.append(
            {
                "level": "HIGH",
                "type": "artifact_freshness",
                "message": (
                    f"Prediction artifacts are stale: {days_since_update} days since last update. "
                    f"Threshold is {FRESHNESS_THRESHOLD_DAYS} days."
                ),
                "recommended_action": "Run scoring refresh or retraining pipeline.",
            }
        )

    if surge_avg is not None and not (SURGE_AVG_MIN <= surge_avg <= SURGE_AVG_MAX):
        alerts.append(
            {
                "level": "MEDIUM",
                "type": "surge_prediction_drift",
                "message": (
                    f"Average surge probability is outside expected range: {surge_avg:.4f}."
                ),
                "recommended_action": "Inspect recent surge predictions and input features.",
            }
        )

    if nbhd_avg is not None and not (NBHD_AVG_MIN <= nbhd_avg <= NBHD_AVG_MAX):
        alerts.append(
            {
                "level": "MEDIUM",
                "type": "neighbourhood_prediction_drift",
                "message": (
                    f"Average neighbourhood collision probability is outside expected range: {nbhd_avg:.4f}."
                ),
                "recommended_action": "Inspect neighbourhood prediction distribution.",
            }
        )

    status = "PASS" if not alerts else "ALERT"

    return {
        "timestamp": report.get("timestamp"),
        "status": status,
        "alert_count": len(alerts),
        "alerts": alerts,
    }

def main():
    print("=== Monitoring Model ===")

    features, surge, nbhd = load_data()

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_drift": check_data_drift(features),
        "prediction_drift": check_prediction_drift(surge, nbhd),
        "freshness": check_freshness(surge, nbhd),
    }

    alerts = generate_alerts(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("Monitoring report saved:", REPORT_PATH)
    print(json.dumps(report, indent=2))

    with open(ALERT_REPORT_PATH, "w") as f:
        json.dump(alerts, f, indent=2)

    print("Monitoring alerts saved:", ALERT_REPORT_PATH)
    print(json.dumps(alerts, indent=2))


if __name__ == "__main__":
    main()