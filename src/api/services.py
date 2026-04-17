import json
from typing import List, Dict, Any

import pandas as pd

from src.io_paths import (
    DIM_PATH,
    SURGE_PRED_PATH,
    NBHD_PRED_PATH,
    SURGE_METRICS_PATH,
    NBHD_METRICS_PATH,
)


def _load_dim_names() -> pd.DataFrame:
    if not DIM_PATH.exists():
        return pd.DataFrame(columns=["nbhd_id", "area_name"])

    dim = pd.read_parquet(DIM_PATH).copy()

    cols = [c for c in ["nbhd_id", "area_name"] if c in dim.columns]
    if len(cols) < 2:
        return pd.DataFrame(columns=["nbhd_id", "area_name"])

    out = dim[["nbhd_id", "area_name"]].copy()
    out["nbhd_id"] = pd.to_numeric(out["nbhd_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["nbhd_id"]).drop_duplicates(subset=["nbhd_id"])
    out["nbhd_id"] = out["nbhd_id"].astype(int)
    return out


def get_latest_surge() -> Dict[str, Any]:
    if not SURGE_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing surge predictions file: {SURGE_PRED_PATH}")

    df = pd.read_parquet(SURGE_PRED_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()

    latest = (
        df.loc[df["date"] == latest_date]
        .sort_values("date")
        .iloc[0]
        .to_dict()
    )

    return {
        "date": pd.Timestamp(latest_date).strftime("%Y-%m-%d"),
        "surge_proba_t1": float(latest["surge_proba_t1"]) if "surge_proba_t1" in latest and pd.notna(latest["surge_proba_t1"]) else None,
        "surge_proba_t2": float(latest["surge_proba_t2"]) if "surge_proba_t2" in latest and pd.notna(latest["surge_proba_t2"]) else None,
    }


def get_topk_neighbourhoods(horizon: int = 1, k: int = 10) -> Dict[str, Any]:
    if horizon not in (1, 2):
        raise ValueError("horizon must be 1 or 2")

    if not NBHD_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing neighbourhood predictions file: {NBHD_PRED_PATH}")

    df = pd.read_parquet(NBHD_PRED_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])

    latest_date = df["date"].max()

    # DYNAMIC TARGET: Check for risk_score first, fallback to collision_prob
    target_col = f"risk_score_t{horizon}"
    if target_col not in df.columns:
        target_col = f"collision_prob_t{horizon}"
        if target_col not in df.columns:
            raise ValueError(f"Could not find risk or collision column in: {df.columns.tolist()}")

    dim_names = _load_dim_names()

    latest = df.loc[df["date"] == latest_date].copy()
    latest["nbhd_id"] = pd.to_numeric(latest["nbhd_id"], errors="coerce").astype(int)
    latest = latest.merge(dim_names, on="nbhd_id", how="left")

    latest = latest.sort_values(target_col, ascending=False).reset_index(drop=True)
    latest[f"rank_t{horizon}"] = latest.index + 1

    out = latest.head(k).copy()

    records = []
    for _, row in out.iterrows():
        # Safely extract either the collision prob or the risk score for the frontend
        val_t1 = row.get("collision_prob_t1", row.get("risk_score_t1"))
        val_t2 = row.get("collision_prob_t2", row.get("risk_score_t2"))

        records.append({
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "nbhd_id": int(row["nbhd_id"]),
            "area_name": row["area_name"] if "area_name" in row and pd.notna(row["area_name"]) else None,
            "collision_prob_t1": float(val_t1) if pd.notna(val_t1) else None,
            "collision_prob_t2": float(val_t2) if pd.notna(val_t2) else None,
            "rank_t1": int(row["rank_t1"]) if horizon == 1 else None,
            "rank_t2": int(row["rank_t2"]) if horizon == 2 else None,
        })

    return {
        "horizon": horizon,
        "k": k,
        "as_of_date": pd.Timestamp(latest_date).strftime("%Y-%m-%d"),
        "records": records,
    }

def get_metrics() -> Dict[str, List[Dict[str, Any]]]:
    surge_metrics = []
    nbhd_metrics = []

    if SURGE_METRICS_PATH.exists():
        with open(SURGE_METRICS_PATH, "r", encoding="utf-8") as f:
            surge_metrics = json.load(f)

    if NBHD_METRICS_PATH.exists():
        with open(NBHD_METRICS_PATH, "r", encoding="utf-8") as f:
            nbhd_metrics = json.load(f)

    return {
        "surge_metrics": surge_metrics,
        "nbhd_metrics": nbhd_metrics,
    }