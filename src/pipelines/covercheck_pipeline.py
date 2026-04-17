#%% md
# CoverCheck Toronto — Full Pipeline (single file)
# =================================================
# Replaces Phases 00-13 of the original notebook pipeline.
#  
# Sections
# --------
#   SECTION 1  Paths & configuration
#   SECTION 2  Feature builder          (replaces Phase 05, 10, 13)
#                2a  OSM road-network static features
#                2b  Ontario public holiday features
#                2c  Weather interaction features
#                2d  KSI severity ratio
#                2e  Lags & rolling windows
#                2f  Collision momentum
#                2g  Halo (neighbour-zone spillover) features
#                2h  Multi-horizon targets
#   SECTION 3  Citywide surge classifier (replaces Phase 06, 11)
#   SECTION 4  Neighbourhood models      (replaces Phase 07, 12)
#                4a  Collision probability  P(≥1 collision in T+h days)
#                4b  KSI severity           P(≥1 KSI collision in T+h days)
#                4c  Risk ranking           Precision@K / Recall@K / Lift
#   SECTION 5  Pipeline runner
#  
# Inputs  (your existing processed outputs — nothing new to download)
# -------
#   data/processed/gold_nbhd_day_weather_511.parquet
#   data/interim/nbhd_adjacency.parquet
#   data/interim/dim_neighbourhoods.parquet
#  
# New data sources added
# ----------------------
#   Ontario public holidays   → pip install holidays   (no download)
#   OSM road network          → pip install osmnx      (fetched once, cached)
#  
# Outputs
# -------
#   data/interim/road_network_static.parquet    OSM cache (one-time)
#   data/processed/features_v2.parquet          full feature table
#   data/processed/surge_predictions.parquet    citywide surge probabilities
#   data/processed/nbhd_predictions.parquet     per-neighbourhood predictions
#   docs/surge_metrics.json
#   docs/nbhd_metrics.json
#   docs/fi_collision_t{h}.csv                  feature importances
#   docs/fi_ksi_t{h}.csv
#  
# Usage
# -----
#   python covercheck_pipeline.py               # run full pipeline
#   python covercheck_pipeline.py --from surge  # skip feature build
#   python covercheck_pipeline.py --only nbhd   # run only neighbourhood models
#  
# Install
# -------
#   pip install holidays osmnx lightgbm scikit-learn geopandas pyarrow
#%% md
# **Libraries**
#%%
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import holidays
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)



warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
#%% md
# **Section 1: Paths & configuration**
#%%

import sys
from pathlib import Path

# Add the repo root to the Python path so 'src' becomes importable
_CURRENT_ROOT = Path(__file__).resolve().parents[2]
if str(_CURRENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CURRENT_ROOT))

from src.io_paths import (
    REPO_ROOT,
    DATA_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    DOCS_DIR,
    DIM_PATH,
    GOLD_PATH,
    FEATURES_PATH,
    SURGE_PRED_PATH,
    NBHD_PRED_PATH,
    SURGE_METRICS_PATH,
    NBHD_METRICS_PATH,
    COLLISION_FI_PATH,
    KSI_FI_PATH,
)

import mlflow
import mlflow.sklearn
from prefect import flow, task
from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    SPLIT_DATE as _RAW_SPLIT_DATE,
    SURGE_PCT,
    FORECAST_HORIZONS as HORIZONS
)

# Convert string date from YAML to Pandas Timestamp
SPLIT_DATE = pd.Timestamp(_RAW_SPLIT_DATE)

# Setup MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Aliases to keep the rest of your pipeline script from breaking
PROCESSED = PROCESSED_DIR
DOCS = DOCS_DIR
INTERIM = INTERIM_DIR

# ── intermediate files not explicitly in io_paths yet ────────────────────────
ADJ_PATH  = INTERIM / "nbhd_adjacency.parquet"
OSM_CACHE = INTERIM / "road_network_static.parquet"


# ── feature engineering configuration ────────────────────────────────────────
ROAD_COLS = [
    "road_events_count",
    "road_events_severity_weighted",
    "road_events_full_closure_count",
    "road_construction_count",
    "road_construction_severity_weighted",
    "road_construction_full_closure_count",
]

LAG_COLS = [
    "collisions",
    "ksi_collisions",
    "ksi_weighted_score",
    "road_events_count",
    "road_events_severity_weighted",
    "road_construction_count",
    "road_construction_severity_weighted",
]

LAGS    = [1, 7, 14]
WINDOWS = [7, 14, 30]

# ── leakage guard — columns excluded from all model feature sets ──────────────
# Same-day actuals and identifiers that must never be features
_BASE_LEAKAGE: set[str] = {
    "date", "nbhd_id", "area_id", "AREA_ID", "area_name", "AREA_NAME",
    "geometry",
    "collisions", "city_collisions",
    "ksi_collisions", "ksi_weighted_score",
    "ksi_fatal_collisions", "ksi_serious_collisions",
    "ksi_fatal_victims", "ksi_victim_count",
    # same-day KSI ratio (numerator uses ksi_collisions) — lagged version is safe
    "ksi_ratio",
}

# Additional columns excluded from the surge classifier only
SURGE_LEAKAGE_COLS: set[str] = _BASE_LEAKAGE | set()

# Additional columns excluded from neighbourhood models only
NBHD_LEAKAGE_COLS: set[str] = _BASE_LEAKAGE | {
    f"collisions_t{h}" for h in HORIZONS
}

# ── Schema Validation & Helpers ───────────────────────────────────────────────
from src.schemas import (
    gold_schema,
    FeatureSchema,
    surge_predictions_schema,
    nbhd_predictions_schema,
)

def validate_unique_key(df, cols, name: str):
    dupes = df.duplicated(subset=cols).sum()
    if dupes > 0:
        raise ValueError(f"{name} has {dupes} duplicate rows on key {cols}")

def validate_expected_neighbourhoods(df, name: str, expected: int = 158):
    n = df["nbhd_id"].nunique()
    if n != expected:
        raise ValueError(f"{name} has {n} unique neighbourhoods; expected {expected}")

#%% md
# **Section 2: Feature builder**
#%%
# ── 2a. OSM road-network static features ─────────────────────────────────────

def build_osm_features(dim: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Fetch 4 static road-network exposure features per neighbourhood from OSM.

    intersection_count      number of road intersections
    street_segment_count    number of unique street segments
    arterial_length_km      total driveable street length in km
    streets_per_node_avg    average road connectivity (higher = denser grid)

    Why these matter
    ----------------
    These features are static — they don't change day to day — and explain why
    certain neighbourhoods have structurally higher collision counts regardless
    of weather or events. A neighbourhood with 400 intersections is exposed to
    ~10× more potential conflict points than one with 40. The KSI severity model
    benefits most: dense arterial networks concentrate serious collisions at
    high-speed intersections.

    Caching
    -------
    Result is saved to OSM_CACHE after the first run (~5 min for 158 nbhds).
    Subsequent runs load from cache instantly. Delete the cache file to refresh.

    Requires: pip install osmnx
    """
    if OSM_CACHE.exists():
        print(f"  OSM cache found — loading {OSM_CACHE.name}")
        return pd.read_parquet(OSM_CACHE)

    try:
        import osmnx as ox
        ox.settings.log_console = False
    except ImportError:
        print("  osmnx not installed — skipping road-network features.\n"
              "  Run: pip install osmnx")
        return pd.DataFrame(
            columns=["nbhd_id", "intersection_count", "street_segment_count",
                     "arterial_length_km", "streets_per_node_avg"]
        )

    print(f"  Fetching OSM road network for {len(dim)} neighbourhoods "
          "(one-time, result is cached)...")
    dim_proj = dim.to_crs(epsg=3978)
    records: list[dict] = []
    failed = 0

    for idx, (_, row) in enumerate(dim_proj.iterrows(), start=1):
        nbhd_id = int(row["nbhd_id"])
        try:
            poly_wgs84 = (
                gpd.GeoSeries([row.geometry], crs=3978)
                .to_crs(4326)
                .iloc[0]
            )
            G     = ox.graph_from_polygon(
                poly_wgs84, network_type="drive", retain_all=False
            )
            stats = ox.basic_stats(G)
            records.append({
                "nbhd_id":              nbhd_id,
                "intersection_count":   int(stats.get("intersection_count", 0)),
                "street_segment_count": int(stats.get("street_segment_count", 0)),
                "arterial_length_km":   round(
                    stats.get("street_length_total", 0.0) / 1000, 4
                ),
                "streets_per_node_avg": round(
                    float(stats.get("streets_per_node_avg", 0.0)), 4
                ),
            })
        except Exception as exc:
            failed += 1
            print(f"    nbhd {nbhd_id}: fetch failed ({exc}) — zero-filled")
            records.append({
                "nbhd_id":              nbhd_id,
                "intersection_count":   0,
                "street_segment_count": 0,
                "arterial_length_km":   0.0,
                "streets_per_node_avg": 0.0,
            })
        if idx % 20 == 0:
            print(f"    {idx}/{len(dim)} done...")

    osm_df = pd.DataFrame(records)
    osm_df.to_parquet(OSM_CACHE, index=False)
    print(f"  Cached → {OSM_CACHE}  ({len(osm_df)} nbhds, {failed} zero-filled)")
    return osm_df


# ── 2b. Ontario public holiday features ──────────────────────────────────────

def _build_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ontario statutory holidays and cyclical DOW/month encodings.

    is_holiday     statutory Ontario holiday (9 per year)
                   Includes Family Day, Good Friday, Victoria Day, Canada Day,
                   Labour Day, Thanksgiving, Christmas, Boxing Day, New Year's
    pre_holiday    day before a holiday — elevated early-departure traffic
    post_holiday   day after a holiday — elevated return-trip traffic
    is_weekend     Saturday or Sunday
    dow_sin/cos    cyclical day-of-week encoding
                   Preserves Mon↔Sun continuity that raw integer breaks
    month_sin/cos  cyclical month encoding
                   Preserves Dec↔Jan continuity

    Requires: pip install holidays
    """
    years   = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1)
    on_hols = holidays.Canada(prov="ON", years=list(years))
    hol_set = set(on_hols.keys())

    dow   = df["date"].dt.dayofweek
    month = df["date"].dt.month

    new: dict[str, pd.Series] = {
        "dow":          dow,
        "month":        month,
        "year":         df["date"].dt.year,
        "is_weekend":   (dow >= 5).astype(int),
        "is_holiday":   df["date"].dt.date.isin(hol_set).astype(int),
        "pre_holiday":  df["date"].apply(
            lambda d: int((d + pd.Timedelta(days=1)).date() in hol_set)
        ),
        "post_holiday": df["date"].apply(
            lambda d: int((d - pd.Timedelta(days=1)).date() in hol_set)
        ),
        "dow_sin":   np.sin(2 * np.pi * dow   / 7),
        "dow_cos":   np.cos(2 * np.pi * dow   / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# ── 2c. Weather interaction features ─────────────────────────────────────────

def _build_weather_interaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compound risk flags derived entirely from existing weather columns.
    No new data required — these are products/combinations of what the
    Open-Meteo fetch already provides.

    freezing_rain  prcp > 0 AND tmin ≤ 0°C
                   Worst road-surface condition: liquid water freezes on contact
    wind_rain      prcp > 2mm AND wspd > 30 km/h
                   Low visibility + hydroplaning risk
    heavy_snow     snow > 5cm/day
                   Significant plowing lag; road surface uncleared for hours
    cold_dry       tmin ≤ -10°C AND prcp = 0
                   Black ice on un-salted, un-treated surfaces
    feels_bad      OR of freezing_rain, wind_rain, heavy_snow
                   Summary "adverse condition day" flag for the model
    """
    fr = ((df["prcp"] > 0) & (df["tmin"] <= 0)).astype(int)
    wr = ((df["prcp"] > 2) & (df["wspd"] > 30)).astype(int)
    hs = (df["snow"] > 5).astype(int)
    cd = ((df["tmin"] <= -10) & (df["prcp"] == 0)).astype(int)

    new: dict[str, pd.Series] = {
        "freezing_rain": fr,
        "wind_rain":     wr,
        "heavy_snow":    hs,
        "cold_dry":      cd,
        "feels_bad":     (fr | wr | hs).astype(int),
    }
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# ── 2d. KSI severity ratio ────────────────────────────────────────────────────

def _build_ksi_ratio_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ksi_ratio      ksi_collisions / total collisions on that day.
                   Captures structural neighbourhood severity independent of
                   volume: a zone with ratio 0.8 is inherently more dangerous
                   than one with ratio 0.1, regardless of collision count.
    ksi_ratio_lag1 yesterday's ratio — lagged to prevent same-day leakage.
                   The same-day ratio uses ksi_collisions (a leakage column)
                   in its numerator, so only the lagged version is safe.
    """
    ratio = (
        df["ksi_collisions"] / df["collisions"].replace(0, np.nan)
    ).fillna(0.0)
    ratio_lag = df.groupby("nbhd_id")["ksi_ratio"].shift(1) \
        if "ksi_ratio" in df.columns \
        else ratio.groupby(df["nbhd_id"]).shift(1)

    return pd.concat(
        [df, pd.DataFrame(
            {"ksi_ratio": ratio, "ksi_ratio_lag1": ratio_lag},
            index=df.index,
        )],
        axis=1,
    )


# ── 2e. Lags & rolling windows ────────────────────────────────────────────────

def _build_lag_roll_columns(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int] = LAGS,
    windows: list[int] = WINDOWS,
) -> pd.DataFrame:
    """
    Lags (shift 1, 7, 14 days) and rolling sum/mean (7, 14, 30 days) computed
    per neighbourhood group.

    Leakage prevention
    ------------------
    All rolling windows apply .shift(1) before .rolling(W), so the current
    day's value is never included in any rolling feature. This is the single
    most important leakage guard in the feature pipeline.

    Performance
    -----------
    All new columns are built in one pd.concat call rather than assigned one
    at a time, avoiding Pandas DataFrame fragmentation warnings that the
    original pipeline generated.
    """
    new: dict[str, pd.Series] = {}
    grp = df.groupby("nbhd_id", group_keys=False)

    for col in cols:
        if col not in df.columns:
            continue
        for L in lags:
            new[f"{col}_lag{L}"] = grp[col].shift(L)
        shifted = grp[col].shift(1)
        for W in windows:
            new[f"{col}_roll{W}_sum"]  = (
                shifted.rolling(W).sum().reset_index(level=0, drop=True)
            )
            new[f"{col}_roll{W}_mean"] = (
                shifted.rolling(W).mean().reset_index(level=0, drop=True)
            )

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# ── 2f. Collision momentum ────────────────────────────────────────────────────

def _build_momentum_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    collision_momentum = yesterday's collisions / 14-day rolling baseline.

    This ratio tells the model whether yesterday was abnormal relative to
    the neighbourhood's recent history — more informative than the raw lag
    count alone, because it normalises for neighbourhood size and seasonality.
    Values > 1.5 indicate a spike; < 0.5 indicate an unusual lull.
    Fills to 1.0 (neutral) where the baseline is zero (start of series).
    """
    grp      = df.groupby("nbhd_id", group_keys=False)
    baseline = (
        grp["collisions"].shift(1).rolling(14).mean()
        .reset_index(level=0, drop=True)
    )
    momentum = (
        grp["collisions"].shift(1) / baseline.replace(0, np.nan)
    ).fillna(1.0)
    return pd.concat(
        [df, pd.DataFrame({"collision_momentum": momentum}, index=df.index)],
        axis=1,
    )


# ── 2g. Halo (neighbour-zone spillover) features ─────────────────────────────

def _build_halo_columns(df: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """
    For each neighbourhood, aggregate road events and construction counts from
    all directly adjacent neighbours (the "halo zone").

    Why this matters
    ----------------
    A road closure or major construction project in an adjacent zone diverts
    traffic through the current zone — increasing collision risk even when the
    current zone shows no direct disruption signal. The halo captures this
    spatial spillover that a naive neighbourhood-level feature misses.

    Derived features per halo column
    ---------------------------------
    {col}_halo        raw sum from adjacent neighbourhoods
    {col}_halo_norm   log(1 + raw) — squashes outliers for tree stability

    Binary disruption indicators
    ----------------------------
    is_zone_disrupted          any adjacent nbhd has active construction
    is_event_disrupted_halo    any adjacent nbhd has an active road event
    construction_total_pressure  direct + halo construction count combined
    construction_pressure_norm   log(1 + total_pressure)

    Halo columns also receive their own lag (1, 7 day) and rolling (7, 14 day)
    features so the model can learn delayed spillover effects.
    """
    road_daily = df[["date", "nbhd_id"] + ROAD_COLS].copy()

    halo_agg = (
        adj
        .merge(
            road_daily,
            left_on="adjacent_nbhd_id",
            right_on="nbhd_id",
            how="left",
            suffixes=("", "_nbr"),
        )
        .groupby(["date", "nbhd_id"], as_index=False)[ROAD_COLS]
        .sum()
        .rename(columns={c: f"{c}_halo" for c in ROAD_COLS})
    )

    before = len(df)
    df = df.merge(halo_agg, on=["date", "nbhd_id"], how="left")
    assert len(df) == before, "Halo merge changed row count"

    halo_cols = [f"{c}_halo" for c in ROAD_COLS]
    new: dict[str, pd.Series] = {}

    for c in halo_cols:
        df[c] = df[c].fillna(0)
        new[f"{c}_norm"] = np.log1p(df[c])

    total_pressure = (
        df["road_construction_count"] + df["road_construction_count_halo"]
    )
    new["is_zone_disrupted"]           = (df["road_construction_count_halo"] > 0).astype(int)
    new["is_event_disrupted_halo"]     = (df["road_events_count_halo"] > 0).astype(int)
    new["construction_total_pressure"] = total_pressure
    new["construction_pressure_norm"]  = np.log1p(total_pressure)

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    df = _build_lag_roll_columns(df, halo_cols, lags=[1, 7], windows=[7, 14])
    return df


# ── 2h. Multi-horizon targets ─────────────────────────────────────────────────

def _build_target_columns(
    df: pd.DataFrame,
    horizons: list[int] = HORIZONS,
) -> pd.DataFrame:
    """
    Build three target types per forecast horizon h:

    collisions_t{h}           raw future collision count
                              used for regression and ranking evaluation
    target_collision_t{h}     binary: will ≥1 collision occur h days from now?
                              used for the collision probability classifier
    target_ksi_t{h}           binary: will ≥1 KSI collision occur h days from now?
                              used for the severity classifier

    Leakage prevention
    ------------------
    Each shift is applied per-neighbourhood group (groupby("nbhd_id")), so the
    final day of neighbourhood A never bleeds into the first day of neighbourhood
    B. Rows where the furthest-horizon target is unavailable (end of time series)
    are dropped — they cannot be used as labelled training examples.
    """
    new: dict[str, pd.Series] = {}
    for h in horizons:
        fc = df.groupby("nbhd_id")["collisions"].shift(-h)
        fk = df.groupby("nbhd_id")["ksi_collisions"].shift(-h)
        new[f"collisions_t{h}"]       = fc
        new[f"target_collision_t{h}"] = (fc.fillna(0) >= 1).astype(int)
        new[f"target_ksi_t{h}"]       = (fk.fillna(0) >= 1).astype(int)

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df.dropna(subset=[f"collisions_t{max(horizons)}"]).copy()


# ── 2 main: feature builder entry point ──────────────────────────────────────

@task(name="Build Features")
def run_feature_builder() -> pd.DataFrame:
    """
    Orchestrates all feature engineering steps and saves features_v2.parquet.
    Called by run_pipeline() or directly via --only features.
    """
    print("=" * 60)
    print("  SECTION 2 — Feature Builder")
    print("=" * 60)

    # Load inputs
    print("\n[1/8] Loading inputs...")
    df  = pd.read_parquet(GOLD_PATH)
    adj = pd.read_parquet(ADJ_PATH)
    dim = gpd.read_parquet(DIM_PATH)

    df["date"]              = pd.to_datetime(df["date"])
    df["nbhd_id"]           = df["nbhd_id"].astype(int)
    adj["nbhd_id"]          = adj["nbhd_id"].astype(int)
    adj["adjacent_nbhd_id"] = adj["adjacent_nbhd_id"].astype(int)
    dim["nbhd_id"]          = dim["nbhd_id"].astype(int)

    #INPUT VALIDATION
    df = gold_schema.validate(df)
    validate_unique_key(df, ["date", "nbhd_id"], "gold input table")
    validate_expected_neighbourhoods(df, "gold input table")

    # Normalise column name (older pipeline versions used collision_count)
    if "collision_count" in df.columns and "collisions" not in df.columns:
        df = df.rename(columns={"collision_count": "collisions"})

    # Drop duplicate columns left from prior pipeline joins
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_values(["nbhd_id", "date"]).reset_index(drop=True)

    print(f"  Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Neighbourhoods: {df['nbhd_id'].nunique()}")

    # OSM static features
    print("\n[2/8] OSM road-network features (one-time, cached)...")
    osm = build_osm_features(dim)
    if not osm.empty:
        osm["nbhd_id"] = osm["nbhd_id"].astype(int)
        before = len(df)
        df = df.merge(osm, on="nbhd_id", how="left")
        assert len(df) == before, "OSM merge changed row count"
        # Median-impute any neighbourhoods where OSM fetch failed
        for c in ["intersection_count", "street_segment_count",
                  "arterial_length_km", "streets_per_node_avg"]:
            df[c] = df[c].fillna(df[c].median())
        print(f"  Added: intersection_count, street_segment_count, "
              f"arterial_length_km, streets_per_node_avg")

    # Calendar + holiday features
    print("\n[3/8] Calendar + Ontario holiday features...")
    df = _build_calendar_columns(df)
    hol_days = int(df.drop_duplicates("date")["is_holiday"].sum())
    print(f"  Holiday days flagged: {hol_days} "
          f"(~{hol_days / df['date'].nunique() * 100:.1f}% of dates)")

    # Weather interaction features
    print("\n[4/8] Weather interaction features...")
    df = _build_weather_interaction_columns(df)
    print(f"  Added: freezing_rain, wind_rain, heavy_snow, cold_dry, feels_bad")

    # KSI severity ratio
    print("\n[5/8] KSI severity ratio...")
    df = _build_ksi_ratio_columns(df)
    print(f"  Added: ksi_ratio, ksi_ratio_lag1")

    # Lags + rolling windows
    print("\n[6/8] Lags and rolling windows...")
    df = _build_lag_roll_columns(df, LAG_COLS)
    n_lag = sum(1 for c in df.columns if "_lag" in c or "_roll" in c)
    print(f"  {n_lag} lag/rolling columns created")

    # Collision momentum
    print("\n[6b/8] Collision momentum...")
    df = _build_momentum_column(df)
    print(f"  Added: collision_momentum")

    # Halo features
    print("\n[7/8] Halo (neighbour-zone spillover) features...")
    df = _build_halo_columns(df, adj)
    n_halo = sum(1 for c in df.columns if "_halo" in c)
    print(f"  {n_halo} halo-derived columns created")

    # Multi-horizon targets
    print("\n[8/8] Multi-horizon targets...")
    df = _build_target_columns(df, HORIZONS)
    for h in HORIZONS:
        cr = df[f"target_collision_t{h}"].mean()
        kr = df[f"target_ksi_t{h}"].mean()
        print(f"  T+{h}  collision positive rate={cr:.3f}  "
              f"KSI positive rate={kr:.3f}")

    # Drop rows missing the minimum required lag features
    df = df.dropna(subset=["collisions_lag1", "collisions_roll7_mean"]).copy()

    #OUTPUT VALIDATION
    df = FeatureSchema.validate(df)
    validate_unique_key(df, ["date", "nbhd_id"], "features output")
    validate_expected_neighbourhoods(df, "features output")

    df.to_parquet(FEATURES_PATH, index=False)
    print(f"\n  Output: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Saved  → {FEATURES_PATH}\n")
    return df
#%% md
# **Section 3: Citywide surge classifier**
#%%
def _surge_feature_cols(
    df: pd.DataFrame,
    extra_drop: set[str] | None = None,
) -> list[str]:
    """Return numeric columns safe to use as surge model features."""
    drop = SURGE_LEAKAGE_COLS.copy()
    if extra_drop:
        drop |= extra_drop
    return [c for c in df.columns if c not in drop and df[c].dtype.kind in "biuf"]


def _surge_metrics(y_true: pd.Series, y_prob: np.ndarray, label: str) -> dict:
    return {
        "model":    label,
        "roc_auc":  round(roc_auc_score(y_true, y_prob), 6),
        "pr_auc":   round(average_precision_score(y_true, y_prob), 6),
        "brier":    round(brier_score_loss(y_true, y_prob), 6),
        "n_test":   int(len(y_true)),
        "pos_rate": round(float(y_true.mean()), 4),
    }


def _seasonally_adjusted_threshold(
    train_city: pd.DataFrame,
    pct: float = SURGE_PCT,
) -> float:
    """
    Compute a season-aware surge threshold from the training split only.

    Raw quantile thresholds conflate seasonal variation with genuine surges:
    a July day with 300 collisions is normal, but a February day with 300 is
    extreme. This function z-scores each daily total against a 60-day trailing
    mean, takes the pct-quantile of those z-scores, then converts back to a
    raw collision count using the training period's global mean and std.

    Falls back to a raw quantile if the training window is too short (<30 days
    of valid z-scores).
    """
    tc = train_city.copy().sort_values("date")
    tc["roll60"] = (
        tc["city_collisions"].shift(1).rolling(60, min_periods=14).mean()
    )
    tc["roll60_std"] = (
        tc["city_collisions"].shift(1).rolling(60, min_periods=14).std()
    )
    tc["zscore"] = (
        (tc["city_collisions"] - tc["roll60"]) /
        tc["roll60_std"].replace(0, np.nan)
    )
    valid = tc["zscore"].dropna()
    if len(valid) < 30:
        return float(np.quantile(tc["city_collisions"].dropna(), pct))

    z_thr       = float(np.quantile(valid, pct))
    global_mean = tc["roll60"].median()
    global_std  = tc["roll60_std"].median()
    return float(global_mean + z_thr * global_std)


def _train_surge_horizon(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Train a calibrated LightGBM surge classifier for one horizon.

    The surge label is defined on the FUTURE city total (horizon days from now),
    computed entirely within the training split to prevent threshold leakage
    into the test period.

    Isotonic calibration (3-fold CV) is applied to improve Brier score — raw
    LightGBM probabilities are typically overconfident on imbalanced targets.

    Returns (predictions_df, metrics_dict).
    """
    print(f"\n  ── Surge T+{horizon} ──")

    # Aggregate neighbourhood rows to one row per date
    count_kws = ("count", "collisions", "weighted", "score",
                 "victims", "halo", "pressure", "lag", "roll", "momentum")
    feat_cands = _surge_feature_cols(df)
    agg: dict[str, str] = {
        c: ("sum" if any(k in c for k in count_kws) else "mean")
        for c in feat_cands
    }
    daily = (
        df.groupby("date")
          .agg({**agg, "collisions": "sum"})
          .sort_values("date")
          .reset_index()
    )

    # Future city total (the thing we're trying to predict)
    daily["city_collisions_future"] = daily["collisions"].shift(-horizon)
    daily = daily.dropna(subset=["city_collisions_future"]).copy()

    # Threshold from training data only
    train_daily = daily[daily["date"] < SPLIT_DATE]
    threshold   = _seasonally_adjusted_threshold(
        train_daily.rename(columns={"city_collisions_future": "city_collisions"}),
        pct=SURGE_PCT,
    )
    print(f"    Surge threshold (season-adjusted): {threshold:.1f} collisions/day")

    daily["y"] = (daily["city_collisions_future"] >= threshold).astype(int)
    train = daily[daily["date"] < SPLIT_DATE].copy()
    test  = daily[daily["date"] >= SPLIT_DATE].copy()

    extra = {
        "city_collisions_future", "y",
        *[c for c in daily.columns if c.startswith("target_")],
        *[c for c in daily.columns if c.startswith("collisions_t")],
    }
    xcols = _surge_feature_cols(daily, extra_drop=extra)

    X_train, y_train = train[xcols], train["y"]
    X_test,  y_test  = test[xcols],  test["y"]

    n_est = 200 if len(train) < 500 else 400
    base  = LGBMClassifier(
        objective="binary",
        n_estimators=n_est,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
        random_state=42,
        verbose=-1,
    )
    with mlflow.start_run(run_name=f"surge_t{horizon}", nested=True):
        model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        m = _surge_metrics(y_test, proba, label=f"surge_t{horizon}")

        # Log to MLflow
        mlflow.log_params({"horizon": horizon, "n_estimators": n_est, "threshold": threshold})
        mlflow.log_metrics({"roc_auc": m['roc_auc'], "pr_auc": m['pr_auc'], "brier": m['brier']})
        mlflow.sklearn.log_model(model, f"model_surge_t{horizon}")
    print(f"    ROC-AUC={m['roc_auc']}  PR-AUC={m['pr_auc']}  Brier={m['brier']}")

    pred_df = test[["date"]].copy()
    pred_df[f"surge_proba_t{horizon}"] = proba
    pred_df[f"surge_label_t{horizon}"] = test["y"].values
    pred_df["threshold"] = threshold
    return pred_df, m


@task(name="Train Surge Classifier")
def run_surge_classifier() -> pd.DataFrame:
    """
    Train surge classifiers for all horizons and save surge_predictions.parquet.
    Called by run_pipeline() or directly via --only surge.
    """
    print("=" * 60)
    print("  SECTION 3 — Citywide Surge Classifier")
    print("=" * 60)

    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    all_preds:   list[pd.DataFrame] = []
    all_metrics: list[dict]         = []

    for h in HORIZONS:
        pred_df, m = _train_surge_horizon(df, horizon=h)
        all_preds.append(pred_df)
        all_metrics.append(m)

    # Merge all horizons on date
    surge_pred = all_preds[0]
    for p in all_preds[1:]:
        surge_pred = surge_pred.merge(
            p.drop(columns=["threshold"], errors="ignore"),
            on="date",
            how="outer",
        )
    surge_pred = surge_pred.sort_values("date").reset_index(drop=True)

    #OUTPUT VALIDATION
    pred_df = surge_predictions_schema.validate(pred_df)
    validate_unique_key(pred_df, ["date"], "surge predictions")

    surge_pred.to_parquet(SURGE_PRED_PATH, index=False)
    with open(SURGE_METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n  Saved surge predictions → {SURGE_PRED_PATH}  "
          f"shape={surge_pred.shape}")
    print(f"  Saved surge metrics     → {SURGE_METRICS_PATH}\n")
    return surge_pred

#%% md
# **Section 4: Neighbourhood models**
#%%
def _nbhd_feature_cols(
    df: pd.DataFrame,
    extra_drop: set[str],
) -> list[str]:
    """Return numeric columns safe to use as neighbourhood model features."""
    drop = NBHD_LEAKAGE_COLS | extra_drop
    return [c for c in df.columns if c not in drop and df[c].dtype.kind in "biuf"]


def _classification_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    label: str,
) -> dict:
    return {
        "model":    label,
        "roc_auc":  round(roc_auc_score(y_true, y_prob), 6),
        "pr_auc":   round(average_precision_score(y_true, y_prob), 6),
        "brier":    round(brier_score_loss(y_true, y_prob), 6),
        "n_test":   int(len(y_true)),
        "pos_rate": round(float(y_true.mean()), 4),
    }


def _precision_recall_at_k(
    test_df: pd.DataFrame,
    score_col: str,
    target_col: str,
    k: int,
) -> dict:
    """
    Day-by-day ranking evaluation.

    Precision@K  fraction of the top-K predicted neighbourhoods that actually
                 had a collision (or KSI event) on that day
    Recall@K     fraction of actual positive neighbourhoods captured in top-K
                 (NaN days where total positives = 0 are skipped for recall)
    """
    daily: list[dict] = []
    for _, g in test_df.groupby("date"):
        top_k     = g.nlargest(k, score_col)
        hits      = top_k[target_col].sum()
        total_pos = g[target_col].sum()
        daily.append({
            "precision": hits / k,
            "recall": hits / total_pos if total_pos > 0 else 0.0,
        })
    ddf = pd.DataFrame(daily)
    return {
        f"precision_at_{k}": round(float(ddf["precision"].mean()), 6),
        f"recall_at_{k}":    round(float(ddf["recall"].mean(skipna=True)), 6),
    }


def _lift_at_k(
    test_df: pd.DataFrame,
    score_col: str,
    target_col: str,
    ks: tuple[int, ...] = (5, 10, 15, 20),
) -> dict:
    """
    Precision@K and Recall@K for multiple K values, plus lift over random
    (precision@K / baseline positive rate). Lift > 1 means the model is
    doing better than random selection.
    """
    baseline = float(test_df[target_col].mean())
    out: dict = {"baseline_pos_rate": round(baseline, 4)}
    for k in ks:
        r = _precision_recall_at_k(test_df, score_col, target_col, k)
        p = r[f"precision_at_{k}"]
        out.update(r)
        out[f"lift_at_{k}"] = round(p / baseline, 3) if baseline > 0 else None
    return out


def _build_lgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_est: int = 400,
) -> CalibratedClassifierCV:
    """
    Calibrated LightGBM binary classifier.

    scale_pos_weight balances the class imbalance without oversampling.
    Isotonic calibration on 3-fold CV corrects the probability miscalibration
    that raw LightGBM produces on imbalanced targets, improving Brier score.
    """
    pos  = y_train.sum()
    neg  = len(y_train) - pos
    base = LGBMClassifier(
        objective="binary",
        n_estimators=n_est,
        learning_rate=0.04,
        num_leaves=47,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        scale_pos_weight=neg / max(pos, 1),
        random_state=42,
        verbose=-1,
    )
    return CalibratedClassifierCV(base, cv=3, method="isotonic")


def _extract_feature_importance(
    model: CalibratedClassifierCV,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Average feature importances across the 3 calibration folds.
    Returns a DataFrame sorted descending by importance.
    """
    importances = np.zeros(len(feature_cols))
    for estimator in model.calibrated_classifiers_:
        importances += estimator.estimator.feature_importances_
    importances /= len(model.calibrated_classifiers_)
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _merge_surge_proba_safe(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Merge citywide surge probability into the neighbourhood feature table.

    Leakage prevention
    ------------------
    The surge model was trained on all data before split_date. If we used its
    predictions on training-period rows as a feature, we would be feeding a
    model output that has already "seen" those labels. Only test-period surge
    probabilities are genuine out-of-sample predictions.

    Fix: training-period rows receive a neutral fill of 0.5 — the midpoint of
    the probability scale, conveying no information either way.
    """
    if not SURGE_PRED_PATH.exists():
        print("  No surge predictions found — skipping surge_proba feature")
        return df

    surge = pd.read_parquet(SURGE_PRED_PATH)
    surge["date"] = pd.to_datetime(surge["date"])
    surge_cols    = [c for c in surge.columns if c.startswith("surge_proba")]
    surge         = surge[["date"] + surge_cols].drop_duplicates("date")

    before = len(df)
    df     = df.merge(surge, on="date", how="left")
    assert len(df) == before, "Surge merge changed row count"

    for c in surge_cols:
        mask       = (df["date"] < split_date) | df[c].isna()
        df.loc[mask, c] = 0.5   # neutral fill for training rows

    print(f"  Surge features merged: {surge_cols}  "
          f"(training rows filled with 0.5)")
    return df


def _check_drift(
    pred_df: pd.DataFrame,
    score_col: str,
    target_col: str,
    horizon: int,
) -> None:
    """
    Split the test period in half chronologically and compare Precision@10.
    A drop > 15% in the second half signals model drift and triggers a warning.
    """
    dates = sorted(pred_df["date"].unique())
    mid   = dates[len(dates) // 2]
    early = pred_df[pred_df["date"] <  mid]
    late  = pred_df[pred_df["date"] >= mid]

    p_e   = _precision_recall_at_k(early, score_col, target_col, k=10)
    p_l   = _precision_recall_at_k(late,  score_col, target_col, k=10)
    e10   = p_e["precision_at_10"]
    l10   = p_l["precision_at_10"]
    drop  = (e10 - l10) / e10 if e10 > 0 else 0
    flag  = "STABLE" if drop <= 0.15 else "⚠ WARNING — retrain recommended"

    print(f"  Drift T+{horizon}: early={e10:.3f}  late={l10:.3f}  "
          f"drop={drop:.1%}  [{flag}]")


def _train_nbhd_horizon(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Train collision and KSI models for one horizon and evaluate ranking quality.

    Model A  collision probability  P(≥1 collision h days from now)
    Model B  KSI severity           P(≥1 KSI collision h days from now)
    Score C  weighted combination   0.6 × P(collision) + 0.4 × P(KSI)
             used as the neighbourhood risk ranking score

    Returns (predictions_df, [collision_metrics, ksi_metrics]).
    """
    print(f"\n  ── Neighbourhood models T+{horizon} ──")
    col_tgt = f"target_collision_t{horizon}"
    ksi_tgt = f"target_ksi_t{horizon}"

    # All target columns must be excluded from features
    all_tgt = {
        *[f"target_collision_t{h}" for h in HORIZONS],
        *[f"target_ksi_t{h}"       for h in HORIZONS],
        *[f"collisions_t{h}"       for h in HORIZONS],
    }
    xcols = _nbhd_feature_cols(df, extra_drop=all_tgt)

    train = df[df["date"] <  SPLIT_DATE].copy()
    test  = df[df["date"] >= SPLIT_DATE].copy()

    X_train = train[xcols]
    X_test  = test[xcols]
    n_est   = 200 if len(train) < 10_000 else 400

    # ── Model A: Collision Forecast ──────────────────────────────────────────
    with mlflow.start_run(run_name=f"nbhd_collision_t{horizon}", nested=True):
        m_col_obj = _build_lgbm_model(X_train, train[col_tgt], n_est=n_est)
        m_col_obj.fit(X_train, train[col_tgt])
        p_col = m_col_obj.predict_proba(X_test)[:, 1]

        m_col = _classification_metrics(test[col_tgt], p_col, label=f"collision_prob_t{horizon}")
        m_col.update(_lift_at_k(test.assign(score=p_col), "score", col_tgt, ks=(5, 10)))

        mlflow.log_params({"horizon": horizon, "target": "collision"})

        # SAFE LOGGING
        safe_metrics = {k: v for k, v in m_col.items() if isinstance(v, (int, float)) and not pd.isna(v)}
        mlflow.log_metrics(safe_metrics)

        # FIXED LOGGING: Using 'artifact_path' and 'skops'/'cloudpickle' serialization to avoid warnings
        mlflow.sklearn.log_model(
            m_col_obj,
            artifact_path=f"model_nbhd_col_t{horizon}",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        # ── Model B: Serious Incident (KSI) Forecast ─────────────────────────────
        # CONDITIONAL CHECK: Ensure we actually have positive cases to train/test against
        total_ksi_train = train[ksi_tgt].sum()
        total_ksi_test = test[ksi_tgt].sum()

        if total_ksi_train == 0 or total_ksi_test == 0:
            print(f"  Skipping KSI T+{horizon} evaluation — no positive samples in train/test set.")
            # Return structurally consistent metrics for this horizon
            m_ksi = {
                "model": f"ksi_prob_t{horizon}",
                "roc_auc": float("nan"),
                "pr_auc": 0.0,
                "brier": 0.0,
                "precision_at_10": None,  # Defensive explicit key for summary logic
            }
            p_ksi = np.zeros(len(X_test))
        else:
            with mlflow.start_run(run_name=f"nbhd_ksi_t{horizon}", nested=True):
                m_ksi_obj = _build_lgbm_model(X_train, train[ksi_tgt], n_est=n_est)
                m_ksi_obj.fit(X_train, train[ksi_tgt])
                p_ksi = m_ksi_obj.predict_proba(X_test)[:, 1]

                m_ksi = _classification_metrics(test[ksi_tgt], p_ksi, label=f"ksi_prob_t{horizon}")

                mlflow.log_params({"horizon": horizon, "target": "ksi"})

                # SAFE LOGGING
                safe_metrics = {k: v for k, v in m_ksi.items() if isinstance(v, (int, float)) and not pd.isna(v)}
                mlflow.log_metrics(safe_metrics)

                # FIXED LOGGING
                mlflow.sklearn.log_model(
                    m_ksi_obj,
                    artifact_path=f"model_nbhd_ksi_t{horizon}",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                )

    # ── Score C: combined risk ranking score ──────────────────────────────────
    combined = 0.6 * p_col + 0.4 * p_ksi

    pred_df = test[["date", "nbhd_id", col_tgt, ksi_tgt]].copy()
    pred_df[f"proba_collision_t{horizon}"] = p_col
    pred_df[f"proba_ksi_t{horizon}"]       = p_ksi
    pred_df[f"risk_score_t{horizon}"]      = combined

    return pred_df, [m_col, m_ksi]


@task(name="Train Neighbourhood Models")
def run_nbhd_models() -> pd.DataFrame:
    """
    Train neighbourhood models for all horizons and save nbhd_predictions.parquet.
    Called by run_pipeline() or directly via --only nbhd.
    """
    print("=" * 60)
    print("  SECTION 4 — Neighbourhood Models")
    print("=" * 60)

    df = pd.read_parquet(FEATURES_PATH)
    df["date"]    = pd.to_datetime(df["date"])
    df["nbhd_id"] = df["nbhd_id"].astype(int)

    # Leakage-safe surge feature merge
    df = _merge_surge_proba_safe(df, SPLIT_DATE)

    all_preds:   list[pd.DataFrame] = []
    all_metrics: list[dict]         = []

    for h in HORIZONS:
        pred_df, metrics = _train_nbhd_horizon(df, h)
        all_preds.append(pred_df)
        all_metrics.extend(metrics)

        _check_drift(
            pred_df,
            score_col=f"risk_score_t{h}",
            target_col=f"target_collision_t{h}",
            horizon=h,
        )

    # Merge all horizon predictions
    result = all_preds[0]
    for p in all_preds[1:]:
        result = result.merge(
            p, on=["date", "nbhd_id"], how="outer", suffixes=("", "_dup")
        )
        result = result.drop(
            columns=[c for c in result.columns if c.endswith("_dup")]
        )

    result = result.sort_values(["date", "nbhd_id"]).reset_index(drop=True)

    #OUTPUT VALIDATION
    pred_df = nbhd_predictions_schema.validate(pred_df)
    validate_unique_key(pred_df, ["date", "nbhd_id"], "neighbourhood predictions")
    validate_expected_neighbourhoods(pred_df, "neighbourhood predictions")

    result.to_parquet(NBHD_PRED_PATH, index=False)
    with open(NBHD_METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n  Saved neighbourhood predictions → {NBHD_PRED_PATH}  "
          f"shape={result.shape}")
    print(f"  Saved neighbourhood metrics     → {NBHD_METRICS_PATH}")

    print("\n  ── Final summary ──")
    for m in all_metrics:
        p10 = m.get("precision_at_10", "N/A")
        print(f"  {m['model']:30s}  ROC={m['roc_auc']:.4f}  "
              f"PR={m['pr_auc']:.4f}  Brier={m['brier']:.4f}  P@10={p10}")

    return result
#%% md
# **Section 5: Pipeline runner**
#%%
# ── SECTION 5: Pipeline runner ────────────────────────────────────────────────
@flow(name="CoverCheck Main Pipeline", log_prints=True)
def run_pipeline_flow(start_from: str = "features", only: str | None = None):
    """The main Prefect orchestration flow."""

    # Run a single stage if requested
    if only == "features":
        run_feature_builder()
        return
    elif only == "surge":
        run_surge_classifier()
        return
    elif only == "nbhd":
        run_nbhd_models()
        return

    # Otherwise, run the DAG sequentially based on start_from
    if start_from == "features":
        run_feature_builder()
        run_surge_classifier()
        run_nbhd_models()
    elif start_from == "surge":
        run_surge_classifier()
        run_nbhd_models()
    elif start_from == "nbhd":
        run_nbhd_models()

# ── CLI entry point ───────────────────────────────────────────────────────────
import typer

app = typer.Typer(help="CoverCheck Toronto — Production ML Pipeline")

@app.command()
def run(
    start_from: str = typer.Option("features", help="Start from: features | surge | nbhd"),
    only: str = typer.Option(None, help="Run only this stage: features | surge | nbhd")
):
    """Execute the CoverCheck pipeline stages."""
    run_pipeline_flow(start_from=start_from, only=only)

if __name__ == "__main__":
    app()
#%%
