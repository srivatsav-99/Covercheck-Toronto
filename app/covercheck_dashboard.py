"""
CoverCheck Toronto — Streamlit Dashboard (Decoupled Hybrid Version)
Light mode only. Professional blue palette (#5DA5DA brand colour).

Usage:
    poetry run python -m streamlit run app/covercheck_dashboard.py
"""
from __future__ import annotations

import base64
import sys
import warnings
from pathlib import Path

import requests

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ONLY import paths. NO pipeline logic.
from src.io_paths import (
    ASSETS_DIR,
    FEATURES_PATH,
    NBHD_PRED_PATH,
    DIM_PATH,
    COLLISION_FI_PATH,
)

warnings.filterwarnings("ignore")

# ── API Configuration ────────────────────────────────────────────────────────
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 15

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
B_SIDEBAR = "#1B4F72"
B_SIDE_TX = "#EBF5FB"
B_DEEP = "#1A2E44"
B_LABEL = "#2E6DA4"
B_MUTED = "#5D7A8A"
B_PAGE = "#F0F6FB"
B_CARD = "#FFFFFF"
B_FILL = "#EBF5FB"
B_BORDER = "#C8E0F0"

C_BLUE = "#2980B9"
C_RED = "#C0392B"
C_AMBER = "#CA6F1E"
C_GREEN = "#1E8449"
C_TEAL = "#1A6B57"
C_GOLD = "#B7950B"
C_SLATE = "#5D7A8A"
C_GREY = "#9EB3C2"

G_GRID = "rgba(41,128,185,0.07)"
LOGO_PATH = ASSETS_DIR / "covercheck_logo.png"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CoverCheck Toronto",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
  .stApp {{
    background-color: {B_PAGE} !important;
  }}

  [data-testid="stAppViewContainer"] {{
    background-color: {B_PAGE} !important;
  }}

  section[data-testid="stSidebar"] {{
    background: {B_SIDEBAR} !important;
  }}

  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div,
  section[data-testid="stSidebar"] caption {{
    color: {B_SIDE_TX} !important;
  }}

  section[data-testid="stSidebar"] hr {{
    border-color: {B_SIDE_TX} !important;
    opacity: 0.25;
  }}

  section[data-testid="stSidebar"] .stButton > button {{
    background: {C_BLUE} !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
  }}

  section[data-testid="stSidebar"] .stButton > button:hover {{
    background: #1A6FA8 !important;
  }}

  .metric-card {{
    background: {B_CARD};
    border: 1.5px solid {B_BORDER};
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    height: 100%;
    box-sizing: border-box;
  }}

  .metric-label {{
    font-size: 11px;
    color: {B_LABEL};
    font-weight: 700;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }}

  .metric-value {{
    font-size: 24px;
    font-weight: 700;
    color: {B_DEEP};
    line-height: 1.2;
  }}

  .metric-sub {{
    font-size: 11px;
    color: {B_MUTED};
    margin-top: 4px;
  }}

  .risk-high {{
    color: {C_RED} !important;
  }}

  .risk-medium {{
    color: {C_AMBER} !important;
  }}

  .risk-low {{
    color: {C_GREEN} !important;
  }}

  .section-header {{
    font-size: 14px;
    font-weight: 700;
    color: {B_DEEP};
    margin: 1.2rem 0 0.4rem 0;
    padding-bottom: 6px;
    border-bottom: 2px solid {B_BORDER};
  }}

  .alert-high {{
    background: #FDEDEC;
    border: 1.5px solid {C_RED};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 12px;
    color: {B_DEEP};
  }}

  .alert-medium {{
    background: #FDF3E7;
    border: 1.5px solid {C_AMBER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 12px;
    color: {B_DEEP};
  }}

  .alert-low {{
    background: #E9F7EF;
    border: 1.5px solid {C_GREEN};
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 12px;
    color: {B_DEEP};
  }}

  .profile-panel {{
    background: {B_CARD};
    border: 1.5px solid {B_BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 6px;
  }}

  .profile-name {{
    font-size: 15px;
    font-weight: 700;
    color: {B_DEEP};
    margin-bottom: 10px;
  }}

  .mini-stat {{
    text-align: center;
    padding: 8px;
    background: {B_FILL};
    border-radius: 8px;
    border: 1px solid {B_BORDER};
  }}

  .mini-stat-label {{
    font-size: 10px;
    color: {B_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }}

  .mini-stat-value {{
    font-size: 20px;
    font-weight: 700;
    color: {B_DEEP};
  }}

  .driver-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px 3px 2px 0;
  }}

  .driver-weather {{
    background: #D6EAF8;
    color: #1A5276;
  }}

  .driver-traffic {{
    background: #D4E6F1;
    color: {B_DEEP};
  }}

  .driver-calendar {{
    background: #FEF3E2;
    color: #7D4E08;
  }}

  .driver-road {{
    background: #D5EFE3;
    color: #1A5738;
  }}

  .def-box {{
    background: {B_CARD};
    border-left: 4px solid {C_BLUE};
    border-top: 0.5px solid {B_BORDER};
    border-right: 0.5px solid {B_BORDER};
    border-bottom: 0.5px solid {B_BORDER};
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
  }}

  .def-title {{
    font-size: 13px;
    font-weight: 700;
    color: {B_DEEP};
    margin-bottom: 6px;
  }}

  .def-row {{
    font-size: 12px;
    color: #2C3E50;
    margin-bottom: 4px;
    line-height: 1.55;
  }}

  .def-label {{
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 4px;
    margin-right: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  .lbl-what {{
    background: #D4E6F1;
    color: #1A3E5C;
  }}

  .lbl-why {{
    background: #D5EFE3;
    color: #1A5738;
  }}

  .lbl-how {{
    background: #FEF3E2;
    color: #7D4E08;
  }}

  .legend-row {{
    display: flex;
    gap: 18px;
    align-items: center;
    flex-wrap: wrap;
    padding: 8px 14px;
    background: {B_CARD};
    border: 1px solid {B_BORDER};
    border-radius: 8px;
    margin-top: 6px;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: {B_DEEP};
    white-space: nowrap;
  }}

  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  .why-row {{
    margin-bottom: 8px;
    padding: 8px 12px;
    background: {B_FILL};
    border-radius: 8px;
    border: 1px solid {B_BORDER};
  }}

  .why-name {{
    font-size: 13px;
    font-weight: 700;
    color: {B_DEEP};
  }}

  .why-score {{
    font-size: 11px;
    color: {B_MUTED};
    margin-left: 8px;
  }}

  .why-tags {{
    margin-top: 5px;
  }}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI CACHED LOADERS (LIVE DATA)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=60)
def api_get_health():
    resp = requests.get(f"{API_BASE_URL}/health", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False, ttl=60)
def api_get_latest_surge():
    resp = requests.get(f"{API_BASE_URL}/surge/latest", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False, ttl=60)
def api_get_topk(horizon: int = 1, k: int = 10):
    resp = requests.get(
        f"{API_BASE_URL}/neighbourhoods/topk",
        params={"horizon": horizon, "k": k},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False, ttl=60)
def api_get_metrics():
    resp = requests.get(f"{API_BASE_URL}/metrics", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL ARTIFACT LOADERS (HEAVY HISTORICAL DATA & MAPS)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_features():
    if not FEATURES_PATH.exists():
        return pd.DataFrame()

    df = pd.read_parquet(FEATURES_PATH).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "nbhd_id" in df.columns:
        df["nbhd_id"] = pd.to_numeric(df["nbhd_id"], errors="coerce")
        df = df.dropna(subset=["nbhd_id"])
        df["nbhd_id"] = df["nbhd_id"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_nbhd():
    if not NBHD_PRED_PATH.exists():
        return pd.DataFrame()

    df = pd.read_parquet(NBHD_PRED_PATH).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "nbhd_id" in df.columns:
        df["nbhd_id"] = pd.to_numeric(df["nbhd_id"], errors="coerce")
        df = df.dropna(subset=["nbhd_id"])
        df["nbhd_id"] = df["nbhd_id"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_dim():
    if not DIM_PATH.exists():
        return gpd.GeoDataFrame()

    dim = gpd.read_parquet(DIM_PATH).copy()
    if "nbhd_id" in dim.columns:
        dim["nbhd_id"] = pd.to_numeric(dim["nbhd_id"], errors="coerce")
        dim = dim.dropna(subset=["nbhd_id"])
        dim["nbhd_id"] = dim["nbhd_id"].astype(int)
    return dim


def clear_and_refresh() -> None:
    st.cache_data.clear()
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
FRIENDLY = {
    "collisions_lag1": "Yesterday's Collisions",
    "collisions_roll7_mean": "7-Day Average Collisions",
    "collisions_roll14_mean": "14-Day Average Collisions",
    "collisions_roll30_mean": "30-Day Average Collisions",
    "collision_momentum": "Collision Spike vs Baseline",
    "ksi_ratio_lag1": "Yesterday's Serious Incident Rate",
    "tavg": "Average Temperature",
    "tmin": "Minimum Temperature",
    "prcp": "Rainfall Amount",
    "snow": "Snowfall Amount",
    "wspd": "Wind Speed",
    "freezing_rain": "Freezing Rain",
    "heavy_snow": "Heavy Snow",
    "feels_bad": "Adverse Weather",
    "is_holiday": "Public Holiday",
    "pre_holiday": "Day Before A Holiday",
    "post_holiday": "Day After A Holiday",
    "is_weekend": "Weekend",
    "dow_sin": "Day Of Week Pattern",
    "month_sin": "Time Of Year Pattern",
    "road_construction_count_halo_lag1": "Nearby Construction Yesterday",
    "road_construction_count_lag1": "Local Construction Yesterday",
    "is_zone_disrupted": "Neighbouring Zone Disrupted",
    "construction_total_pressure": "Road Disruption Pressure",
    "intersection_count": "Number Of Intersections",
    "arterial_length_km": "Total Road Length",
    "streets_per_node_avg": "Road Network Density",
    "surge_proba_t1": "Citywide Surge Risk",
    "surge_proba_t2": "Citywide Surge Risk",
}

def resolve_nbhd_prediction_columns(df: pd.DataFrame, horizon: int) -> tuple[str, str]:
    score_candidates = [
        f"risk_score_t{horizon}",
        f"collision_prob_t{horizon}",
        f"proba_collision_t{horizon}",
    ]
    prob_candidates = [
        f"collision_prob_t{horizon}",
        f"proba_collision_t{horizon}",
        f"risk_score_t{horizon}",
    ]

    score_col = next((c for c in score_candidates if c in df.columns), None)
    prob_col = next((c for c in prob_candidates if c in df.columns), None)

    if score_col is None:
        raise ValueError(
            f"Could not find a risk score column for T+{horizon}. "
            f"Available columns: {df.columns.tolist()}"
        )
    if prob_col is None:
        prob_col = score_col

    return score_col, prob_col

def risk_colour(score: float) -> str:
    if score >= 0.7:
        return "risk-high"
    if score >= 0.4:
        return "risk-medium"
    return "risk-low"


def metric_card(label: str, value: str, sub: str = "", colour_class: str = "") -> None:
    val_class = f"metric-value {colour_class}" if colour_class else "metric-value"
    st.markdown(
        f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{val_class}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def risk_score_legend() -> None:
    st.markdown(
        f"""
    <div class="legend-row">
        <span style="font-size:12px; font-weight:700; color:{B_DEEP};">Risk Score</span>
        <div class="legend-item"><div class="legend-dot" style="background:{C_RED};"></div><span>High (0.70 and above)</span></div>
        <div class="legend-item"><div class="legend-dot" style="background:{C_AMBER};"></div><span>Medium (0.40 to 0.69)</span></div>
        <div class="legend-item"><div class="legend-dot" style="background:{C_GREEN};"></div><span>Low (below 0.40)</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def style_risk_table(styler: "pd.io.formats.style.Styler") -> "pd.io.formats.style.Styler":
    def colour_score(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v >= 0.7:
            return f"background-color: #FDECEA; color: {C_RED}; font-weight: 700;"
        if v >= 0.4:
            return f"background-color: #FDF3E7; color: {C_AMBER}; font-weight: 700;"
        return f"background-color: #E9F7EF; color: {C_GREEN}; font-weight: 700;"

    return styler.map(colour_score, subset=["Risk Score"])


def colour_legend(items: list[tuple[str, str]]) -> None:
    dots = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:5px;margin-right:16px;font-size:12px;color:{B_DEEP};white-space:nowrap;'><span style='width:10px;height:10px;border-radius:50%;background:{col};display:inline-block;flex-shrink:0;'></span>{lbl}</span>"
        for col, lbl in items
    )
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;padding:6px 12px;background:{B_CARD};border:1px solid {B_BORDER};border-radius:8px;margin-bottom:8px;'>{dots}</div>",
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=44, b=0),
        plot_bgcolor=B_CARD,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=B_DEEP, size=12, family="Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color=B_DEEP),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=B_BORDER,
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor=G_GRID,
            linecolor=B_BORDER,
            zeroline=False,
            tickfont=dict(color=B_MUTED, size=11),
        ),
        yaxis=dict(
            gridcolor=G_GRID,
            linecolor=B_BORDER,
            zeroline=False,
            tickfont=dict(color=B_MUTED, size=11),
        ),
        hovermode="x unified",
    )
    return fig


def nbhd_avg_collisions(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["nbhd_id", "avg_30d"])

    result = (
        features.sort_values("date")
        .groupby("nbhd_id", group_keys=False)
        .apply(lambda x: pd.Series({"avg_30d": x["collisions"].tail(30).mean()}))
        .reset_index()
    )
    result["nbhd_id"] = result["nbhd_id"].astype(int)
    return result


def build_map(
    gdf: gpd.GeoDataFrame,
    score_col: str,
    legend_name: str,
    colour_scale: str = "Blues",
    top_k_ids: list[int] | None = None,
    tip_fields: list[str] | None = None,
    tip_aliases: list[str] | None = None,
) -> folium.Map:
    import branca.colormap as bcm

    gdf = gdf.copy()

    # Ensure CRS is correct for Folium
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # Clean score column
    gdf[score_col] = pd.to_numeric(gdf[score_col], errors="coerce")
    gdf[score_col] = gdf[score_col].fillna(0)

    # Map center
    m = folium.Map(
        location=[43.72, -79.38],
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    # Manual colormap instead of folium.Choropleth
    vmin = float(gdf[score_col].min())
    vmax = float(gdf[score_col].max())
    if vmin == vmax:
        vmax = vmin + 1e-6

    cmap = bcm.linear.Blues_09.scale(vmin, vmax)
    cmap.caption = legend_name
    cmap.add_to(m)

    tf = tip_fields or ["area_name", score_col]
    ta = tip_aliases or ["Neighbourhood", legend_name]

    def style_function(feature):
        props = feature["properties"]
        val = props.get(score_col, 0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        return {
            "fillColor": cmap(val),
            "color": "#5D6D7E",
            "weight": 0.8,
            "fillOpacity": 0.72,
        }

    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tf,
            aliases=ta,
            localize=True,
            sticky=False,
            labels=True,
            style=(
                "background-color: white; color: #1A2E44; "
                "font-family: Arial; font-size: 12px; padding: 8px;"
            ),
        ),
    ).add_to(m)

    # Outline top-k zones
    if top_k_ids:
        top_gdf = gdf[gdf["nbhd_id"].isin(top_k_ids)].copy()
        if not top_gdf.empty:
            folium.GeoJson(
                top_gdf,
                style_function=lambda feature: {
                    "fillOpacity": 0,
                    "color": C_RED,
                    "weight": 3,
                    "dashArray": "5 3",
                },
            ).add_to(m)

    return m

def why_high_risk(nbhd_id: int, features: pd.DataFrame) -> list[tuple[str, str]]:
    if features.empty:
        return [("traffic", "Elevated collision history")]

    sub = features[features["nbhd_id"] == nbhd_id].sort_values("date")
    if sub.empty:
        return [("traffic", "High recent collision activity")]

    latest = sub.iloc[-1]
    avg_coll = sub["collisions"].tail(30).mean()
    reasons = []

    lag1 = float(latest.get("collisions_lag1", 0) or 0)
    if lag1 > avg_coll * 1.5:
        reasons.append(("traffic", f"Yesterday had {int(lag1)} collisions (above average)"))

    mom = float(latest.get("collision_momentum", 1.0) or 1.0)
    if mom > 1.4:
        reasons.append(("traffic", f"Collision activity is {mom:.1f}x the recent baseline"))

    if latest.get("freezing_rain", 0):
        reasons.append(("weather", "Freezing rain forecast"))
    elif latest.get("heavy_snow", 0):
        reasons.append(("weather", "Heavy snow forecast"))
    elif latest.get("feels_bad", 0):
        reasons.append(("weather", "Adverse weather conditions"))

    if latest.get("is_holiday", 0):
        reasons.append(("calendar", "Public holiday"))
    elif latest.get("pre_holiday", 0):
        reasons.append(("calendar", "Day before a public holiday"))
    elif latest.get("is_weekend", 0):
        reasons.append(("calendar", "Weekend traffic patterns"))

    if latest.get("road_construction_count_lag1", 0) > 0:
        reasons.append(("road", "Active road construction nearby"))
    elif latest.get("is_zone_disrupted", 0):
        reasons.append(("road", "Road disruptions in adjacent zones"))

    if not reasons:
        reasons.append(("traffic", "Elevated collision history in this area"))

    return reasons[:4]


# ══════════════════════════════════════════════════════════════════════════════
# DEFINITIONS PANEL
# ══════════════════════════════════════════════════════════════════════════════
DEFINITIONS = {
    "City Overview": [
        {
            "title": "Surge Risk",
            "what": "The probability that tomorrow will be a high-collision day across all of Toronto.",
            "why": "Tells teams whether to expect above-normal activity citywide before the day begins.",
            "how": "🔴 Red = High risk (70%+). 🟡 Amber = Moderate (40–69%). 🟢 Green = Low (below 40%).",
        },
        {
            "title": "Expected Collisions",
            "what": "An estimate of how many collisions are likely tomorrow across the entire city.",
            "why": "Helps teams plan resource levels.",
            "how": "Based on the 30-day historical average adjusted by the surge risk level.",
        },
        {
            "title": "Daily Collision Trend",
            "what": "How daily collision counts have moved over the past 90 days vs the 30-day average.",
            "why": "Reveals whether the city is in a rising or falling risk period.",
            "how": "🔴 Red shading = above the 30-day average. 🟢 Green = below. Gold dotted line = 30-day average.",
        },
        {
            "title": "Weather Conditions",
            "what": "Current weather readings including temperature, rain, snow, and wind.",
            "why": "Adverse weather is one of the strongest drivers of collision risk.",
            "how": "Road Conditions turns 🔴 red when freezing rain or heavy snow is active.",
        },
    ],
    "Risk Map": [
        {
            "title": "Risk Score",
            "what": "A score from 0 to 1 showing how likely a collision is in each neighbourhood tomorrow.",
            "why": "Lets teams identify where risk is concentrated.",
            "how": "🔴 Red (0.70+) = High. 🟡 Amber (0.40–0.69) = Medium. 🟢 Green (below 0.40) = Low. Red outlines = top-ranked zones.",
        },
        {
            "title": "Neighbourhood Profile",
            "what": "Detailed breakdown — risk score, collision chance, expected collisions, and risk drivers.",
            "why": "A number alone does not tell you what to do. The profile explains why an area is risky.",
            "how": "Click any neighbourhood on the map. The profile appears on the right side of the page.",
        },
        {
            "title": "Risk Drivers",
            "what": "Tags explaining what is pushing the risk score up for each neighbourhood.",
            "why": "Helps teams decide the right response.",
            "how": "🔵 Blue = Traffic spike. 🔷 Light blue = Weather. 🟡 Gold = Calendar. 🟢 Green = Road disruption.",
        },
        {
            "title": "Zone Counts",
            "what": "How many of the 158 Toronto neighbourhoods are at High, Medium, and Low risk today.",
            "why": "Shows whether today is a concentrated or widespread risk day across the city.",
            "how": "🔴 High = 0.70+. 🟡 Medium = 0.40–0.69. 🟢 Low = below 0.40.",
        },
    ],
    "Trends And Seasonality": [
        {
            "title": "Collision Volume",
            "what": "Historical collision counts by year, month, or day of week — select the view using the dropdown.",
            "why": "Reveals patterns that repeat every year so teams can plan ahead.",
            "how": "🔴 Red bars = highest-risk periods. 🟢 Green = lowest-risk. ⬜ Grey = typical.",
        },
        {
            "title": "Weather Impact",
            "what": "How much adverse weather conditions increase average daily collisions vs normal days.",
            "why": "Quantifies the weather risk.",
            "how": "🔵 Blue bar = normal days. 🔴 Red bars = adverse conditions.",
        },
        {
            "title": "Serious Incident Trend",
            "what": "Monthly count of collisions resulting in serious injury or fatality (KSI) over time.",
            "why": "Tracks whether the most severe incidents are getting better or worse long-term.",
            "how": "🔴 Red shading = above long-term average (worsening). 🟢 Green = below average (improving).",
        },
    ],
    "Model Performance": [
        {
            "title": "Accuracy Score (ROC-AUC)",
            "what": "How well the model separates high-risk days from low-risk days.",
            "why": "A higher score means the model is more reliable for operational decisions.",
            "how": "0.50 = random guessing. 0.70+ = strong. CoverCheck scores around 0.72 on collision forecasts.",
        },
        {
            "title": "Ranking Accuracy (Precision@K)",
            "what": "Of the top K flagged neighbourhoods, how many actually had a collision that day.",
            "why": "The most operationally important metric — tells you whether acting on the list works.",
            "how": "90% at Top 10 means 9 out of 10 flagged zones had an actual incident.",
        },
        {
            "title": "Coverage (Recall@K)",
            "what": "Of all neighbourhoods that had a collision, what share were in the flagged list.",
            "why": "Measures how many real incidents the model would have helped prevent.",
            "how": "Higher is better. Low coverage means some high-risk zones were missed.",
        },
        {
            "title": "Top Contributing Factors",
            "what": "The data inputs that most influence the model's risk predictions.",
            "why": "Confirms the model is learning from sensible signals — not noise or coincidence.",
            "how": "Longer bars = greater influence on the prediction.",
        },
        {
            "title": "Forecast Stability",
            "what": "Whether model accuracy holds up consistently across the full test period.",
            "why": "A model that degrades over time needs retraining — this check flags it early.",
            "how": "🟢 Stable = accuracy is consistent. ⚠️ Warning = significant drop.",
        },
    ],
}


def render_definitions(page: str) -> None:
    defs = DEFINITIONS.get(page, [])
    if not defs:
        return

    with st.expander("What Does This Page Show? Click Here For Definitions", expanded=False):
        for d in defs:
            st.markdown(
                f"""
            <div class="def-box">
                <div class="def-title">{d['title']}</div>
                <div class="def-row"><span class="def-label lbl-what">What</span>{d['what']}</div>
                <div class="def-row"><span class="def-label lbl-why">Why It Matters</span>{d['why']}</div>
                <div class="def-row"><span class="def-label lbl-how">How To Read It</span>{d['how']}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> dict:
    with st.sidebar:
        if LOGO_PATH.exists():
            with open(LOGO_PATH, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()

            st.markdown(
                "<div style='background:#FFFFFF; border-radius:10px; padding:12px 16px; margin-bottom:8px; text-align:center;'>"
                f"<img src='data:image/png;base64,{logo_b64}' style='width:100%; max-width:240px; height:auto;'>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:center; padding:0.6rem 0 0.8rem;'>"
                f"<div style='font-size:1.2rem; font-weight:700; color:{B_DEEP};'>🚦 CoverCheck</div>"
                f"<div style='font-size:0.72rem; color:{B_DEEP}; opacity:0.7; margin-top:2px;'>Toronto Collision Risk Forecasting</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        try:
            api_health = api_get_health()
            st.success(f"API Connected: {api_health['status']}")
        except Exception as e:
            st.error(f"API Unavailable: {e}")
            st.warning("Please ensure the FastAPI backend is running on port 8000.")
            st.stop()

        st.divider()

        st.markdown(
            f"<div style='font-size:11px; font-weight:700; color:{B_SIDE_TX}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;'>Forecast Horizon</div>",
            unsafe_allow_html=True,
        )
        horizon = st.radio(
            "Forecast Horizon",
            options=[1, 2],
            format_func=lambda x: "Tomorrow" if x == 1 else "Day After Tomorrow",
            label_visibility="collapsed",
        )

        st.markdown(
            f"<div style='font-size:11px; font-weight:700; color:{B_SIDE_TX}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px; margin-top:12px;'>High-Risk Zones To Show</div>",
            unsafe_allow_html=True,
        )
        top_k = st.select_slider("Zones", options=[5, 10, 15, 20], value=10, label_visibility="collapsed")

        st.divider()

        if st.button("Fetch Latest Forecasts", type="primary", width="stretch"):
            clear_and_refresh()

        st.caption("Pulls fresh data from the FastAPI serving layer.")

        st.divider()

        st.markdown(
            f"<div style='font-size:11px; font-weight:700; color:{B_SIDE_TX}; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;'>Pages</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Pages",
            ["City Overview", "Risk Map", "Trends And Seasonality", "Model Performance"],
            label_visibility="collapsed",
        )

    return {"horizon": horizon, "top_k": top_k, "page": page}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CITY OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_city_overview(features: pd.DataFrame, horizon: int, top_k: int) -> None:
    hl = "Tomorrow" if horizon == 1 else "Day After Tomorrow"
    st.markdown(f"<h2 style='color:{B_DEEP}; margin-bottom:2px;'>City Overview</h2>", unsafe_allow_html=True)
    st.caption(f"Operational snapshot for {hl.lower()} across Toronto.")
    render_definitions("City Overview")

    latest_surge = api_get_latest_surge()
    sp = float(latest_surge.get(f"surge_proba_t{horizon}", 0) or 0)

    fdate = pd.Timestamp(latest_surge["date"]).strftime("%B %d, %Y")

    if sp >= 0.7:
        bclass, bicon, bmsg = (
            "alert-high",
            "🔴",
            f"High surge risk for {hl.lower()} ({sp:.0%}). Collision activity is expected to be significantly above normal.",
        )
    elif sp >= 0.4:
        bclass, bicon, bmsg = (
            "alert-medium",
            "🟡",
            f"Moderate surge risk for {hl.lower()} ({sp:.0%}). Collision activity may be above normal in several areas.",
        )
    else:
        bclass, bicon, bmsg = (
            "alert-low",
            "🟢",
            f"Low surge risk for {hl.lower()} ({sp:.0%}). No citywide surge is expected.",
        )

    st.markdown(f'<div class="{bclass}">{bicon}&nbsp; {bmsg}</div>', unsafe_allow_html=True)

    avg_30 = 0
    daily_city = pd.DataFrame()

    if not features.empty:
        daily_city = (
            features.groupby("date", as_index=False)
            .agg(city_collisions=("collisions", "sum"))
            .sort_values("date")
        )
        avg_30 = daily_city.tail(30)["city_collisions"].mean()


    expected = int(avg_30 * (1 + (sp - 0.5))) if avg_30 else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(f"Surge Risk — {hl}", f"{sp:.0%}", sub=f"Forecast for {fdate}", colour_class=risk_colour(sp))
    with c2:
        metric_card("Expected Collisions", f"~{expected:,}", sub=f"30-day city average: {avg_30:,.0f}/day")
    with c3:
        dispatch_label = "Elevated Readiness" if sp >= 0.7 else "Watch" if sp >= 0.4 else "Normal Readiness"
        dispatch_class = "risk-high" if sp >= 0.7 else "risk-medium" if sp >= 0.4 else "risk-low"

        metric_card(
            "Dispatch Posture",
            dispatch_label,
            sub=f"Based on {hl.lower()} citywide surge risk",
            colour_class=dispatch_class,
        )

    st.markdown("---")

    section(f"Top {top_k} Highest-Risk Neighbourhoods For {hl}")
    st.caption("These are the areas most likely to experience a collision. Scores are fetched live from the API.")

    top_payload = api_get_topk(horizon=horizon, k=top_k)
    top_rows = pd.DataFrame(top_payload["records"])

    if not top_rows.empty:
        score_col_api = f"collision_prob_t{horizon}"
        if score_col_api not in top_rows.columns:
            score_col_api = f"risk_score_t{horizon}"

        prob_col_api = f"collision_prob_t{horizon}"
        if prob_col_api not in top_rows.columns:
            prob_col_api = score_col_api

        top_rows.index = top_rows.index + 1

        if not features.empty:
            avgs = nbhd_avg_collisions(features)
            top_rows = top_rows.merge(avgs, on="nbhd_id", how="left")
            top_rows["avg_30d"] = top_rows["avg_30d"].fillna(avg_30 / 158 if avg_30 else 1.0)
            top_rows["exp_coll"] = (top_rows["avg_30d"] * top_rows[prob_col_api] * 2).round(1)
        else:
            top_rows["exp_coll"] = 0.0

        display_df = top_rows[["area_name", score_col_api, prob_col_api, "exp_coll"]].copy()
        display_df.columns = ["Neighbourhood", "Risk Score", "Collision Chance", "Expected Collisions"]

        st.dataframe(
            display_df.style.pipe(style_risk_table).format(
                {
                    "Risk Score": "{:.3f}",
                    "Collision Chance": "{:.1%}",
                    "Expected Collisions": "{:.1f}",
                }
            ),
            width="stretch",
            height=min(80 + top_k * 38, 480),
        )

    risk_score_legend()

    st.markdown("---")

    section("Daily Collision Trend (Past 90 Days)")
    if not features.empty and not daily_city.empty:
        last_90 = daily_city.tail(90).copy()
        last_90["date"] = pd.to_datetime(last_90["date"])
        avg_line = daily_city.tail(30)["city_collisions"].mean()
        last_90["avg_30"] = avg_line

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=last_90["date"],
                y=last_90["city_collisions"].clip(upper=avg_line),
                mode="none",
                fill=None,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=last_90["date"],
                y=last_90["avg_30"],
                mode="none",
                fill="tonexty",
                fillcolor="rgba(30,132,73,0.12)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=last_90["date"],
                y=last_90["avg_30"],
                mode="none",
                fill=None,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=last_90["date"],
                y=last_90["city_collisions"].clip(lower=avg_line),
                mode="none",
                fill="tonexty",
                fillcolor="rgba(192,57,43,0.10)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=last_90["date"],
                y=last_90["avg_30"],
                mode="lines",
                name=f"30-Day Avg ({avg_line:.0f})",
                line=dict(color=C_GOLD, width=1.8, dash="dot"),
                hovertemplate="30-Day Avg: %{y:.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=last_90["date"],
                y=last_90["city_collisions"],
                mode="lines",
                name="Daily Collisions",
                line=dict(color=B_DEEP, width=1.6),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Collisions: %{y:,}<extra></extra>",
            )
        )

        recent_7 = last_90.tail(7)["city_collisions"].mean()
        ann_col, ann_txt = (C_RED, "↑ Above normal") if recent_7 > avg_line else (C_GREEN, "↓ Below normal")
        fig.add_annotation(
            x=last_90["date"].iloc[-1],
            y=last_90["city_collisions"].iloc[-1],
            text=ann_txt,
            showarrow=False,
            font=dict(size=11, color=ann_col),
            xanchor="right",
            yanchor="bottom",
        )

        fig = chart_layout(fig, height=300)
        fig.update_layout(yaxis_title="Collisions Per Day", xaxis_title="")
        colour_legend(
            [
                (C_RED, "Above 30-day average"),
                (C_GREEN, "Below 30-day average"),
                (C_GOLD, "30-day average"),
            ]
        )
        st.plotly_chart(fig, width="stretch")

        section("Current Weather Conditions")
        st.caption(
            "Weather is one of the strongest drivers of collision risk. Freezing rain and heavy snow days see significantly more incidents."
        )

        lw = features.sort_values("date").drop_duplicates("date").iloc[-1]
        w1, w2, w3, w4, w5 = st.columns(5)
        with w1:
            metric_card("Temperature", f"{float(lw.get('tavg', 0)):.1f}°C", sub="Average Today")
        with w2:
            metric_card("Rainfall", f"{float(lw.get('prcp', 0)):.1f} mm", sub="Precipitation")
        with w3:
            metric_card("Snowfall", f"{float(lw.get('snow', 0)):.1f} cm", sub="Snow Accumulation")
        with w4:
            metric_card("Wind Speed", f"{float(lw.get('wspd', 0)):.0f} km/h", sub="Max Wind Today")
        with w5:
            conds = []
            if lw.get("freezing_rain", 0):
                conds.append("Freezing Rain")
            if lw.get("heavy_snow", 0):
                conds.append("Heavy Snow")
            if lw.get("feels_bad", 0) and not conds:
                conds.append("Adverse")
            metric_card(
                "Road Conditions",
                ", ".join(conds) if conds else "Normal",
                colour_class="risk-high" if conds else "risk-low",
            )



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK MAP
# ══════════════════════════════════════════════════════════════════════════════
def page_risk_map(dim: gpd.GeoDataFrame, features: pd.DataFrame, nbhd: pd.DataFrame, horizon: int, top_k: int) -> None:
    hl = "Tomorrow" if horizon == 1 else "Day After Tomorrow"
    st.markdown(f"<h2 style='color:{B_DEEP}; margin-bottom:2px;'>Neighbourhood Risk Map</h2>", unsafe_allow_html=True)
    st.caption(f"Which neighbourhoods are most likely to experience elevated collision risk {hl.lower()}?")
    render_definitions("Risk Map")

    if dim.empty or nbhd.empty:
        st.warning("Map boundaries or local predictions unavailable. Please run the data pipeline first.")
        return


    score_col, prob_col = resolve_nbhd_prediction_columns(nbhd, horizon)

    latest_date = nbhd["date"].max()
    pred_today = nbhd[nbhd["date"] == latest_date].copy()
    pred_today["nbhd_id"] = pd.to_numeric(pred_today["nbhd_id"], errors="coerce").astype(int)

    avgs = nbhd_avg_collisions(features)
    pred_today = pred_today.merge(avgs, on="nbhd_id", how="left")
    pred_today["avg_30d"] = pred_today["avg_30d"].fillna(1.0)


    pred_today["exp_coll"] = (pred_today["avg_30d"] * pred_today[score_col] * 2).round(1)

    gdf = dim.copy()
    gdf["nbhd_id"] = pd.to_numeric(gdf["nbhd_id"], errors="coerce").astype(int)

    # Safely extract columns that actually exist in the dataframe
    cols_to_merge = ["nbhd_id", score_col, "exp_coll"]
    if prob_col in pred_today.columns:
        cols_to_merge.append(prob_col)

    gdf = gdf.merge(pred_today[cols_to_merge], on="nbhd_id", how="left")
    gdf[score_col] = gdf[score_col].round(3)
    if prob_col in gdf.columns:
        gdf[prob_col] = gdf[prob_col].round(3)

    top_k_ids = pred_today.nlargest(top_k, score_col)["nbhd_id"].tolist()

    col_map, col_right = st.columns([3, 2])


    with col_map:
        section(f"Risk Map — Top {top_k} Zones Outlined In Red")
        colour_legend(
            [
                (C_RED, "High Risk (0.70+)"),
                (C_AMBER, "Medium Risk (0.40–0.69)"),
                (C_GREEN, "Low Risk (below 0.40)"),
                ("#C8E0F0", "Top zones outlined in red on map"),
            ]
        )
        st.caption("Click a neighbourhood to inspect its detailed profile.")

        tip_fields = ["area_name", score_col]
        tip_aliases = ["Neighbourhood", "Risk Score"]
        if prob_col in gdf.columns:
            tip_fields.append(prob_col)
            tip_aliases.append("Collision Chance")

        m = build_map(
            gdf,
            score_col,
            "Overall Risk Score",
            "Blues",
            top_k_ids,
            tip_fields,
            tip_aliases,
        )

        map_data = st_folium(
            m,
            width=900,
            height=520,
            returned_objects=["last_object_clicked_tooltip"],
        )

    with col_right:
        selected_name = None
        if map_data and map_data.get("last_object_clicked_tooltip"):
            tooltip = map_data["last_object_clicked_tooltip"]

            if isinstance(tooltip, dict):
                selected_name = (
                        tooltip.get("Neighbourhood")
                        or tooltip.get("area_name")
                        or tooltip.get("neighbourhood")
                )
            elif isinstance(tooltip, list) and len(tooltip) > 0:
                maybe_dict = tooltip[0]
                if isinstance(maybe_dict, dict):
                    selected_name = (
                            maybe_dict.get("Neighbourhood")
                            or maybe_dict.get("area_name")
                            or maybe_dict.get("neighbourhood")
                    )

        row = None
        if selected_name:
            matches = gdf[gdf["area_name"] == selected_name]
            if not matches.empty:
                row = matches.iloc[0]

        if row is not None:
            score = float(row.get(score_col, 0) or 0)
            exp_v = f"{float(row.get('exp_coll', 0)):.1f}"
            collision_prob = float(row.get(prob_col, score) or 0)
            sc_col = C_RED if score >= 0.7 else C_AMBER if score >= 0.4 else C_GREEN

            drv_html = "".join(
                f'<span class="driver-badge driver-{cat}">{txt}</span>'
                for cat, txt in why_high_risk(int(row["nbhd_id"]), features)
            )

            st.markdown(
                f"""
            <div class="profile-panel">
                <div class="profile-name">📍 {selected_name}</div>
                <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:12px;'>
                    <div class="mini-stat">
                        <div class="mini-stat-label">Risk Score</div>
                        <div class="mini-stat-value" style='color:{sc_col};'>{score:.2f}</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-label">Collision Chance</div>
                        <div class="mini-stat-value">{collision_prob:.0%}</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-label">Expected Collisions</div>
                        <div class="mini-stat-value">{exp_v}</div>
                    </div>
                </div>
                <div style='font-size:11px; font-weight:700; color:{B_MUTED}; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;'>
                    Why This Area Is High Risk
                </div>
                <div>{drv_html}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("👆 Click a neighbourhood on the map to see its risk profile.")

        section(f"Top {top_k} Highest-Risk Neighbourhoods")
        top_gdf = gdf[gdf["nbhd_id"].isin(top_k_ids)].sort_values(score_col, ascending=False).reset_index(drop=True)
        top_gdf.index = top_gdf.index + 1

        display_cols = ["area_name", score_col]
        format_dict = {"Risk Score": "{:.3f}", "Expected Collisions": "{:.1f}"}
        col_names = ["Neighbourhood", "Risk Score"]

        if prob_col in top_gdf.columns:
            display_cols.append(prob_col)
            format_dict["Collision Chance"] = "{:.1%}"
            col_names.append("Collision Chance")

        display_cols.append("exp_coll")
        col_names.append("Expected Collisions")

        display_df = top_gdf[display_cols].copy()
        display_df.columns = col_names

        st.dataframe(
            display_df.style.pipe(style_risk_table).format(format_dict),
            width="stretch",
            height=320,
        )
        risk_score_legend()

    with st.expander(f"Why These {top_k} Neighbourhoods Are High Risk", expanded=False):
        top_gdf2 = gdf[gdf["nbhd_id"].isin(top_k_ids)].sort_values(score_col, ascending=False)
        for _, row in top_gdf2.iterrows():
            drv_html = "".join(
                f'<span class="driver-badge driver-{c}">{t}</span>'
                for c, t in why_high_risk(int(row["nbhd_id"]), features)
            )
            st.markdown(
                f'<div class="why-row"><span class="why-name">{row["area_name"]}</span><span class="why-score">Risk Score {row[score_col]:.2f}</span><div class="why-tags">{drv_html}</div></div>',
                unsafe_allow_html=True,
            )

    section("How Risk Is Spread Across All Neighbourhoods Today")
    n_high = int((pred_today[score_col] >= 0.7).sum())
    n_med = int(((pred_today[score_col] >= 0.4) & (pred_today[score_col] < 0.7)).sum())
    n_low = int((pred_today[score_col] < 0.4).sum())

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(
            f"""<div class="metric-card" style="border-color:{C_RED}; border-width:2px;"><div class="metric-label" style="color:{C_RED};">High Risk Zones</div><div class="metric-value" style="color:{C_RED}; font-size:32px;">{n_high}</div></div>""",
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            f"""<div class="metric-card" style="border-color:{C_AMBER}; border-width:2px;"><div class="metric-label" style="color:{C_AMBER};">Medium Risk Zones</div><div class="metric-value" style="color:{C_AMBER}; font-size:32px;">{n_med}</div></div>""",
            unsafe_allow_html=True,
        )
    with d3:
        st.markdown(
            f"""<div class="metric-card" style="border-color:{C_GREEN}; border-width:2px;"><div class="metric-label" style="color:{C_GREEN};">Low Risk Zones</div><div class="metric-value" style="color:{C_GREEN}; font-size:32px;">{n_low}</div></div>""",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRENDS AND SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════
def page_trends(features: pd.DataFrame) -> None:
    st.markdown(f"<h2 style='color:{B_DEEP}; margin-bottom:2px;'>Trends And Seasonality</h2>", unsafe_allow_html=True)
    st.caption("Long-term patterns in collision activity. Useful for planning, budgeting, and understanding what drives risk over time.")
    render_definitions("Trends And Seasonality")

    if features.empty:
        st.warning("Historical data unavailable. Please run the pipeline first.")
        return

    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["dow"] = df["date"].dt.dayofweek
    df["dow_name"] = df["date"].dt.strftime("%a")

    if "ksi_collisions" not in df.columns:
        df["ksi_collisions"] = 0

    daily = df.groupby(["date", "year", "month", "month_name", "dow", "dow_name"], as_index=False).agg(
        collisions=("collisions", "sum"),
        ksi=("ksi_collisions", "sum"),
    )

    tab1, tab2, tab3 = st.tabs(["Collision Volume", "Weather Impact", "Serious Incidents"])

    with tab1:
        section("Collision Volume")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        view = st.selectbox("View by", ["Yearly", "Monthly", "Daily (Day of Week)"], label_visibility="collapsed")

        if view == "Yearly":
            colour_legend(
                [
                    (C_RED, "Above long-term average (worse)"),
                    (C_GREEN, "Below long-term average (better)"),
                    (C_GOLD, "Long-term average line"),
                ]
            )
            annual = daily.groupby("year", as_index=False)["collisions"].sum().rename(columns={"collisions": "total"})
            current_year = pd.Timestamp.now().year
            if daily[daily["year"] == current_year]["month"].nunique() < 6:
                annual = annual[annual["year"] < current_year]

            long_avg = annual["total"].mean()
            bar_cols = [C_RED if v > long_avg else C_GREEN for v in annual["total"]]

            fig = go.Figure(
                go.Bar(
                    x=annual["year"],
                    y=annual["total"],
                    marker_color=bar_cols,
                    hovertemplate="<b>%{x}</b><br>Total Collisions: %{y:,}<extra></extra>",
                )
            )
            fig.add_hline(
                y=long_avg,
                line_dash="dot",
                line_color=C_GOLD,
                line_width=2,
                annotation_text=f"Long-term avg ({long_avg:,.0f})",
                annotation_font_color=C_GOLD,
            )
            fig = chart_layout(fig, height=340)
            fig.update_layout(xaxis=dict(tickmode="linear", dtick=1), yaxis_title="Total Collisions", bargap=0.2)
            st.plotly_chart(fig, width="stretch")

        elif view == "Monthly":
            colour_legend(
                [
                    (C_RED, "Top 3 highest-risk months"),
                    (C_GREEN, "Bottom 3 lowest-risk months"),
                    (C_GREY, "Typical months"),
                    (C_GOLD, "Annual average line"),
                ]
            )
            monthly = daily.groupby(["month", "month_name"], as_index=False)["collisions"].mean()
            monthly["month_name"] = pd.Categorical(monthly["month_name"], categories=month_order, ordered=True)
            monthly = monthly.sort_values("month_name")
            top3_m = set(monthly.nlargest(3, "collisions")["month_name"])
            bot3_m = set(monthly.nsmallest(3, "collisions")["month_name"])
            m_cols = [C_RED if m in top3_m else C_GREEN if m in bot3_m else C_GREY for m in monthly["month_name"]]
            m_avg = monthly["collisions"].mean()

            fig = go.Figure(
                go.Bar(
                    x=monthly["month_name"],
                    y=monthly["collisions"],
                    marker_color=m_cols,
                    hovertemplate="<b>%{x}</b><br>Avg Daily Collisions: %{y:.0f}<extra></extra>",
                )
            )
            fig.add_hline(
                y=m_avg,
                line_dash="dot",
                line_color=C_GOLD,
                line_width=2,
                annotation_text=f"Annual avg ({m_avg:.0f})",
                annotation_font_color=C_GOLD,
            )
            fig = chart_layout(fig, height=320)
            fig.update_layout(yaxis_title="Avg Daily Collisions", bargap=0.15)
            st.plotly_chart(fig, width="stretch")

        else:
            colour_legend(
                [
                    (C_RED, "Top 3 highest-risk days"),
                    (C_GREEN, "Bottom 3 lowest-risk days"),
                    (C_GREY, "Typical days"),
                    (C_GOLD, "Weekly average line"),
                ]
            )
            dow_avg = daily.groupby(["dow", "dow_name"], as_index=False)["collisions"].mean()
            dow_avg["dow_name"] = pd.Categorical(dow_avg["dow_name"], categories=dow_order, ordered=True)
            dow_avg = dow_avg.sort_values("dow_name")
            top3_d = set(dow_avg.nlargest(3, "collisions")["dow_name"])
            bot3_d = set(dow_avg.nsmallest(3, "collisions")["dow_name"])
            d_cols = [C_RED if d in top3_d else C_GREEN if d in bot3_d else C_GREY for d in dow_avg["dow_name"]]
            d_avg = dow_avg["collisions"].mean()

            fig = go.Figure(
                go.Bar(
                    x=dow_avg["dow_name"],
                    y=dow_avg["collisions"],
                    marker_color=d_cols,
                    hovertemplate="<b>%{x}</b><br>Avg Daily Collisions: %{y:.0f}<extra></extra>",
                )
            )
            fig.add_hline(
                y=d_avg,
                line_dash="dot",
                line_color=C_GOLD,
                line_width=2,
                annotation_text=f"Weekly avg ({d_avg:.0f})",
                annotation_font_color=C_GOLD,
            )
            fig = chart_layout(fig, height=300)
            fig.update_layout(yaxis_title="Avg Daily Collisions", bargap=0.2)
            st.plotly_chart(fig, width="stretch")

    with tab2:
        section("How Much Does Weather Increase Collisions?")
        wx_rows = []

        for label, col, val in [
            ("Normal Days", None, None),
            ("Freezing Rain", "freezing_rain", 1),
            ("Heavy Snow", "heavy_snow", 1),
            ("High Wind And Rain", "wind_rain", 1),
        ]:
            if col is None:
                cond = (
                    (df.get("freezing_rain", pd.Series(0, index=df.index)) == 0)
                    & (df.get("heavy_snow", pd.Series(0, index=df.index)) == 0)
                    & (df.get("wind_rain", pd.Series(0, index=df.index)) == 0)
                )
            else:
                cond = df[col] == val if col in df.columns else pd.Series(False, index=df.index)

            subset = daily.merge(df[cond]["date"].drop_duplicates(), on="date", how="inner")
            if not subset.empty:
                wx_rows.append({"Condition": label, "avg": subset["collisions"].mean()})

        if wx_rows:
            wx_df = pd.DataFrame(wx_rows)
            norm = wx_df.iloc[0]["avg"] if len(wx_rows) > 0 else 1

            fig = go.Figure()
            for i, r in wx_df.iterrows():
                is_normal = i == 0
                pct = (r["avg"] - norm) / norm * 100 if norm > 0 else 0
                bar_col = C_BLUE if is_normal else C_RED
                fig.add_trace(
                    go.Bar(
                        x=[r["Condition"]],
                        y=[r["avg"]],
                        marker_color=bar_col,
                        text=["Baseline" if is_normal else f"+{pct:.0f}%"],
                        textposition="outside",
                        textfont=dict(size=12, color=bar_col),
                        hovertemplate=f"<b>{r['Condition']}</b><br>Avg Daily Collisions: {r['avg']:.0f}<extra></extra>",
                    )
                )

            fig = chart_layout(fig, height=340)
            fig.update_layout(
                yaxis_title="Avg Daily Collisions",
                bargap=0.3,
                showlegend=False,
                yaxis=dict(range=[0, wx_df["avg"].max() * 1.2]),
            )
            st.plotly_chart(fig, width="stretch")

    with tab3:
        section("Serious And Fatal Collision Trend")
        ksi_daily = df.groupby(["date", "year", "month"], as_index=False)["ksi_collisions"].sum()
        ksi_monthly = ksi_daily.groupby(["year", "month"], as_index=False)["ksi_collisions"].sum()
        ksi_monthly["period"] = pd.to_datetime(ksi_monthly[["year", "month"]].assign(day=1))
        ksi_monthly = ksi_monthly.sort_values("period")

        if ksi_monthly["ksi_collisions"].sum() > 0:
            long_avg_ksi = ksi_monthly["ksi_collisions"].mean()
            ksi_base = [long_avg_ksi] * len(ksi_monthly)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ksi_monthly["period"],
                    y=ksi_monthly["ksi_collisions"].clip(upper=long_avg_ksi),
                    mode="none",
                    fill=None,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ksi_monthly["period"],
                    y=ksi_base,
                    mode="none",
                    fill="tonexty",
                    fillcolor="rgba(30,132,73,0.10)",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ksi_monthly["period"],
                    y=ksi_base,
                    mode="none",
                    fill=None,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ksi_monthly["period"],
                    y=ksi_monthly["ksi_collisions"].clip(lower=long_avg_ksi),
                    mode="none",
                    fill="tonexty",
                    fillcolor="rgba(192,57,43,0.10)",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ksi_monthly["period"],
                    y=ksi_base,
                    mode="lines",
                    name=f"Long-term avg ({long_avg_ksi:.1f})",
                    line=dict(color=C_GOLD, width=1.8, dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=ksi_monthly["period"],
                    y=ksi_monthly["ksi_collisions"],
                    mode="lines",
                    name="Serious Incidents Per Month",
                    line=dict(color=C_TEAL, width=2),
                )
            )
            fig = chart_layout(fig, height=340)
            fig.update_layout(yaxis_title="Serious Incidents Per Month")
            st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def page_model_performance(horizon: int) -> None:
    st.markdown(f"<h2 style='color:{B_DEEP}; margin-bottom:2px;'>Model Performance</h2>", unsafe_allow_html=True)
    st.caption("How accurate are the forecasts? This page shows how the models performed on data they had not seen before.")
    render_definitions("Model Performance")

    metrics = api_get_metrics()
    surge_metrics = metrics.get("surge_metrics", [])
    nbhd_metrics = metrics.get("nbhd_metrics", [])

    section("Citywide Surge Forecast Accuracy")
    if surge_metrics:
        surge_h = next((m for m in surge_metrics if f"t{horizon}" in m.get("model", "")), surge_metrics[0])

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            metric_card("Accuracy Score", f"{surge_h.get('roc_auc', 0):.2f}", sub="Above 0.70 is strong")
        with s2:
            metric_card("Forecast Quality", f"{surge_h.get('pr_auc', 0):.2f}", sub="Handles rare surge days")
        with s3:
            metric_card("Calibration", f"{surge_h.get('brier', 0):.3f}", sub="Lower is better")
        with s4:
            metric_card(
                "Surge Days Tested",
                f"{surge_h.get('pos_rate', 0):.1%}",
                sub=f"Out of {surge_h.get('n_test', 0)} total days",
            )

    section("Neighbourhood Forecast Accuracy")
    if nbhd_metrics:
        model_candidates = [
            f"collision_prob_t{horizon}",
            f"proba_collision_t{horizon}",
            f"risk_score_t{horizon}",
        ]

        col_m = next(
            (m for m in nbhd_metrics if m.get("model") in model_candidates),
            None
        )

        tabs = st.tabs(["Collision Forecast"])
        for tab, m in zip(tabs, [col_m]):
            if m is None:
                continue

            with tab:
                roc_val = m.get("roc_auc", float("nan"))
                pr_val = m.get("pr_auc", 0.0)
                brier = m.get("brier", 0.0)

                roc_str = f"{roc_val:.2f}" if not (isinstance(roc_val, float) and np.isnan(roc_val)) else "N/A"
                pr_str = f"{pr_val:.2f}" if pr_val > 0 else "N/A"

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    metric_card("Accuracy Score", roc_str)
                with mc2:
                    metric_card("Forecast Quality", pr_str)
                with mc3:
                    metric_card("Calibration", f"{brier:.3f}")

                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.markdown("**How accurately does the model rank the highest-risk neighbourhoods?**")

                k_vals = [5, 10, 15, 20]
                p_vals = [m.get(f"precision_at_{k}", 0) for k in k_vals]
                r_vals = [m.get(f"recall_at_{k}", 0) for k in k_vals]
                baseline = m.get("baseline_pos_rate", 0)

                k_fig = go.Figure()
                k_fig.add_trace(go.Bar(x=[f"Top {k}" for k in k_vals], y=p_vals, name="Accuracy", marker_color=C_BLUE))
                k_fig.add_trace(go.Bar(x=[f"Top {k}" for k in k_vals], y=r_vals, name="Coverage", marker_color=C_GOLD))
                k_fig.add_hline(
                    y=baseline,
                    line_dash="dash",
                    line_color=C_SLATE,
                    annotation_text=f"Random Chance ({baseline:.0%})",
                    annotation_font_color=C_SLATE,
                )
                k_fig.update_layout(
                    barmode="group",
                    height=280,
                    plot_bgcolor=B_CARD,
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(k_fig, width="stretch")

    section("Top Factors Driving The Forecasts")
    if COLLISION_FI_PATH.exists():
        fi_col = pd.read_csv(COLLISION_FI_PATH).head(15)

        fi_tabs = st.tabs(["Collision Forecast Factors"])
        for fi_tab, fi_df, colour in zip(fi_tabs, [fi_col], [C_BLUE]):
            with fi_tab:
                fi_df = fi_df[fi_df["importance"] > 0].copy()
                fi_df["label"] = fi_df["feature"].map(FRIENDLY).fillna(fi_df["feature"])
                fi_df = fi_df.sort_values("importance")

                fig = go.Figure(
                    go.Bar(
                        x=fi_df["importance"],
                        y=fi_df["label"],
                        orientation="h",
                        marker_color=colour,
                    )
                )
                fig.update_layout(
                    height=420,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor=B_CARD,
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, width="stretch")
    else:
        st.info("Factor analysis files not yet available.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    controls = render_sidebar()
    horizon = controls["horizon"]
    top_k = controls["top_k"]
    page = controls["page"]

    features = load_features()
    nbhd = load_nbhd()
    dim = load_dim()

    if page == "City Overview":
        page_city_overview(features, horizon, top_k)
    elif page == "Risk Map":
        page_risk_map(dim, features, nbhd, horizon, top_k)
    elif page == "Trends And Seasonality":
        page_trends(features)
    elif page == "Model Performance":
        page_model_performance(horizon)


if __name__ == "__main__":
    main()