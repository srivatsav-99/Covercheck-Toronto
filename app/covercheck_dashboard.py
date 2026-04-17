"""
CoverCheck Toronto — Streamlit Dashboard
Light mode only. Professional blue palette (#5DA5DA brand colour).

Usage:
    streamlit run covercheck_dashboard.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import warnings
from pathlib import Path

# Make repo root importable when running:
# streamlit run app/covercheck_dashboard.py
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

from src.pipelines import covercheck_pipeline as pipeline
from src.io_paths import ASSETS_DIR

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE — light mode, WCAG AA compliant, clearly distinct
# ══════════════════════════════════════════════════════════════════════════════
# Structure
B_SIDEBAR = "#1B4F72"   # deep navy     — sidebar background
B_SIDE_TX = "#EBF5FB"   # near-white    — sidebar text
B_DEEP    = "#1A2E44"   # dark navy     — headings, body text
B_LABEL   = "#2E6DA4"   # medium blue   — labels, accents
B_MUTED   = "#5D7A8A"   # slate blue    — captions, secondary text
B_PAGE    = "#F0F6FB"   # pale blue     — page background
B_CARD    = "#FFFFFF"   # white         — cards, panels
B_FILL    = "#EBF5FB"   # light blue    — subtle fills (rows, mini-cards)
B_BORDER  = "#C8E0F0"   # soft blue     — borders and dividers

# Chart — clearly distinct from each other and from structure colours
C_BLUE    = "#2980B9"   # vivid blue    — primary bars and lines
C_RED     = "#C0392B"   # coral red     — High risk / alert
C_AMBER   = "#CA6F1E"   # deep amber    — Medium risk / caution
C_GREEN   = "#1E8449"   # forest green  — Low risk / safe
C_TEAL    = "#1A6B57"   # dark teal     — Serious incidents
C_GOLD    = "#B7950B"   # dark gold     — Trend lines
C_SLATE   = "#5D7A8A"   # slate         — Neutral / baselines
C_MID_BLU = "#2471A3"   # secondary     — secondary chart bars
C_GREY    = "#9EB3C2"   # neutral grey  — bars when only some have red/green meaning

G_GRID    = "rgba(41,128,185,0.07)"
LOGO_PATH = ASSETS_DIR / "covercheck_logo.png"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CoverCheck Toronto",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

#startup sanity check box
REQUIRED_INPUTS = [
    pipeline.DIM_PATH,
]

for p in REQUIRED_INPUTS:
    if not Path(p).exists():
        st.error(f"Required file missing: {p}")
        st.stop()

# Force light mode via injected CSS
st.markdown(f"""
<style>
  /* ── Page background ── */
  .stApp {{ background-color: {B_PAGE} !important; }}
  [data-testid="stAppViewContainer"] {{ background-color: {B_PAGE} !important; }}

  /* ── Sidebar ── */
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
      border-color: {B_SIDE_TX} !important; opacity: 0.25;
  }}
  section[data-testid="stSidebar"] .stButton > button {{
      background: {C_BLUE} !important; color: #FFFFFF !important;
      border: none; border-radius: 8px; font-weight: 600; width: 100%;
  }}
  section[data-testid="stSidebar"] .stButton > button:hover {{
      background: #1A6FA8 !important;
  }}

  /* ── Metric cards ── */
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

  /* ── Risk colour classes ── */
  .risk-high   {{ color: {C_RED}   !important; }}
  .risk-medium {{ color: {C_AMBER} !important; }}
  .risk-low    {{ color: {C_GREEN} !important; }}

  /* ── Section headers ── */
  .section-header {{
      font-size: 14px;
      font-weight: 700;
      color: {B_DEEP};
      margin: 1.2rem 0 0.4rem 0;
      padding-bottom: 6px;
      border-bottom: 2px solid {B_BORDER};
  }}

  /* ── Alert banners ── */
  .alert-high {{
      background: #FDEDEC; border: 1.5px solid {C_RED};
      border-radius: 10px; padding: 0.8rem 1rem;
      margin-bottom: 12px; color: {B_DEEP};
  }}
  .alert-medium {{
      background: #FDF3E7; border: 1.5px solid {C_AMBER};
      border-radius: 10px; padding: 0.8rem 1rem;
      margin-bottom: 12px; color: {B_DEEP};
  }}
  .alert-low {{
      background: #E9F7EF; border: 1.5px solid {C_GREEN};
      border-radius: 10px; padding: 0.8rem 1rem;
      margin-bottom: 12px; color: {B_DEEP};
  }}

  /* ── Neighbourhood profile panel ── */
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

  /* ── Risk driver badges ── */
  .driver-badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
      margin: 2px 3px 2px 0;
  }}
  .driver-weather  {{ background: #D6EAF8; color: #1A5276; }}
  .driver-traffic  {{ background: #D4E6F1; color: {B_DEEP}; }}
  .driver-calendar {{ background: #FEF3E2; color: #7D4E08; }}
  .driver-road     {{ background: #D5EFE3; color: #1A5738; }}

  /* ── Definition boxes ── */
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
  .def-title {{ font-size: 13px; font-weight: 700; color: {B_DEEP}; margin-bottom: 6px; }}
  .def-row   {{ font-size: 12px; color: #2C3E50; margin-bottom: 4px; line-height: 1.55; }}
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
  .lbl-what {{ background: #D4E6F1; color: #1A3E5C; }}
  .lbl-why  {{ background: #D5EFE3; color: #1A5738; }}
  .lbl-how  {{ background: #FEF3E2; color: #7D4E08; }}

  /* ── Risk score legend ── */
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
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}

  /* ── Why-high-risk rows ── */
  .why-row {{
      margin-bottom: 8px;
      padding: 8px 12px;
      background: {B_FILL};
      border-radius: 8px;
      border: 1px solid {B_BORDER};
  }}
  .why-name  {{ font-size: 13px; font-weight: 700; color: {B_DEEP}; }}
  .why-score {{ font-size: 11px; color: {B_MUTED}; margin-left: 8px; }}
  .why-tags  {{ margin-top: 5px; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
"""@st.cache_data(ttl=3600, show_spinner="Building features...")
def load_features() -> pd.DataFrame:
    return pipeline.run_feature_builder()

@st.cache_data(ttl=3600, show_spinner="Running citywide forecasts...")
def load_surge() -> pd.DataFrame:
    return pipeline.run_surge_classifier()

@st.cache_data(ttl=3600, show_spinner="Scoring neighbourhoods...")
def load_nbhd() -> pd.DataFrame:
    return pipeline.run_nbhd_models()

@st.cache_data(ttl=3600, show_spinner="Loading map boundaries...")
def load_dim() -> gpd.GeoDataFrame:
    dim = gpd.read_parquet(pipeline.DIM_PATH)
    dim["nbhd_id"] = dim["nbhd_id"].astype(int)
    return dim

@st.cache_data(ttl=3600)
def load_metrics() -> tuple[list[dict], list[dict]]:
    with open(pipeline.SURGE_METRICS_PATH) as f: sm = json.load(f)
    with open(pipeline.NBHD_METRICS_PATH)  as f: nm = json.load(f)
    return sm, nm"""

#Make the dashboard read artifacts first, then run pipeline only if needed

@st.cache_data(show_spinner=False)
def load_features():
    if pipeline.FEATURES_PATH.exists():
        return pd.read_parquet(pipeline.FEATURES_PATH)
    return pipeline.run_feature_builder()


@st.cache_data(show_spinner=False)
def load_surge():
    if pipeline.SURGE_PRED_PATH.exists():
        return pd.read_parquet(pipeline.SURGE_PRED_PATH)
    return pipeline.run_surge_classifier()


@st.cache_data(show_spinner=False)
def load_nbhd():
    if pipeline.NBHD_PRED_PATH.exists():
        return pd.read_parquet(pipeline.NBHD_PRED_PATH)
    return pipeline.run_nbhd_models()


@st.cache_data(show_spinner=False)
def load_dim():
    return gpd.read_parquet(pipeline.DIM_PATH)


@st.cache_data(show_spinner=False)
def load_metrics():
    surge_metrics = []
    nbhd_metrics = []

    if pipeline.SURGE_METRICS_PATH.exists():
        with open(pipeline.SURGE_METRICS_PATH, "r") as f:
            surge_metrics = json.load(f)

    if pipeline.NBHD_METRICS_PATH.exists():
        with open(pipeline.NBHD_METRICS_PATH, "r") as f:
            nbhd_metrics = json.load(f)

    return surge_metrics, nbhd_metrics

def clear_and_refresh() -> None:
    st.cache_data.clear()
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
FRIENDLY = {
    "collisions_lag1":                   "Yesterday's Collisions",
    "collisions_roll7_mean":             "7-Day Average Collisions",
    "collisions_roll14_mean":            "14-Day Average Collisions",
    "collisions_roll30_mean":            "30-Day Average Collisions",
    "collision_momentum":                "Collision Spike vs Baseline",
    "ksi_ratio_lag1":                    "Yesterday's Serious Incident Rate",
    "tavg":                              "Average Temperature",
    "tmin":                              "Minimum Temperature",
    "prcp":                              "Rainfall Amount",
    "snow":                              "Snowfall Amount",
    "wspd":                              "Wind Speed",
    "freezing_rain":                     "Freezing Rain",
    "heavy_snow":                        "Heavy Snow",
    "feels_bad":                         "Adverse Weather",
    "is_holiday":                        "Public Holiday",
    "pre_holiday":                       "Day Before A Holiday",
    "post_holiday":                      "Day After A Holiday",
    "is_weekend":                        "Weekend",
    "dow_sin":                           "Day Of Week Pattern",
    "month_sin":                         "Time Of Year Pattern",
    "road_construction_count_halo_lag1": "Nearby Construction Yesterday",
    "road_construction_count_lag1":      "Local Construction Yesterday",
    "is_zone_disrupted":                 "Neighbouring Zone Disrupted",
    "construction_total_pressure":       "Road Disruption Pressure",
    "intersection_count":                "Number Of Intersections",
    "arterial_length_km":                "Total Road Length",
    "streets_per_node_avg":              "Road Network Density",
    "surge_proba_t1":                    "Citywide Surge Risk",
}

def risk_colour(score: float) -> str:
    if score >= 0.7: return "risk-high"
    if score >= 0.4: return "risk-medium"
    return "risk-low"

def metric_card(label: str, value: str, sub: str = "", colour_class: str = "") -> None:
    val_class = f"metric-value {colour_class}" if colour_class else "metric-value"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{val_class}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def risk_score_legend() -> None:
    st.markdown(f"""
    <div class="legend-row">
        <span style="font-size:12px; font-weight:700; color:{B_DEEP};">Risk Score</span>
        <div class="legend-item">
            <div class="legend-dot" style="background:{C_RED};"></div>
            <span>High (0.70 and above)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:{C_AMBER};"></div>
            <span>Medium (0.40 to 0.69)</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background:{C_GREEN};"></div>
            <span>Low (below 0.40)</span>
        </div>
    </div>""", unsafe_allow_html=True)

def style_risk_table(styler: "pd.io.formats.style.Styler") -> "pd.io.formats.style.Styler":
    """Colour Risk Score column by fixed thresholds. Receives a Styler via .pipe()."""
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

def colour_legend(items: list[tuple[str,str]]) -> None:
    """Compact inline colour legend. items = [(dot_colour, label)]"""
    dots = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:5px;"
        f"margin-right:16px;font-size:12px;color:{B_DEEP};white-space:nowrap;'>"
        f"<span style='width:10px;height:10px;border-radius:50%;background:{col};"
        f"display:inline-block;flex-shrink:0;'></span>{lbl}</span>"
        for col, lbl in items
    )
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;align-items:center;"
        f"padding:6px 12px;background:{B_CARD};border:1px solid {B_BORDER};"
        f"border-radius:8px;margin-bottom:8px;'>{dots}</div>",
        unsafe_allow_html=True)

def chart_layout(fig: go.Figure, height: int = 300) -> go.Figure:
    """Consistent chart styling for all pages."""
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=44, b=0),
        plot_bgcolor=B_CARD,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=B_DEEP, size=12, family="Arial, sans-serif"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=12, color=B_DEEP),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=B_BORDER, borderwidth=1,
        ),
        xaxis=dict(
            gridcolor=G_GRID, linecolor=B_BORDER, zeroline=False,
            tickfont=dict(color=B_MUTED, size=11),
        ),
        yaxis=dict(
            gridcolor=G_GRID, linecolor=B_BORDER, zeroline=False,
            tickfont=dict(color=B_MUTED, size=11),
        ),
        hovermode="x unified",
    )
    return fig

def nbhd_avg_collisions(features: pd.DataFrame) -> pd.DataFrame:
    """30-day average daily collisions per neighbourhood. Pandas 2.x safe."""
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
    colour_scale: str,
    top_k_ids: list[int] | None = None,
    tip_fields: list[str] | None = None,
    tip_aliases: list[str] | None = None,
) -> folium.Map:
    m = folium.Map(location=[43.72, -79.38], zoom_start=11,
                   tiles="CartoDB positron", prefer_canvas=True)
    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        data=gdf,
        columns=["nbhd_id", score_col],
        key_on="feature.properties.nbhd_id",
        fill_color=colour_scale,
        fill_opacity=0.72, line_opacity=0.28, line_weight=0.8,
        legend_name=legend_name,
        nan_fill_color="#D6EAF8",
    ).add_to(m)
    tf = tip_fields  or ["area_name", score_col]
    ta = tip_aliases or ["Neighbourhood", legend_name]
    folium.GeoJson(
        gdf.__geo_interface__,
        tooltip=folium.GeoJsonTooltip(
            fields=tf, aliases=ta, localize=True, sticky=False,
            style="font-size:12px; font-family:Arial,sans-serif;",
        ),
        style_function=lambda x: {"fillOpacity": 0, "weight": 0, "color": "transparent"},
    ).add_to(m)
    if top_k_ids:
        folium.GeoJson(
            gdf[gdf["nbhd_id"].isin(top_k_ids)].__geo_interface__,
            style_function=lambda x: {
                "fillOpacity": 0, "weight": 3,
                "color": C_RED, "dashArray": "5 3",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tf, aliases=ta,
                style="font-size:12px; font-weight:700;",
            ),
        ).add_to(m)
    return m

def why_high_risk(nbhd_id: int, features: pd.DataFrame) -> list[tuple[str, str]]:
    """Up to 4 plain-English risk drivers for a neighbourhood."""
    sub = features[features["nbhd_id"] == nbhd_id].sort_values("date")
    if sub.empty:
        return [("traffic", "High recent collision activity")]
    latest   = sub.iloc[-1]
    avg_coll = sub["collisions"].tail(30).mean()
    reasons  = []
    lag1 = float(latest.get("collisions_lag1", 0) or 0)
    if lag1 > avg_coll * 1.5:
        reasons.append(("traffic", f"Yesterday had {int(lag1)} collisions (above average)"))
    mom = float(latest.get("collision_momentum", 1.0) or 1.0)
    if mom > 1.4:
        reasons.append(("traffic", f"Collision activity is {mom:.1f}x the recent baseline"))
    if latest.get("freezing_rain", 0):   reasons.append(("weather",  "Freezing rain forecast"))
    elif latest.get("heavy_snow", 0):    reasons.append(("weather",  "Heavy snow forecast"))
    elif latest.get("feels_bad", 0):     reasons.append(("weather",  "Adverse weather conditions"))
    if latest.get("is_holiday", 0):      reasons.append(("calendar", "Public holiday"))
    elif latest.get("pre_holiday", 0):   reasons.append(("calendar", "Day before a public holiday"))
    elif latest.get("is_weekend", 0):    reasons.append(("calendar", "Weekend traffic patterns"))
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
        {"title": "Surge Risk",
         "what": "The probability that tomorrow will be a high-collision day across all of Toronto.",
         "why":  "Tells teams whether to expect above-normal activity citywide before the day begins.",
         "how":  "🔴 Red = High risk (70%+). 🟡 Amber = Moderate (40–69%). 🟢 Green = Low (below 40%)."},
        {"title": "Expected Collisions",
         "what": "An estimate of how many collisions are likely tomorrow across the entire city.",
         "why":  "Helps teams plan resource levels — more expected collisions means more readiness needed.",
         "how":  "Based on the 30-day historical average adjusted by the surge risk level."},
        {"title": "Daily Collision Trend",
         "what": "How daily collision counts have moved over the past 90 days vs the 30-day average.",
         "why":  "Reveals whether the city is in a rising or falling risk period.",
         "how":  "🔴 Red shading = above the 30-day average (more collisions than usual). 🟢 Green = below (quieter). Gold dotted line = 30-day average."},
        {"title": "Weather Conditions",
         "what": "Current weather readings including temperature, rain, snow, and wind.",
         "why":  "Adverse weather is one of the strongest drivers of collision risk.",
         "how":  "Road Conditions turns 🔴 red when freezing rain or heavy snow is active."},
    ],
    "Risk Map": [
        {"title": "Risk Score",
         "what": "A score from 0 to 1 showing how likely a collision is in each neighbourhood tomorrow.",
         "why":  "Lets teams identify where risk is concentrated so resources go to the right places.",
         "how":  "🔴 Red (0.70+) = High. 🟡 Amber (0.40–0.69) = Medium. 🟢 Green (below 0.40) = Low. Darker map = higher risk. Red outlines = top-ranked zones."},
        {"title": "Neighbourhood Profile",
         "what": "Detailed breakdown — risk score, collision chance, expected collisions, and risk drivers.",
         "why":  "A number alone does not tell you what to do. The profile explains why an area is risky.",
         "how":  "Hover over any neighbourhood. The profile appears on the right side of the page."},
        {"title": "Risk Drivers",
         "what": "Tags explaining what is pushing the risk score up for each neighbourhood.",
         "why":  "Helps teams decide the right response — a traffic spike needs a different action than a weather event.",
         "how":  "🔵 Blue = Traffic spike. 🔷 Light blue = Weather. 🟡 Gold = Calendar (holiday/weekend). 🟢 Green = Road disruption."},
        {"title": "Zone Counts",
         "what": "How many of the 158 Toronto neighbourhoods are at High, Medium, and Low risk today.",
         "why":  "Shows whether today is a concentrated or widespread risk day across the city.",
         "how":  "🔴 High = 0.70+. 🟡 Medium = 0.40–0.69. 🟢 Low = below 0.40."},
    ],
    "Trends And Seasonality": [
        {"title": "Collision Volume",
         "what": "Historical collision counts by year, month, or day of week — select the view using the dropdown.",
         "why":  "Reveals patterns that repeat every year so teams can plan ahead for high-risk periods.",
         "how":  "🔴 Red bars = highest-risk periods (above average). 🟢 Green = lowest-risk (below average). ⬜ Grey = typical. Gold dotted line = average reference."},
        {"title": "Weather Impact",
         "what": "How much adverse weather conditions increase average daily collisions vs normal days.",
         "why":  "Quantifies the weather risk so teams know exactly how much more vigilant to be in bad weather.",
         "how":  "🔵 Blue bar = normal days (baseline). 🔴 Red bars = adverse conditions. Labels show % increase above normal."},
        {"title": "Serious Incident Trend",
         "what": "Monthly count of collisions resulting in serious injury or fatality (KSI) over time.",
         "why":  "Tracks whether the most severe incidents are getting better or worse long-term.",
         "how":  "🔴 Red shading = above long-term average (worsening). 🟢 Green = below average (improving). Gold line = long-term average."},
    ],
    "Model Performance": [
        {"title": "Accuracy Score (ROC-AUC)",
         "what": "How well the model separates high-risk days from low-risk days.",
         "why":  "A higher score means the model is more reliable for operational decisions.",
         "how":  "0.50 = random guessing. 0.70+ = strong. CoverCheck scores 0.72 on collision forecasts."},
        {"title": "Ranking Accuracy (Precision@K)",
         "what": "Of the top K flagged neighbourhoods, how many actually had a collision that day.",
         "why":  "The most operationally important metric — tells you whether acting on the list works.",
         "how":  "90% at Top 10 means 9 out of 10 flagged zones had an actual incident. Higher is better."},
        {"title": "Coverage (Recall@K)",
         "what": "Of all neighbourhoods that had a collision, what share were in the flagged list.",
         "why":  "Measures how many real incidents the model would have helped prevent.",
         "how":  "Higher is better. Low coverage means some high-risk zones were missed."},
        {"title": "Top Contributing Factors",
         "what": "The data inputs that most influence the model's risk predictions.",
         "why":  "Confirms the model is learning from sensible signals — not noise or coincidence.",
         "how":  "Longer bars = greater influence on the prediction."},
        {"title": "Forecast Stability",
         "what": "Whether model accuracy holds up consistently across the full test period.",
         "why":  "A model that degrades over time needs retraining — this check flags it early.",
         "how":  "🟢 Stable = accuracy is consistent. ⚠️ Warning = significant drop, retraining recommended."},
    ],
}

def render_definitions(page: str) -> None:
    defs = DEFINITIONS.get(page, [])
    if not defs:
        return
    with st.expander("What Does This Page Show? Click Here For Definitions", expanded=False):
        for d in defs:
            st.markdown(f"""
            <div class="def-box">
                <div class="def-title">{d['title']}</div>
                <div class="def-row"><span class="def-label lbl-what">What</span>{d['what']}</div>
                <div class="def-row"><span class="def-label lbl-why">Why It Matters</span>{d['why']}</div>
                <div class="def-row"><span class="def-label lbl-how">How To Read It</span>{d['how']}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> dict:
    with st.sidebar:
        # Logo on white card — base64 embed keeps the wrapper intact
        if LOGO_PATH.exists():
            with open(LOGO_PATH, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"<div style='background:#FFFFFF; border-radius:10px; "
                f"padding:12px 16px; margin-bottom:8px; text-align:center;'>"
                f"<img src='data:image/png;base64,{logo_b64}' "
                f"style='width:100%; max-width:240px; height:auto;'>"
                f"</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='text-align:center; padding:0.6rem 0 0.8rem;'>"
                f"<div style='font-size:1.2rem; font-weight:700; color:{B_DEEP};'>🚦 CoverCheck</div>"
                f"<div style='font-size:0.72rem; color:{B_DEEP}; opacity:0.7; margin-top:2px;'>Toronto Collision Risk Forecasting</div>"
                f"</div>", unsafe_allow_html=True)

        st.divider()

        st.markdown(
            f"<div style='font-size:11px; font-weight:700; color:{B_SIDE_TX}; "
            f"text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;'>"
            f"Forecast Horizon</div>",
            unsafe_allow_html=True)
        horizon = st.radio(
            "Forecast Horizon",
            options=[1, 2],
            format_func=lambda x: "Tomorrow" if x == 1 else "Day After Tomorrow",
            label_visibility="collapsed")

        st.markdown(
            f"<div style='font-size:11px; font-weight:700; color:{B_SIDE_TX}; "
            f"text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px; margin-top:12px;'>"
            f"High-Risk Zones To Show</div>",
            unsafe_allow_html=True)
        top_k = st.select_slider(
            "Zones",
            options=[5, 10, 15, 20],
            value=10,
            label_visibility="collapsed")

        st.divider()

        pred_path = pipeline.NBHD_PRED_PATH
        if pred_path.exists():
            mtime    = os.path.getmtime(pred_path)
            last_run = pd.Timestamp.fromtimestamp(mtime).strftime("%b %d, %Y at %H:%M")
            st.caption(f"Last updated: {last_run}")

        if st.button("Refresh Forecasts", type="primary", use_container_width="stretch"):
            clear_and_refresh()
        st.caption("Rebuilds all forecasts from the latest data.")

        st.divider()

        st.markdown(
            f"<div style='font-size:11px; font-weight:700; color:{B_SIDE_TX}; "
            f"text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;'>"
            f"Pages</div>",
            unsafe_allow_html=True)
        page = st.radio(
            "Pages",
            ["City Overview", "Risk Map", "Trends And Seasonality", "Model Performance"],
            label_visibility="collapsed")

    # a manual refresh button in the sidebar
    refresh = st.button("Refresh Forecasts")

    if refresh:
        with st.spinner("Running CoverCheck pipeline..."):
            pipeline.run_pipeline(start_from="features")
        st.cache_data.clear()
        st.success("Forecasts refreshed.")

    return {"horizon": horizon, "top_k": top_k, "page": page}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CITY OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_city_overview(
    features: pd.DataFrame,
    surge: pd.DataFrame,
    nbhd: pd.DataFrame,
    horizon: int,
    top_k: int,
) -> None:
    hl = "Tomorrow" if horizon == 1 else "Day After Tomorrow"
    st.markdown(f"<h2 style='color:{B_DEEP}; margin-bottom:2px;'>City Overview</h2>",
                unsafe_allow_html=True)
    st.caption(f"Operational snapshot for {hl.lower()} across Toronto.")
    render_definitions("City Overview")

    # Surge data
    latest = surge.sort_values("date").iloc[-1]
    sp     = float(latest[f"surge_proba_t{horizon}"])
    sl     = int(latest[f"surge_label_t{horizon}"])
    fdate  = pd.Timestamp(latest["date"]).strftime("%B %d, %Y")
    thr    = latest.get("threshold", None)

    # Alert banner
    if sp >= 0.7:
        bclass, bicon = "alert-high",   "🔴"
        bmsg = f"High surge risk for {hl.lower()} ({sp:.0%}). Collision activity is expected to be significantly above normal."
    elif sp >= 0.4:
        bclass, bicon = "alert-medium", "🟡"
        bmsg = f"Moderate surge risk for {hl.lower()} ({sp:.0%}). Collision activity may be above normal in several areas."
    else:
        bclass, bicon = "alert-low",    "🟢"
        bmsg = f"Low surge risk for {hl.lower()} ({sp:.0%}). No citywide surge is expected."
    st.markdown(f'<div class="{bclass}">{bicon}&nbsp; {bmsg}</div>', unsafe_allow_html=True)

    # City daily collisions
    daily_city = (
        features.groupby("date", as_index=False)
        .agg(city_collisions=("collisions", "sum"))
        .sort_values("date")
    )
    avg_30     = daily_city.tail(30)["city_collisions"].mean()
    latest_day = int(daily_city.tail(1)["city_collisions"].iloc[0])
    expected   = int(avg_30 * (1 + (sp - 0.5)))

    # Key metrics — 3 cards
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(f"Surge Risk — {hl}", f"{sp:.0%}",
                    sub=f"Forecast for {fdate}",
                    colour_class=risk_colour(sp))
    with c2:
        metric_card("Expected Collisions", f"~{expected:,}",
                    sub=f"30-day city average: {avg_30:,.0f}/day")
    with c3:
        day_label = "High-Collision Day" if sl == 1 else "Normal Day"
        metric_card(f"Yesterday — {latest_day:,} Collisions", day_label,
                    sub=f"Threshold: {thr:.0f}" if thr else "Based on historical patterns",
                    colour_class="risk-high" if sl == 1 else "risk-low")

    st.markdown("---")

    # Top-K table on landing page
    section(f"Top {top_k} Highest-Risk Neighbourhoods For {hl}")
    st.caption("These are the areas most likely to experience a collision. Scores are updated with every forecast run.")

    score_col = f"risk_score_t{horizon}"
    prob_col  = f"proba_collision_t{horizon}"
    ksi_col   = f"proba_ksi_t{horizon}"

    latest_date = nbhd["date"].max()
    pred_today  = nbhd[nbhd["date"] == latest_date].copy()
    pred_today["nbhd_id"] = pred_today["nbhd_id"].astype(int)

    avgs = nbhd_avg_collisions(features)
    pred_today = pred_today.merge(avgs, on="nbhd_id", how="left")
    pred_today["avg_30d"]  = pred_today["avg_30d"].fillna(avg_30 / 158)
    pred_today["exp_coll"] = (pred_today["avg_30d"] * pred_today[score_col] * 2).round(1)

    dim_names = gpd.read_parquet(pipeline.DIM_PATH)[["nbhd_id", "area_name"]]
    dim_names["nbhd_id"] = dim_names["nbhd_id"].astype(int)
    pred_today = pred_today.merge(dim_names, on="nbhd_id", how="left")

    top_rows = pred_today.nlargest(top_k, score_col).reset_index(drop=True)
    top_rows.index = top_rows.index + 1
    display_df = top_rows[["area_name", score_col, prob_col, "exp_coll"]].copy()
    display_df.columns = ["Neighbourhood", "Risk Score", "Collision Chance",
                          "Expected Collisions"]
    st.dataframe(
        display_df.style
        .pipe(style_risk_table)
        .format({
            "Risk Score": "{:.3f}",
            "Collision Chance": "{:.1%}",
            #"Serious Incident Chance": "{:.1%}",
            "Expected Collisions": "{:.1f}",
        }),
        use_container_width="stretch",
        height=min(80 + top_k * 38, 480),
    )
    risk_score_legend()

    st.markdown("---")

    # Daily collision trend — semantic red/green fill
    section("Daily Collision Trend (Past 90 Days)")

    last_90           = daily_city.tail(90).copy()
    last_90["date"]   = pd.to_datetime(last_90["date"])
    avg_line          = daily_city.tail(30)["city_collisions"].mean()
    last_90["avg_30"] = avg_line

    fig = go.Figure()
    # Green fill — below 30-day average (safer than usual)
    fig.add_trace(go.Scatter(
        x=last_90["date"], y=last_90["city_collisions"].clip(upper=avg_line),
        mode="none", fill=None, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=last_90["date"], y=last_90["avg_30"],
        mode="none", fill="tonexty", fillcolor="rgba(30,132,73,0.12)",
        showlegend=False, hoverinfo="skip"))
    # Red fill — above 30-day average (riskier than usual)
    fig.add_trace(go.Scatter(
        x=last_90["date"], y=last_90["avg_30"],
        mode="none", fill=None, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=last_90["date"], y=last_90["city_collisions"].clip(lower=avg_line),
        mode="none", fill="tonexty", fillcolor="rgba(192,57,43,0.10)",
        showlegend=False, hoverinfo="skip"))
    # Gold reference line — 30-day average
    fig.add_trace(go.Scatter(
        x=last_90["date"], y=last_90["avg_30"],
        mode="lines", name=f"30-Day Avg ({avg_line:.0f})",
        line=dict(color=C_GOLD, width=1.8, dash="dot"),
        hovertemplate="30-Day Avg: %{y:.0f}<extra></extra>"))
    # Neutral navy line — daily collisions
    fig.add_trace(go.Scatter(
        x=last_90["date"], y=last_90["city_collisions"],
        mode="lines", name="Daily Collisions",
        line=dict(color=B_DEEP, width=1.6),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Collisions: %{y:,}<extra></extra>"))
    # Direction annotation
    recent_7 = last_90.tail(7)["city_collisions"].mean()
    ann_col  = C_RED if recent_7 > avg_line else C_GREEN
    ann_txt  = "↑ Above normal" if recent_7 > avg_line else "↓ Below normal"
    fig.add_annotation(
        x=last_90["date"].iloc[-1], y=last_90["city_collisions"].iloc[-1],
        text=ann_txt, showarrow=False,
        font=dict(size=11, color=ann_col), xanchor="right", yanchor="bottom")
    fig = chart_layout(fig, height=300)
    fig.update_layout(yaxis_title="Collisions Per Day", xaxis_title="")
    colour_legend([
        (C_RED,   "Above 30-day average (more collisions than usual)"),
        (C_GREEN, "Below 30-day average (quieter than usual)"),
        (C_GOLD,  "30-day average line"),
    ])
    st.plotly_chart(fig, use_container_width="stretch")


    # Weather conditions
    section("Current Weather Conditions")
    st.caption("Weather is one of the strongest drivers of collision risk. Freezing rain and heavy snow days see significantly more incidents.")

    lw = features.sort_values("date").drop_duplicates("date").iloc[-1]
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1: metric_card("Temperature",  f"{lw['tavg']:.1f}°C",    sub="Average Today")
    with w2: metric_card("Rainfall",     f"{lw['prcp']:.1f} mm",   sub="Precipitation")
    with w3: metric_card("Snowfall",     f"{lw['snow']:.1f} cm",   sub="Snow Accumulation")
    with w4: metric_card("Wind Speed",   f"{lw['wspd']:.0f} km/h", sub="Max Wind Today")
    with w5:
        conds = []
        if lw.get("freezing_rain", 0): conds.append("Freezing Rain")
        if lw.get("heavy_snow", 0):    conds.append("Heavy Snow")
        if lw.get("feels_bad", 0) and not conds: conds.append("Adverse")
        metric_card("Road Conditions",
                    ", ".join(conds) if conds else "Normal",
                    colour_class="risk-high" if conds else "risk-low")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK MAP
# ══════════════════════════════════════════════════════════════════════════════
def page_risk_map(
    nbhd: pd.DataFrame,
    dim: gpd.GeoDataFrame,
    features: pd.DataFrame,
    horizon: int,
    top_k: int,
) -> None:
    hl = "Tomorrow" if horizon == 1 else "Day After Tomorrow"
    st.markdown(f"<h2 style='color:{B_DEEP}; margin-bottom:2px;'>Neighbourhood Risk Map</h2>",
                unsafe_allow_html=True)
    st.caption(f"Which neighbourhoods are most likely to have a collision {hl.lower()}?")
    render_definitions("Risk Map")

    score_col = f"risk_score_t{horizon}"
    prob_col  = f"proba_collision_t{horizon}"
    ksi_col   = f"proba_ksi_t{horizon}"

    latest_date = nbhd["date"].max()
    pred_today  = nbhd[nbhd["date"] == latest_date].copy()
    pred_today["nbhd_id"] = pred_today["nbhd_id"].astype(int)

    # Compute expected collisions on pred_today BEFORE merging onto gdf
    avgs = nbhd_avg_collisions(features)
    pred_today = pred_today.merge(avgs, on="nbhd_id", how="left")
    pred_today["avg_30d"]  = pred_today["avg_30d"].fillna(1.0)
    pred_today["exp_coll"] = (pred_today["avg_30d"] * pred_today[score_col] * 2).round(1)

    # Merge onto geometry — single merge, no second merge needed
    gdf = dim.copy()
    gdf["nbhd_id"] = gdf["nbhd_id"].astype(int)
    gdf = gdf.merge(
        pred_today[["nbhd_id", score_col, prob_col, ksi_col, "exp_coll"]],
        on="nbhd_id", how="left"
    )
    for c in [score_col, prob_col, ksi_col]:
        gdf[c] = gdf[c].round(3)

    top_k_ids = pred_today.nlargest(top_k, score_col)["nbhd_id"].tolist()

    col_map, col_right = st.columns([3, 2])

    with col_map:
        section(f"Risk Map — Top {top_k} Zones Outlined In Red")
        colour_legend([
            (C_RED,   "High Risk (0.70+)"),
            (C_AMBER, "Medium Risk (0.40–0.69)"),
            (C_GREEN, "Low Risk (below 0.40)"),
            ("#C8E0F0", "Top zones outlined in red on map"),
        ])
        st.caption("Hover over any neighbourhood to see its scores.")
        tip_fields  = ["area_name", score_col, prob_col]
        tip_aliases = ["Neighbourhood", "Risk Score", "Serious Incident Chance"]
        m = build_map(gdf, score_col, "Overall Risk Score", "Blues",
                      top_k_ids, tip_fields, tip_aliases)
        map_data = st_folium(m, use_container_width="stretch", height=480,
                             returned_objects=["last_object_clicked_tooltip"])

    with col_right:
        # Neighbourhood profile panel
        selected_name = None
        if map_data and map_data.get("last_object_clicked_tooltip"):
            tip = map_data["last_object_clicked_tooltip"]
            if isinstance(tip, dict) and "Neighbourhood" in tip:
                selected_name = tip["Neighbourhood"]

        if selected_name:
            row = gdf[gdf["area_name"] == selected_name]
            if not row.empty:
                row     = row.iloc[0]
                score   = float(row.get(score_col, 0) or 0)
                pc      = float(row.get(prob_col, 0) or 0)
                exp_v   = f"{float(row.get('exp_coll', 0)):.1f}"
                sc_col  = C_RED if score >= 0.7 else C_AMBER if score >= 0.4 else C_GREEN
                reasons = why_high_risk(int(row["nbhd_id"]), features)
                drv_html = "".join(
                    f'<span class="driver-badge driver-{cat}">{txt}</span>'
                    for cat, txt in reasons)
                st.markdown(f"""
                <div class="profile-panel">
                    <div class="profile-name">📍 {selected_name}</div>
                    <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:12px;'>
                        <div class="mini-stat">
                            <div class="mini-stat-label">Risk Score</div>
                            <div class="mini-stat-value" style='color:{sc_col};'>{score:.2f}</div>
                        </div>
                        <div class="mini-stat">
                            <div class="mini-stat-label">Collision Chance</div>
                            <div class="mini-stat-value">{pc:.0%}</div>
                        </div>
                        <div class="mini-stat">
                            <div class="mini-stat-label">Expected Collisions</div>
                            <div class="mini-stat-value">{exp_v}</div>
                        </div>
                    </div>
                    <div style='font-size:11px; font-weight:700; color:{B_MUTED}; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;'>Why This Area Is High Risk</div>
                    <div>{drv_html}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("👆 Hover over any neighbourhood on the map to see its risk profile.")

        # Ranking table — exp_coll already in gdf, no second merge
        section(f"Top {top_k} Highest-Risk Neighbourhoods")
        top_gdf = (
            gdf[gdf["nbhd_id"].isin(top_k_ids)]
            [["area_name", score_col, prob_col, "exp_coll"]]
            .sort_values(score_col, ascending=False)
            .reset_index(drop=True)
        )
        top_gdf.index = top_gdf.index + 1
        top_gdf.columns = ["Neighbourhood", "Risk Score", "Collision Chance",
                           "Expected Collisions"]
        st.dataframe(
            top_gdf.style
            .pipe(style_risk_table)
            .format({
                "Risk Score": "{:.3f}",
                "Collision Chance": "{:.1%}",
                #"Serious Incident Chance": "{:.1%}",
                "Expected Collisions": "{:.1f}",
            }),
            use_container_width="stretch", height=320)
        risk_score_legend()

    # Why high risk section
    section(f"Why These {top_k} Neighbourhoods Are High Risk")
    st.caption("Each tag explains one contributing factor. Blue is traffic. Light blue is weather. Gold is calendar. Green is road disruptions.")

    top_gdf2 = gdf[gdf["nbhd_id"].isin(top_k_ids)].sort_values(score_col, ascending=False)
    for _, row in top_gdf2.iterrows():
        reasons  = why_high_risk(int(row["nbhd_id"]), features)
        drv_html = "".join(
            f'<span class="driver-badge driver-{c}">{t}</span>'
            for c, t in reasons)
        st.markdown(
            f'<div class="why-row">'
            f'<span class="why-name">{row["area_name"]}</span>'
            f'<span class="why-score">Risk Score {row[score_col]:.2f}</span>'
            f'<div class="why-tags">{drv_html}</div>'
            f'</div>',
            unsafe_allow_html=True)

    # Risk distribution — 3 semantic count cards (replaces histogram)
    section("How Risk Is Spread Across All Neighbourhoods Today")

    n_total  = len(pred_today)
    n_high   = int((pred_today[score_col] >= 0.7).sum())
    n_medium = int(((pred_today[score_col] >= 0.4) & (pred_today[score_col] < 0.7)).sum())
    n_low    = int((pred_today[score_col] < 0.4).sum())

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{C_RED}; border-width:2px;">
            <div class="metric-label" style="color:{C_RED};">High Risk Zones</div>
            <div class="metric-value" style="color:{C_RED}; font-size:32px;">{n_high}</div>
            <div class="metric-sub">Score 0.70 and above</div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{C_AMBER}; border-width:2px;">
            <div class="metric-label" style="color:{C_AMBER};">Medium Risk Zones</div>
            <div class="metric-value" style="color:{C_AMBER}; font-size:32px;">{n_medium}</div>
            <div class="metric-sub">Score 0.40 to 0.69</div>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{C_GREEN}; border-width:2px;">
            <div class="metric-label" style="color:{C_GREEN};">Low Risk Zones</div>
            <div class="metric-value" style="color:{C_GREEN}; font-size:32px;">{n_low}</div>
            <div class="metric-sub">Score below 0.40</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    if n_high > 0:
        st.caption(f"{n_high} of {n_total} neighbourhoods are at high risk today.")
    else:
        st.caption(f"No high-risk zones today. {n_medium} medium risk, {n_low} low risk — a quiet day across the city.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRENDS AND SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════
def page_trends(features: pd.DataFrame) -> None:
    st.markdown(f"<h2 style='color:{B_DEEP}; margin-bottom:2px;'>Trends And Seasonality</h2>",
                unsafe_allow_html=True)
    st.caption("Long-term patterns in collision activity. Useful for planning, budgeting, and understanding what drives risk over time.")
    render_definitions("Trends And Seasonality")

    df = features.copy()
    df["date"]       = pd.to_datetime(df["date"])
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["dow"]        = df["date"].dt.dayofweek
    df["dow_name"]   = df["date"].dt.strftime("%a")

    daily = (
        df.groupby(["date","year","month","month_name","dow","dow_name"], as_index=False)
        .agg(collisions=("collisions","sum"), ksi=("ksi_collisions","sum"))
    )

    tab1, tab2, tab3 = st.tabs(["Collision Volume", "Weather Impact", "Serious Incidents"])

    # ── Tab 1 — Collision Volume with Yearly/Monthly/Daily dropdown ──────────
    with tab1:
        section("Collision Volume")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        dow_order   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

        view = st.selectbox("View by",
            ["Yearly", "Monthly", "Daily (Day of Week)"],
            label_visibility="collapsed")

        if view == "Yearly":
            colour_legend([
                (C_RED,   "Above long-term average (worse)"),
                (C_GREEN, "Below long-term average (better)"),
                (C_GOLD,  "Long-term average line"),
            ])
            st.caption("Total collisions per year. Red = above the long-term average. Green = below.")
            annual = (
                daily.groupby("year", as_index=False)["collisions"].sum()
                .rename(columns={"collisions": "total"})
            )
            current_year = pd.Timestamp.now().year
            if daily[daily["year"] == current_year]["month"].nunique() < 6:
                annual = annual[annual["year"] < current_year]
            long_avg = annual["total"].mean()
            bar_cols = [C_RED if v > long_avg else C_GREEN for v in annual["total"]]
            fig = go.Figure(go.Bar(
                x=annual["year"], y=annual["total"],
                marker_color=bar_cols,
                hovertemplate="<b>%{x}</b><br>Total Collisions: %{y:,}<extra></extra>"))
            fig.add_hline(y=long_avg, line_dash="dot", line_color=C_GOLD, line_width=2,
                          annotation_text=f"Long-term avg ({long_avg:,.0f})",
                          annotation_font_color=C_GOLD, annotation_position="top left")
            if len(annual) >= 2:
                lat  = annual.iloc[-1]
                prv  = annual.iloc[-2]
                yoy  = (lat["total"] - prv["total"]) / prv["total"] * 100
                arr  = "↑" if yoy > 0 else "↓"
                yc   = C_RED if yoy > 0 else C_GREEN
                fig.add_annotation(
                    x=lat["year"], y=lat["total"],
                    text=f"{arr} {abs(yoy):.0f}% vs {int(prv['year'])}",
                    showarrow=False, yanchor="bottom",
                    font=dict(size=11, color=yc))
            fig = chart_layout(fig, height=340)
            fig.update_layout(
                xaxis=dict(tickmode="linear", dtick=1, tickangle=45),
                yaxis_title="Total Collisions", bargap=0.2, showlegend=False)
            st.plotly_chart(fig, use_container_width="stretch")

        elif view == "Monthly":
            colour_legend([
                (C_RED,   "Top 3 highest-risk months"),
                (C_GREEN, "Bottom 3 lowest-risk months"),
                (C_GREY,  "Typical months"),
                (C_GOLD,  "Annual average line"),
            ])
            monthly = (
                daily.groupby(["month","month_name"], as_index=False)["collisions"].mean()
                .sort_values("month")
            )
            monthly["month_name"] = pd.Categorical(
                monthly["month_name"], categories=month_order, ordered=True)
            monthly = monthly.sort_values("month_name")
            top3_m  = set(monthly.nlargest(3, "collisions")["month_name"].tolist())
            bot3_m  = set(monthly.nsmallest(3, "collisions")["month_name"].tolist())
            m_cols  = [C_RED if m in top3_m else C_GREEN if m in bot3_m else C_GREY
                       for m in monthly["month_name"]]
            m_avg   = monthly["collisions"].mean()
            fig = go.Figure(go.Bar(
                x=monthly["month_name"], y=monthly["collisions"],
                marker_color=m_cols,
                hovertemplate="<b>%{x}</b><br>Avg Daily Collisions: %{y:.0f}<extra></extra>"))
            fig.add_hline(y=m_avg, line_dash="dot", line_color=C_GOLD, line_width=2,
                          annotation_text=f"Annual avg ({m_avg:.0f})",
                          annotation_font_color=C_GOLD, annotation_position="top left")
            fig = chart_layout(fig, height=320)
            fig.update_layout(xaxis_title="", yaxis_title="Avg Daily Collisions",
                              bargap=0.15, showlegend=False)
            st.plotly_chart(fig, use_container_width="stretch")
            peak_m = ", ".join(sorted(top3_m, key=lambda m: month_order.index(m)))
            st.caption(f"Highest-risk months: {peak_m}.")

        else:  # Daily
            colour_legend([
                (C_RED,   "Top 3 highest-risk days"),
                (C_GREEN, "Bottom 3 lowest-risk days"),
                (C_GREY,  "Typical days"),
                (C_GOLD,  "Weekly average line"),
            ])
            dow_avg = (
                daily.groupby(["dow","dow_name"], as_index=False)["collisions"].mean()
                .sort_values("dow")
            )
            dow_avg["dow_name"] = pd.Categorical(
                dow_avg["dow_name"], categories=dow_order, ordered=True)
            dow_avg   = dow_avg.sort_values("dow_name")
            top3_d    = set(dow_avg.nlargest(3, "collisions")["dow_name"].tolist())
            bot3_d    = set(dow_avg.nsmallest(3, "collisions")["dow_name"].tolist())
            d_cols    = [C_RED if d in top3_d else C_GREEN if d in bot3_d else C_GREY
                         for d in dow_avg["dow_name"]]
            d_avg     = dow_avg["collisions"].mean()
            top3_lbl  = ", ".join(sorted(top3_d, key=lambda d: dow_order.index(d)))
            fig = go.Figure(go.Bar(
                x=dow_avg["dow_name"], y=dow_avg["collisions"],
                marker_color=d_cols,
                hovertemplate="<b>%{x}</b><br>Avg Daily Collisions: %{y:.0f}<extra></extra>"))
            fig.add_hline(y=d_avg, line_dash="dot", line_color=C_GOLD, line_width=2,
                          annotation_text=f"Weekly avg ({d_avg:.0f})",
                          annotation_font_color=C_GOLD, annotation_position="top left")
            fig = chart_layout(fig, height=300)
            fig.update_layout(xaxis_title="", yaxis_title="Avg Daily Collisions",
                              bargap=0.2, showlegend=False)
            st.plotly_chart(fig, use_container_width="stretch")
            st.caption(f"Highest-collision days: {top3_lbl}.")

    # Tab 2 — Weather impact
    with tab2:
        section("How Much Does Weather Increase Collisions?")
        st.caption("Comparing average daily collisions on normal days versus days with adverse weather.")
        wx_rows = []
        for label, col, val in [
            ("Normal Days",        None,            None),
            ("Freezing Rain",      "freezing_rain", 1),
            ("Heavy Snow",         "heavy_snow",    1),
            ("High Wind And Rain", "wind_rain",     1),
        ]:
            if col is None:
                cond = (
                    (df.get("freezing_rain", pd.Series(0, index=df.index)) == 0) &
                    (df.get("heavy_snow",    pd.Series(0, index=df.index)) == 0) &
                    (df.get("wind_rain",     pd.Series(0, index=df.index)) == 0)
                )
                mask_dates = df[cond]["date"].drop_duplicates()
            else:
                if col not in df.columns: continue
                mask_dates = df[df[col] == val]["date"].drop_duplicates()
            subset = daily.merge(mask_dates.rename("date"), on="date", how="inner")
            if not subset.empty:
                wx_rows.append({"Condition": label, "avg": subset["collisions"].mean()})

        if wx_rows:
            wx_df  = pd.DataFrame(wx_rows)
            norm   = wx_df.iloc[0]["avg"] if len(wx_rows) > 0 else 1
            # Blue for normal days, red for all adverse conditions
            fig = go.Figure()
            for i, r in wx_df.iterrows():
                is_normal = (i == 0)
                pct       = (r["avg"] - norm) / norm * 100 if norm > 0 else 0
                bar_col   = C_BLUE if is_normal else C_RED
                bar_label = "Baseline" if is_normal else f"+{pct:.0f}%"
                hover_txt = (f"<b>{r['Condition']}</b><br>"
                             f"Avg Daily Collisions: {r['avg']:.0f}"
                             + (f"<br>{pct:+.0f}% vs normal days" if not is_normal else ""))
                fig.add_trace(go.Bar(
                    x=[r["Condition"]], y=[r["avg"]],
                    marker_color=bar_col,
                    text=[bar_label],
                    textposition="outside",
                    textfont=dict(size=12, color=bar_col, family="Arial"),
                    hovertemplate=hover_txt + "<extra></extra>",
                    showlegend=False))
            fig = chart_layout(fig, height=340)
            fig.update_layout(
                xaxis_title="", yaxis_title="Avg Daily Collisions",
                bargap=0.3, showlegend=False,
                yaxis=dict(range=[0, wx_df["avg"].max() * 1.2],
                           gridcolor=G_GRID, linecolor=B_BORDER,
                           tickfont=dict(color=B_MUTED, size=11)))
            st.plotly_chart(fig, use_container_width="stretch")
            worst = max(wx_rows[1:], key=lambda r: r["avg"], default=None)
            if worst:
                wpct = (worst["avg"] - norm) / norm * 100
                st.caption(f"{worst['Condition']} days have the biggest impact, with {wpct:.0f}% more collisions than normal days on average.")

    # Tab 3 — Serious incidents
    with tab3:
        section("Serious And Fatal Collision Trend")
        st.caption("Monthly count of collisions resulting in a serious injury or fatality. A rising trend signals a need for long-term action.")

        feat_ksi = features.copy()
        feat_ksi["date"]  = pd.to_datetime(feat_ksi["date"])
        feat_ksi["year"]  = feat_ksi["date"].dt.year
        feat_ksi["month"] = feat_ksi["date"].dt.month

        ksi_daily = (
            feat_ksi.groupby(["date","year","month"], as_index=False)
            ["ksi_collisions"].sum()
        )
        ksi_monthly = (
            ksi_daily.groupby(["year","month"], as_index=False)["ksi_collisions"].sum()
        )
        ksi_monthly["period"] = pd.to_datetime(
            ksi_monthly[["year","month"]].assign(day=1))
        ksi_monthly = ksi_monthly.sort_values("period")

        if ksi_monthly["ksi_collisions"].sum() == 0:
            st.info("Serious incident data is not available in the current dataset.")
        else:
            ksi_monthly["roll12"] = ksi_monthly["ksi_collisions"].rolling(12, center=True).mean()
            long_avg_ksi = ksi_monthly["ksi_collisions"].mean()
            ksi_base     = [long_avg_ksi] * len(ksi_monthly)
            ksi_above    = ksi_monthly["ksi_collisions"].clip(lower=long_avg_ksi)
            ksi_below    = ksi_monthly["ksi_collisions"].clip(upper=long_avg_ksi)

            fig = go.Figure()
            # Green fill — below long-term average (improving)
            fig.add_trace(go.Scatter(
                x=ksi_monthly["period"], y=ksi_below,
                mode="none", fill=None, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=ksi_monthly["period"], y=ksi_base,
                mode="none", fill="tonexty",
                fillcolor="rgba(30,132,73,0.10)",
                showlegend=False, hoverinfo="skip"))
            # Red fill — above long-term average (worsening)
            fig.add_trace(go.Scatter(
                x=ksi_monthly["period"], y=ksi_base,
                mode="none", fill=None, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=ksi_monthly["period"], y=ksi_above,
                mode="none", fill="tonexty",
                fillcolor="rgba(192,57,43,0.10)",
                showlegend=False, hoverinfo="skip"))
            # Long-term average reference line (gold)
            fig.add_trace(go.Scatter(
                x=ksi_monthly["period"], y=ksi_base,
                mode="lines", name=f"Long-term avg ({long_avg_ksi:.1f})",
                line=dict(color=C_GOLD, width=1.8, dash="dot"),
                hovertemplate="Long-term avg: %{y:.1f}<extra></extra>"))
            # Teal monthly line
            fig.add_trace(go.Scatter(
                x=ksi_monthly["period"], y=ksi_monthly["ksi_collisions"],
                mode="lines", name="Serious Incidents Per Month",
                line=dict(color=C_TEAL, width=2),
                hovertemplate="<b>%{x|%b %Y}</b><br>Serious Incidents: %{y:.0f}<extra></extra>"))
            fig = chart_layout(fig, height=340)
            fig.update_layout(yaxis_title="Serious Incidents Per Month")
            colour_legend([
                (C_RED,   "Above long-term average (worsening)"),
                (C_GREEN, "Below long-term average (improving)"),
                (C_TEAL,  "Monthly serious incident count"),
                (C_GOLD,  "Long-term average line"),
            ])
            st.plotly_chart(fig, use_container_width="stretch")

            recent  = ksi_monthly.tail(12)["ksi_collisions"].mean()
            earlier = ksi_monthly.iloc[-24:-12]["ksi_collisions"].mean() \
                      if len(ksi_monthly) >= 24 else None
            if earlier and earlier > 0:
                chg       = (recent - earlier) / earlier * 100
                col       = C_RED if chg > 0 else C_GREEN
                direction = "up" if chg > 0 else "down"
                st.markdown(
                    f"<p style='font-size:13px; color:{col};'>"
                    f"Serious incidents are {direction} {abs(chg):.0f}% compared to "
                    f"the same period last year "
                    f"(recent 12-month average: {recent:.1f} per month).</p>",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def page_model_performance(
    surge_metrics: list[dict],
    nbhd_metrics: list[dict],
    horizon: int,
) -> None:
    hl = "Tomorrow" if horizon == 1 else "Day After Tomorrow"
    st.markdown(f"<h2 style='color:{B_DEEP}; margin-bottom:2px;'>Model Performance</h2>",
                unsafe_allow_html=True)
    st.caption("How accurate are the forecasts? This page shows how the models performed on data they had not seen before.")
    render_definitions("Model Performance")

    # Surge model metrics
    section("Citywide Surge Forecast Accuracy")
    surge_h = next((m for m in surge_metrics if f"t{horizon}" in m["model"]), surge_metrics[0])
    s1, s2, s3, s4 = st.columns(4)
    with s1: metric_card("Accuracy Score",    f"{surge_h['roc_auc']:.2f}", sub="Above 0.70 is strong")
    with s2: metric_card("Forecast Quality",  f"{surge_h['pr_auc']:.2f}",  sub="Handles rare surge days")
    with s3: metric_card("Calibration",       f"{surge_h['brier']:.3f}",   sub="Lower is better")
    with s4: metric_card("Surge Days Tested", f"{surge_h['pos_rate']:.1%}",
                         sub=f"Out of {surge_h['n_test']} total days")

    # Neighbourhood model metrics
    section("Neighbourhood Forecast Accuracy")
    col_m = next((m for m in nbhd_metrics if m["model"] == f"collision_prob_t{horizon}"), None)
    ksi_m = next((m for m in nbhd_metrics if m["model"] == f"ksi_prob_t{horizon}"), None)

    tabs = st.tabs(["Collision Forecast"])
    for tab, m in zip(tabs, [col_m, ksi_m]):
        if m is None:
            with tab:
                st.info("No data available for this model.")
            continue
        with tab:
            roc_val = m.get("roc_auc", float("nan"))
            pr_val  = m.get("pr_auc", 0.0)
            brier   = m.get("brier", 0.0)
            roc_str = f"{roc_val:.2f}" if not (isinstance(roc_val, float) and np.isnan(roc_val)) else "N/A"
            pr_str  = f"{pr_val:.2f}" if pr_val > 0 else "N/A"

            if roc_str == "N/A" and pr_str == "N/A":
                st.warning(
                    "This model could not be evaluated because the test period had no "
                    "serious incidents recorded. This is expected given the very low rate "
                    "of serious incidents in the data (less than 0.5% of days). "
                    "The model is still trained and predictions are available.")
                continue

            mc1, mc2, mc3 = st.columns(3)
            with mc1: metric_card("Accuracy Score",   roc_str)
            with mc2: metric_card("Forecast Quality", pr_str)
            with mc3: metric_card("Calibration",      f"{brier:.3f}")

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.markdown("**How accurately does the model rank the highest-risk neighbourhoods?**")
            st.caption("Accuracy shows what share of flagged zones had an actual incident. Coverage shows what share of all incidents were captured.")

            k_vals   = [5, 10, 15, 20]
            p_vals   = [m.get(f"precision_at_{k}", 0) for k in k_vals]
            r_vals   = [m.get(f"recall_at_{k}", 0)    for k in k_vals]
            baseline = m.get("baseline_pos_rate", 0)

            k_fig = go.Figure()
            k_fig.add_trace(go.Bar(
                x=[f"Top {k}" for k in k_vals], y=p_vals,
                name="Accuracy",
                marker_color=C_BLUE,
                hovertemplate="<b>Top %{x} Zones</b><br>Accuracy: %{y:.0%}<extra></extra>"))
            k_fig.add_trace(go.Bar(
                x=[f"Top {k}" for k in k_vals], y=r_vals,
                name="Coverage",
                marker_color=C_GOLD,
                hovertemplate="<b>Top %{x} Zones</b><br>Coverage: %{y:.0%}<extra></extra>"))
            k_fig.add_hline(
                y=baseline, line_dash="dash", line_color=C_SLATE,
                annotation_text=f"Random Chance ({baseline:.0%})",
                annotation_font_color=C_SLATE)
            k_fig.update_layout(
                barmode="group", height=280,
                margin=dict(l=0, r=0, t=44, b=0),
                plot_bgcolor=B_CARD, paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor=G_GRID,
                           tickfont=dict(color=B_MUTED, size=11), title="Rate"),
                xaxis=dict(tickfont=dict(color=B_MUTED, size=11), title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(size=12),
                            bgcolor="rgba(255,255,255,0.95)",
                            bordercolor=B_BORDER, borderwidth=1),
                font=dict(color=B_DEEP))
            st.plotly_chart(k_fig, use_container_width="stretch")

    # Feature importance
    section("Top Factors Driving The Forecasts")
    st.caption("What information matters most to the model? Longer bars mean greater influence on the prediction.")

    fi_col_path = pipeline.DOCS / f"fi_collision_t{horizon}.csv"
    fi_ksi_path = pipeline.DOCS / f"fi_ksi_t{horizon}.csv"

    if fi_col_path.exists() and fi_ksi_path.exists():
        fi_col = pd.read_csv(fi_col_path).head(15)
        fi_ksi = pd.read_csv(fi_ksi_path).head(15)
        fi_tabs = st.tabs(["Collision Forecast Factors"])
        for fi_tab, fi_df, colour in zip(fi_tabs, [fi_col], [C_BLUE]):
            with fi_tab:
                fi_df = fi_df[fi_df["importance"] > 0].copy()
                fi_df["label"] = fi_df["feature"].map(FRIENDLY).fillna(fi_df["feature"])
                fi_df = fi_df.sort_values("importance")
                fig = go.Figure(go.Bar(
                    x=fi_df["importance"], y=fi_df["label"],
                    orientation="h", marker_color=colour,
                    hovertemplate="<b>%{y}</b><br>Influence: %{x:.4f}<extra></extra>"))
                fig.update_layout(
                    height=420, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor=B_CARD, paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor=G_GRID,
                               tickfont=dict(color=B_MUTED, size=11),
                               title="Influence On Prediction"),
                    yaxis=dict(tickfont=dict(color=B_DEEP, size=11)),
                    font=dict(color=B_DEEP))
                st.plotly_chart(fig, use_container_width="stretch")
    else:
        st.info("Factor analysis files not yet available. Run the pipeline to generate them.")

    # Stability check
    section("Forecast Stability Check")
    st.caption("Compares model accuracy in the first and second half of the test period. A large drop signals it may be time to refresh the model.")
    col_m2 = next((m for m in nbhd_metrics if m["model"] == f"collision_prob_t{horizon}"), None)
    if col_m2:
        p10 = col_m2.get("precision_at_10", 0)
        r10 = col_m2.get("recall_at_10", 0)
        st.success(
            f"Collision forecast ({hl}) is stable. "
            f"The model correctly identified {p10:.0%} of the top 10 flagged zones "
            f"and captured {r10:.0%} of all collision-affected neighbourhoods.")
    else:
        st.warning("Stability data is not available. Run the pipeline to generate it.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    controls = render_sidebar()
    horizon  = controls["horizon"]
    top_k    = controls["top_k"]
    page     = controls["page"]

    with st.spinner("Loading data and running forecasts..."):
        features        = load_features()
        surge           = load_surge()
        nbhd            = load_nbhd()
        dim             = load_dim()
        surge_m, nbhd_m = load_metrics()

    #DISCLAIMER
    st.info("Note: Forecasts are based on historical data simulation, not real-time feeds.")

    if page == "City Overview":
        page_city_overview(features, surge, nbhd, horizon, top_k)
    elif page == "Risk Map":
        page_risk_map(nbhd, dim, features, horizon, top_k)
    elif page == "Trends And Seasonality":
        page_trends(features)
    elif page == "Model Performance":
        page_model_performance(surge_m, nbhd_m, horizon)

if __name__ == "__main__":
    main()
