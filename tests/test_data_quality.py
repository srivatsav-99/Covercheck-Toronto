import sys
from pathlib import Path
import pandas as pd
import pytest

# Path hack for test isolation
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_paths import FEATURES_PATH, SURGE_PRED_PATH, NBHD_PRED_PATH

@pytest.fixture
def features_df():
    """Pytest fixture to load features once for all tests."""
    return pd.read_parquet(FEATURES_PATH)

@pytest.fixture
def surge_df():
    """Pytest fixture to load surge predictions."""
    return pd.read_parquet(SURGE_PRED_PATH)

@pytest.fixture
def nbhd_df():
    """Pytest fixture to load neighbourhood predictions."""
    return pd.read_parquet(NBHD_PRED_PATH)

def test_no_missing_targets(features_df):
    """Ensure the target columns exist and contain no nulls."""
    targets = ["target_collision_t1", "target_ksi_t1"]
    for col in targets:
        assert col in features_df.columns, f"Missing target column: {col}"
        assert features_df[col].notna().all(), f"Found nulls in {col}"

def test_surge_probabilities_in_range(surge_df):
    """Ensure the surge probabilities are strictly between 0 and 1 (ignoring end-of-series nulls)."""
    assert surge_df["surge_proba_t1"].dropna().between(0, 1).all(), "T+1 surge prob out of bounds"
    assert surge_df["surge_proba_t2"].dropna().between(0, 1).all(), "T+2 surge prob out of bounds"

def test_nbhd_risk_scores_in_range(nbhd_df):
    """Ensure the final risk scores are strictly between 0 and 1."""
    assert nbhd_df["risk_score_t1"].between(0, 1).all(), "T+1 risk score out of bounds"
    assert nbhd_df["risk_score_t2"].between(0, 1).all(), "T+2 risk score out of bounds"

def test_no_future_leakage(features_df):
    """Ensure future targets are strictly 0 or 1 (binary), not raw counts."""
    assert set(features_df["target_collision_t1"].unique()).issubset({0, 1})
    assert set(features_df["target_ksi_t1"].unique()).issubset({0, 1})