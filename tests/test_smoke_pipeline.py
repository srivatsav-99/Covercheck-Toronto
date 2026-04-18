import sys
from pathlib import Path

import pandas as pd
import pytest

# Add the repo root to the Python path so 'src' becomes importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_paths import FEATURES_PATH, SURGE_PRED_PATH, NBHD_PRED_PATH

pytestmark = pytest.mark.local_artifacts


def require_artifacts():
    required = [FEATURES_PATH, SURGE_PRED_PATH, NBHD_PRED_PATH]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        pytest.skip(f"Skipping artifact-dependent tests; missing files: {missing}")


def test_artifacts_exist():
    """Check that the pipeline successfully generated all expected output files."""
    require_artifacts()
    assert FEATURES_PATH.exists(), f"Missing {FEATURES_PATH.name}"
    assert SURGE_PRED_PATH.exists(), f"Missing {SURGE_PRED_PATH.name}"
    assert NBHD_PRED_PATH.exists(), f"Missing {NBHD_PRED_PATH.name}"


def test_nbhd_predictions_shape():
    """Verify the final neighborhood predictions meet the required data contract."""
    require_artifacts()
    df = pd.read_parquet(NBHD_PRED_PATH)

    assert "date" in df.columns, "Missing 'date' column in predictions"
    assert "nbhd_id" in df.columns, "Missing 'nbhd_id' column in predictions"

    unique_nbhds = df["nbhd_id"].nunique()
    assert unique_nbhds == 158, f"Expected 158 neighbourhoods, found {unique_nbhds}"