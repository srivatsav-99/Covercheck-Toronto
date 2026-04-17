import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Path hack for test isolation
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.api import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"

def test_surge_latest():
    resp = client.get("/surge/latest")
    assert resp.status_code in (200, 404)

def test_topk():
    resp = client.get("/neighbourhoods/topk?horizon=1&k=10")
    assert resp.status_code in (200, 404)

def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200