from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALERT_PATH = REPO_ROOT / "docs" / "mlops" / "monitoring_alerts.json"


def load_alerts():
    if not ALERT_PATH.exists():
        print("No alerts file found.")
        return None

    with open(ALERT_PATH) as f:
        return json.load(f)


def should_trigger(alerts: dict) -> bool:
    if alerts["status"] != "ALERT":
        return False

    for alert in alerts["alerts"]:
        if alert["type"] == "artifact_freshness":
            return True

    return False


def trigger_pipeline():
    print("🚀 Triggering ML pipeline due to stale data...")

    subprocess.run(
        ["poetry", "run", "python", "-m", "src.pipeline.run_pipeline"],
        check=True,
    )


def main():
    alerts = load_alerts()

    if not alerts:
        print("No alerts available.")
        return

    if should_trigger(alerts):
        trigger_pipeline()
    else:
        print("No triggering conditions met.")


if __name__ == "__main__":
    main()