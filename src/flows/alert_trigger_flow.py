from __future__ import annotations

from datetime import datetime

from prefect import flow, task

from src.monitoring.monitor_model import main as run_monitoring
from src.monitoring.trigger_pipeline import main as trigger_pipeline_if_needed


@task(name="run-monitoring")
def run_monitoring_task() -> None:
    run_monitoring()


@task(name="trigger-pipeline-if-needed")
def trigger_pipeline_task() -> None:
    trigger_pipeline_if_needed()


@flow(name="covercheck-alert-trigger-flow")
def alert_trigger_flow() -> None:
    print("=== Alert Trigger Flow Started ===")
    print(f"Run time: {datetime.now()}")

    run_monitoring_task()
    trigger_pipeline_task()

    print("=== Alert Trigger Flow Complete ===")


if __name__ == "__main__":
    alert_trigger_flow.serve(
        name="daily-alert-trigger",
        cron="0 9 * * *",
    )