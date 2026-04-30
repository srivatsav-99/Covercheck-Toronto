from __future__ import annotations

from datetime import datetime

from prefect import flow, task

from src.pipeline.run_pipeline import main as run_full_pipeline


@task(name="run-full-ml-pipeline")
def run_pipeline_task() -> None:
    print("Starting full ML pipeline from Prefect...")
    run_full_pipeline()


@flow(name="covercheck-scheduled-ml-pipeline")
def scheduled_ml_pipeline() -> None:
    print("=== Scheduled ML Pipeline Started ===")
    print(f"Run time: {datetime.now()}")

    run_pipeline_task()

    print("=== Scheduled ML Pipeline Complete ===")


if __name__ == "__main__":
    scheduled_ml_pipeline.serve(
        name="weekly-ml-refresh",
        cron="0 8 * * 1",
    )