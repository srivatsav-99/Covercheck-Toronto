from prefect import flow, task
from datetime import datetime
from src.scoring.score_latest import score_latest


@task
def run_scoring():
    print("Starting scoring task...")
    result = score_latest()
    print("Scoring result:", result)
    return result


@flow(name="covercheck-daily-refresh")
def refresh_flow():
    print("=== Prefect Flow Started ===")
    print("Run time:", datetime.utcnow())

    result = run_scoring()

    print("=== Flow Complete ===")
    return result


if __name__ == "__main__":
    refresh_flow.serve(
        name="daily-refresh",
        cron="0 8 * * *",
    )