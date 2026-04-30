from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_DIR = None  # will be captured dynamically


def run_step(cmd: str):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {cmd}")


def train_candidate():
    global CANDIDATE_DIR

    print("\n=== STEP 1: Train Candidate ===")
    result = subprocess.run(
        "poetry run python -m src.training.train_nbhd_collision",
        shell=True,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Candidate training failed")

    # Extract the final JSON block printed by the training script.
    start = result.stdout.rfind("{")
    end = result.stdout.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Could not find JSON summary in training output")

    import json

    summary = json.loads(result.stdout[start : end + 1])
    CANDIDATE_DIR = summary["candidate_dir"]

    print(f"Candidate created at: {CANDIDATE_DIR}")


def promote_if_valid():
    print("\n=== STEP 2: Evaluate & Promote ===")

    cmd = f'poetry run python -m src.registry.promote_model --candidate-dir "{CANDIDATE_DIR}" --approve'
    run_step(cmd)


def score_latest():
    print("\n=== STEP 3: Score Latest ===")
    run_step("poetry run python -m src.scoring.score_latest")


def monitor():
    print("\n=== STEP 4: Monitor ===")
    run_step("poetry run python -m src.monitoring.monitor_model")


def main():
    print("\n🚀 Running Full ML Pipeline")

    train_candidate()
    promote_if_valid()
    score_latest()
    monitor()

    print("\n✅ Pipeline Complete")


if __name__ == "__main__":
    main()