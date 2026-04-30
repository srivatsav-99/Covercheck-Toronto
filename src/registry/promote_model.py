from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_ROOT = REPO_ROOT / "models" / "registry"
NBHD_REGISTRY = REGISTRY_ROOT / "neighbourhood_collision"

CHAMPION_DIR = NBHD_REGISTRY / "champion"
CHAMPION_MODEL_PATH = CHAMPION_DIR / "model.pkl"
CHAMPION_METADATA_PATH = CHAMPION_DIR / "metadata.json"

PRIMARY_METRIC = "precision_at_10"
SECONDARY_METRIC = "pr_auc"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_metric(metadata: dict, metric_name: str) -> float:
    value = metadata.get("metrics", {}).get(metric_name)

    if value is None:
        return float("-inf")

    return float(value)


def candidate_beats_champion(candidate: dict, champion: dict) -> bool:
    cand_primary = get_metric(candidate, PRIMARY_METRIC)
    champ_primary = get_metric(champion, PRIMARY_METRIC)

    if cand_primary > champ_primary:
        return True

    if cand_primary < champ_primary:
        return False

    cand_secondary = get_metric(candidate, SECONDARY_METRIC)
    champ_secondary = get_metric(champion, SECONDARY_METRIC)

    return cand_secondary > champ_secondary


def promote_candidate(candidate_dir: Path, approve: bool = False) -> dict:
    candidate_dir = candidate_dir.resolve()

    candidate_model_path = candidate_dir / "model.pkl"
    candidate_metadata_path = candidate_dir / "metadata.json"

    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate directory not found: {candidate_dir}")

    if not candidate_model_path.exists():
        raise FileNotFoundError(f"Candidate model not found: {candidate_model_path}")

    candidate_metadata = load_json(candidate_metadata_path)
    champion_metadata = load_json(CHAMPION_METADATA_PATH)

    beats = candidate_beats_champion(candidate_metadata, champion_metadata)

    decision = {
        "candidate_dir": str(candidate_dir),
        "candidate_version": candidate_metadata.get("version"),
        "candidate_precision_at_10": get_metric(candidate_metadata, PRIMARY_METRIC),
        "champion_precision_at_10": get_metric(champion_metadata, PRIMARY_METRIC),
        "candidate_pr_auc": get_metric(candidate_metadata, SECONDARY_METRIC),
        "champion_pr_auc": get_metric(champion_metadata, SECONDARY_METRIC),
        "beats_champion": beats,
        "approved": approve,
        "promoted": False,
    }

    if not beats:
        print("Candidate does not beat champion. Promotion blocked.")
        print(json.dumps(decision, indent=2))
        return decision

    if not approve:
        print("Candidate beats champion, but approval flag was not provided.")
        print("Re-run with --approve to promote.")
        print(json.dumps(decision, indent=2))
        return decision

    backup_dir = CHAMPION_DIR / f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if CHAMPION_MODEL_PATH.exists():
        shutil.copy2(CHAMPION_MODEL_PATH, backup_dir / "model.pkl")

    if CHAMPION_METADATA_PATH.exists():
        shutil.copy2(CHAMPION_METADATA_PATH, backup_dir / "metadata.json")

    shutil.copy2(candidate_model_path, CHAMPION_MODEL_PATH)

    promoted_metadata = candidate_metadata.copy()
    promoted_metadata["status"] = "champion"
    promoted_metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
    promoted_metadata["promotion_source"] = str(candidate_dir)
    promoted_metadata["previous_champion_backup"] = str(backup_dir)
    promoted_metadata["promotion_rule"] = (
        f"{PRIMARY_METRIC} improves over champion; "
        f"{SECONDARY_METRIC} used as tie-breaker"
    )

    save_json(CHAMPION_METADATA_PATH, promoted_metadata)

    decision["promoted"] = True
    decision["backup_dir"] = str(backup_dir)

    print("Candidate promoted to champion.")
    print(json.dumps(decision, indent=2))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a candidate model to champion.")
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Path to candidate directory containing model.pkl and metadata.json",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Required to actually promote a candidate that beats champion",
    )

    args = parser.parse_args()

    promote_candidate(
        candidate_dir=Path(args.candidate_dir),
        approve=args.approve,
    )


if __name__ == "__main__":
    main()