"""Apply the lab's two-standard-error gate to a registered model candidate."""

import argparse
import math
import os
from pathlib import Path

from mlflow import MlflowClient


def promotion_threshold(accuracy: float, test_size: int) -> float:
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Accuracy must be a finite fraction between 0 and 1")
    if test_size < 1:
        raise ValueError("Test size must be positive")
    return accuracy + 2 * math.sqrt(accuracy * (1 - accuracy) / test_size)


def promote(client, version: str, test_size: int, name: str = "food11"):
    candidate = client.get_model_version(name, str(version))
    champion = client.get_model_version_by_alias(name, "champion")
    if candidate.version == champion.version:
        raise ValueError("Candidate is already champion")
    if not candidate.run_id or not champion.run_id:
        raise ValueError("Both versions must reference their training runs")
    candidate_run = client.get_run(candidate.run_id)
    champion_run = client.get_run(champion.run_id)
    if candidate_run.info.status != "FINISHED" or champion_run.info.status != "FINISHED":
        raise ValueError("Both training runs must be FINISHED")
    try:
        candidate_accuracy = candidate_run.data.metrics["test_accuracy"]
        champion_accuracy = champion_run.data.metrics["test_accuracy"]
    except KeyError as error:
        raise ValueError("Both runs must log test_accuracy") from error
    promotion_threshold(candidate_accuracy, test_size)
    threshold = promotion_threshold(champion_accuracy, test_size)
    for run in (candidate_run, champion_run):
        recorded_size = run.data.params.get("test_size")
        if recorded_size is not None and int(recorded_size) != test_size:
            raise ValueError("Requested test size does not match the training run")
    candidate_dataset = candidate_run.data.params.get(
        "evaluation_dataset", candidate_run.data.params.get("dataset")
    )
    champion_dataset = champion_run.data.params.get(
        "evaluation_dataset", champion_run.data.params.get("dataset")
    )
    if candidate_dataset and champion_dataset and candidate_dataset != champion_dataset:
        raise ValueError("Candidate and champion must use the same evaluation dataset")
    decision = "promoted" if candidate_accuracy >= threshold else "challenger"
    tags = {
        "promotion.decision": decision,
        "promotion.candidate_accuracy": candidate_accuracy,
        "promotion.champion_accuracy": champion_accuracy,
        "promotion.previous_champion": champion.version,
        "promotion.test_size": test_size,
        "promotion.threshold": threshold,
        "promotion.margin": threshold - champion_accuracy,
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, str(value))
    client.set_registered_model_alias(
        name, "champion" if decision == "promoted" else "challenger", str(version)
    )
    if decision == "promoted" and "challenger" in candidate.aliases:
        client.delete_registered_model_alias(name, "challenger")
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    parser.add_argument("--test-size", type=int)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/food11_processed_mini/evaluation",
    )
    args = parser.parse_args()
    size = args.test_size
    if size is None:
        size = sum(
            1
            for p in args.test_dir.glob("*/*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    result = promote(MlflowClient(tracking_uri=args.tracking_uri), args.version, size)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
