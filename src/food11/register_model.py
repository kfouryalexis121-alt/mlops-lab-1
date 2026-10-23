"""Register a completed Food-11 run and assign the champion alias."""

import argparse
import os

import mlflow
from mlflow import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    run = client.get_run(args.run_id)
    if run.info.status != "FINISHED":
        raise ValueError("Choose a successfully completed training run")

    # Reuse an existing version when repeating this setup command.
    versions = client.search_model_versions("name = 'food11'")
    matches = [version for version in versions if version.run_id == args.run_id]
    if matches:
        version = max(matches, key=lambda item: int(item.version))
    else:
        version = mlflow.register_model(f"runs:/{args.run_id}/model", "food11")
    client.set_registered_model_alias("food11", "champion", version.version)
    print(f"food11 version {version.version}: champion -> run {args.run_id}")


if __name__ == "__main__":
    main()
