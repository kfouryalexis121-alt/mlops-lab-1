"""Copy a saved MLflow model into a new tracking server for Lab 4."""

import argparse
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import mlflow
from mlflow import MlflowClient
import yaml


def main() -> None:
    # MLflow's decorative status emojis fail on some Windows consoles.
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--source-run-id", required=True)
    args = parser.parse_args()
    if not (args.model_dir / "MLmodel").is_file():
        parser.error("--model-dir must contain an MLmodel file and its artifacts")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    # Repeating setup must not overwrite an existing promoted champion.
    versions = client.search_model_versions("name = 'food11'")
    if versions:
        print("food11 already has registered versions; leaving the registry unchanged.")
        return
    mlflow.set_experiment("food11-compose")
    with mlflow.start_run(run_name="Import Lab 2 model") as run:
        mlflow.set_tag("source_run_id", args.source_run_id)
        with TemporaryDirectory(prefix="food11-import-") as temporary:
            copied_model = Path(temporary) / "model"
            shutil.copytree(args.model_dir, copied_model)
            metadata_path = copied_model / "MLmodel"
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
            # The original logged-model ID and Windows location belong to the
            # source server. Keep the model files but give the copy local lineage.
            source_model_id = metadata.pop("model_id", None)
            if source_model_id:
                mlflow.set_tag("source_model_id", source_model_id)
            metadata["run_id"] = run.info.run_id
            metadata["artifact_path"] = f"{run.info.artifact_uri}/model"
            metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
            mlflow.log_artifacts(str(copied_model), artifact_path="model")
        run_id = run.info.run_id
    version = mlflow.register_model(f"runs:/{run_id}/model", "food11")
    client.set_registered_model_alias("food11", "champion", version.version)
    print(f"Imported food11 version {version.version}; champion assigned.")


if __name__ == "__main__":
    main()
