"""Evaluate a registered model on a validated dataset and log its gate inputs."""

import argparse
import os

import mlflow
import numpy as np
import torch

if __package__:
    from .train import DATASETS, create_dataloaders
    from .validate_data import report, validate_dataset
else:
    from train import DATASETS, create_dataloaders
    from validate_data import report, validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="Defaults to the champion version")
    parser.add_argument("--dataset", choices=DATASETS, default="lab5")
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    args = parser.parse_args()
    root = DATASETS[args.dataset]
    if not report(validate_dataset(root)):
        raise ValueError("Dataset contract failed; evaluation has not started")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    version = (
        client.get_model_version("food11", args.version)
        if args.version
        else client.get_model_version_by_alias("food11", "champion")
    )
    if not version.run_id:
        raise ValueError("Model version must reference a run")
    torch.set_num_threads(4)
    model = mlflow.pyfunc.load_model(f"models:/food11/{version.version}")
    # The saved PT2 model was exported with a one-image input example.
    loaders, classes = create_dataloaders(root, 1, 0, 42)
    correct = 0
    for images, labels in loaders["test"]:
        logits = np.asarray(model.predict(images.numpy()))
        if logits.shape != (1, len(classes)) or not np.isfinite(logits).all():
            raise ValueError("Invalid model output")
        correct += int((logits.argmax(axis=1) == labels.numpy()).sum())
    size = len(loaders["test"].dataset)
    client.log_param(version.run_id, "test_size", size)
    client.log_param(version.run_id, "evaluation_dataset", args.dataset)
    client.log_metric(version.run_id, "test_accuracy", correct / size)
    print(f"Version {version.version}: test_accuracy={correct / size:.8f}, test_size={size}")


if __name__ == "__main__":
    main()
