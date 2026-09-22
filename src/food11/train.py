"""Train a transfer-learning classifier for the processed Food-11 dataset."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DATASETS = {
    "processed": DATA_ROOT / "food11_processed",
    "mini": DATA_ROOT / "food11_processed_mini",
}
SPLITS = {
    "train": "training",
    "validation": "validation",
    "test": "evaluation",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for an argparse option."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    """Parse a strictly positive float for an argparse option."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    """Read training hyperparameters from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="mini")
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--lr", type=positive_float, default=0.001)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        help="MLflow tracking server URI (default: %(default)s)",
    )
    parser.add_argument("--experiment", default="food11")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Make data shuffling and model initialization reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_utf8_output() -> None:
    """Allow MLflow's Unicode run links to print in Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def create_dataloaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, DataLoader], list[str]]:
    """Create loaders for the Food-11 training, validation, and test splits."""
    missing = [split for split in SPLITS.values() if not (dataset_root / split).is_dir()]
    if missing:
        missing_paths = ", ".join(str(dataset_root / split) for split in missing)
        raise FileNotFoundError(f"Missing dataset split(s): {missing_paths}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    image_datasets = {
        name: datasets.ImageFolder(dataset_root / folder, transform=transform)
        for name, folder in SPLITS.items()
    }

    class_names = image_datasets["train"].classes
    for name, dataset in image_datasets.items():
        if dataset.classes != class_names:
            raise ValueError(f"Class ordering in {name} does not match the training split")

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if name == "train" else None,
        )
        for name, dataset in image_datasets.items()
    }
    return loaders, class_names


def create_model(num_classes: int) -> nn.Module:
    """Create a pretrained ResNet-18 with a Food-11 classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one train/evaluation epoch and return mean loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train(args: argparse.Namespace) -> None:
    """Train and evaluate one model, recording the run in MLflow."""
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    seed_everything(args.seed)
    dataset_root = DATASETS[args.dataset]
    loaders, class_names = create_dataloaders(
        dataset_root, args.batch_size, args.num_workers, args.seed
    )
    if len(class_names) != 11:
        raise ValueError(f"Expected 11 classes, found {len(class_names)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "optimizer": "Adam",
                "seed": args.seed,
                "device": device.type,
                "pretrained_weights": "ResNet18_Weights.DEFAULT",
            }
        )

        for epoch in range(args.epochs):
            train_loss, train_accuracy = run_epoch(
                model, loaders["train"], loss_fn, device, optimizer
            )
            val_loss, val_accuracy = run_epoch(
                model, loaders["validation"], loss_fn, device
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_accuracy={val_accuracy:.4f}"
            )

        test_loss, test_accuracy = run_epoch(model, loaders["test"], loss_fn, device)
        mlflow.log_metrics(
            {"test_loss": test_loss, "test_accuracy": test_accuracy},
            step=args.epochs - 1,
        )
        model = model.to("cpu").eval()
        input_example = np.zeros((1, 3, 128, 128), dtype=np.float32)
        mlflow.pytorch.log_model(model, name="model", input_example=input_example)
        print(f"Test loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}")


def main() -> None:
    """CLI entry point."""
    configure_utf8_output()
    train(parse_args())


if __name__ == "__main__":
    main()
