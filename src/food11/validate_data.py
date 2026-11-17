"""Validate the processed Food-11 dataset before training or promotion."""

import argparse
import hashlib
import math
from pathlib import Path

from PIL import Image

if __package__:
    from .data import CLASSES, IMAGE_SIZE, SPLITS
else:
    from data import CLASSES, IMAGE_SIZE, SPLITS


def validate_dataset(root: Path, min_per_class: int = 50, tau: float = 0.05):
    """Return errors for each contract check; an empty list means PASS."""
    if min_per_class < 1 or not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("min-per-class must be positive and tau must be between 0 and 1")
    root = Path(root)
    errors = {
        name: []
        for name in (
            "class folders",
            "image format",
            "minimum counts",
            "split leakage",
            "class balance",
        )
    }
    counts = {}
    seen = {}
    for split in SPLITS:
        folder = root / split
        actual = {p.name for p in folder.iterdir() if p.is_dir()} if folder.is_dir() else set()
        if actual != set(CLASSES):
            errors["class folders"].append(
                f"{split}: missing={sorted(set(CLASSES) - actual)}, extra={sorted(actual - set(CLASSES))}"
            )
        counts[split] = {}
        for category in CLASSES:
            class_dir = folder / category
            files = (
                sorted(p for p in class_dir.iterdir() if p.is_file()) if class_dir.is_dir() else []
            )
            counts[split][category] = len(files)
            if len(files) < min_per_class:
                errors["minimum counts"].append(
                    f"{split}/{category}: {len(files)} < {min_per_class}"
                )
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != IMAGE_SIZE:
                            errors["image format"].append(
                                f"{path.relative_to(root)}: {image.mode} {image.size}"
                            )
                    digest = hashlib.md5(path.read_bytes()).hexdigest()
                    if digest in seen and seen[digest][0] != split:
                        errors["split leakage"].append(
                            f"{seen[digest][1]} = {path.relative_to(root)}"
                        )
                    else:
                        seen.setdefault(digest, (split, path.relative_to(root)))
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    errors["image format"].append(f"{path.relative_to(root)}: {error}")
    train_total = sum(counts["training"].values())
    val_total = sum(counts["validation"].values())
    if not train_total or not val_total:
        errors["class balance"].append("Training and validation must both contain images")
    else:
        for category in CLASSES:
            difference = abs(
                counts["training"][category] / train_total
                - counts["validation"][category] / val_total
            )
            if difference > tau + 1e-12:
                errors["class balance"].append(
                    f"{category}: difference {difference:.4f} > {tau:.4f}"
                )
    return errors


def report(errors) -> bool:
    for name, details in errors.items():
        suffix = f" ({len(details)} issues): " + "; ".join(details[:3]) if details else ""
        print(f"{'FAIL' if details else 'PASS'} {name}{suffix}")
    return not any(errors.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    try:
        errors = validate_dataset(args.dataset, args.min_per_class, args.tau)
    except ValueError as error:
        parser.error(str(error))
    raise SystemExit(0 if report(errors) else 1)


if __name__ == "__main__":
    main()
