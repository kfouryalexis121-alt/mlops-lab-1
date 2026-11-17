"""Create a separate deduplicated mini dataset, preserving the original files."""

import argparse
import hashlib
import shutil
from pathlib import Path

if __package__:
    from .data import CLASSES
    from .validate_data import report, validate_dataset
else:
    from data import CLASSES
    from validate_data import report, validate_dataset


def prepare(source: Path, destination: Path) -> int:
    if not source.is_dir():
        raise ValueError(f"Source dataset does not exist: {source}")
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}; choose a new directory")
    seen = {}
    removed = 0
    # Keep the test split intact; remove cross-split duplicates from later splits.
    for split in ("evaluation", "validation", "training"):
        for category in CLASSES:
            target = destination / split / category
            target.mkdir(parents=True)
            for image in sorted((source / split / category).iterdir()):
                if not image.is_file():
                    continue
                digest = hashlib.md5(image.read_bytes()).hexdigest()
                if digest in seen and seen[digest] != split:
                    removed += 1
                    continue
                seen.setdefault(digest, split)
                shutil.copy2(image, target / image.name)
    print(f"Excluded {removed} cross-split copies; original dataset unchanged")
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/food11_processed_mini"))
    parser.add_argument("--output", type=Path, default=Path("data_local/food11_lab5"))
    args = parser.parse_args()
    prepare(args.source, args.output)
    raise SystemExit(0 if report(validate_dataset(args.output)) else 1)


if __name__ == "__main__":
    main()
