"""Prepare the Food-11 dataset for image-classification training.

The raw dataset stores labels in filenames such as ``0_123.jpg``. This
script resizes every image to 128x128 and places it in the class-directory
layout expected by common image-classification data loaders.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


CATEGORIES = (
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
)
SPLITS = ("training", "evaluation", "validation")
IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "food11_raw"
PROCESSED_ROOT = DATA_ROOT / "food11_processed"
MINI_ROOT = DATA_ROOT / "food11_processed_mini"


def category_from_filename(image_path: Path) -> str:
    """Return the category encoded at the start of a Food-11 filename."""
    filename_parts = image_path.stem.split("_", maxsplit=1)
    try:
        if len(filename_parts) != 2:
            raise ValueError
        label = int(filename_parts[0])
        if not 0 <= label < len(CATEGORIES):
            raise ValueError
    except ValueError as error:
        raise ValueError(f"Invalid Food-11 filename: {image_path.name}") from error
    return CATEGORIES[label]


def create_category_directories(root: Path, split: str) -> None:
    """Create every class directory, including classes with no images."""
    for category in CATEGORIES:
        (root / split / category).mkdir(parents=True, exist_ok=True)


def resize_image(source: Path, destination: Path) -> None:
    """Resize one image and save it as an RGB JPEG."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=90, optimize=True)


def process_split(split: str) -> tuple[int, int]:
    """Create the full and mini processed datasets for one data split."""
    source_dir = RAW_ROOT / split
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing raw split directory: {source_dir}")

    create_category_directories(PROCESSED_ROOT, split)
    create_category_directories(MINI_ROOT, split)

    mini_counts: defaultdict[str, int] = defaultdict(int)
    processed_count = 0

    image_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )

    for source in image_paths:
        category = category_from_filename(source)
        processed_path = PROCESSED_ROOT / split / category / source.name
        resize_image(source, processed_path)
        processed_count += 1

        if mini_counts[category] < MINI_LIMIT:
            mini_path = MINI_ROOT / split / category / source.name
            shutil.copy2(processed_path, mini_path)
            mini_counts[category] += 1

    return processed_count, sum(mini_counts.values())


def main() -> None:
    """Rebuild both processed Food-11 datasets from the raw images."""
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_ROOT}. "
            "Expected training, evaluation, and validation folders there."
        )

    for output_root in (PROCESSED_ROOT, MINI_ROOT):
        if output_root.exists():
            shutil.rmtree(output_root)

    total_processed = 0
    total_mini = 0
    for split in SPLITS:
        processed_count, mini_count = process_split(split)
        total_processed += processed_count
        total_mini += mini_count
        print(f"{split}: processed {processed_count}, mini {mini_count}")

    print(f"Done: processed {total_processed} images; mini contains {total_mini} images")


if __name__ == "__main__":
    main()
