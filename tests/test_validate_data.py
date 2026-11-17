import shutil
import subprocess
import sys

import pytest
from PIL import Image

from src.food11.data import CLASSES, SPLITS
from src.food11.validate_data import validate_dataset


@pytest.fixture
def dataset(tmp_path):
    for split_index, split in enumerate(SPLITS):
        for index, category in enumerate(CLASSES):
            folder = tmp_path / split / category
            folder.mkdir(parents=True)
            Image.new("RGB", (128, 128), (split_index * 60 + index, 20, 30)).save(
                folder / "image.png"
            )
    return tmp_path


def test_valid_contract(dataset):
    assert not any(validate_dataset(dataset, 1).values())


def test_cross_split_duplicate(dataset):
    shutil.copyfile(dataset / "training/Bread/image.png", dataset / "evaluation/Bread/leak.png")
    result = validate_dataset(dataset, 1)
    assert result["split leakage"]
    assert not result["image format"]
    command = subprocess.run(
        [sys.executable, "-m", "src.food11.validate_data", str(dataset), "--min-per-class", "1"],
        capture_output=True,
        text=True,
    )
    assert command.returncode == 1
    assert "FAIL split leakage" in command.stdout


def test_corrupt_image(dataset):
    (dataset / "training/Bread/image.png").write_bytes(b"invalid")
    assert validate_dataset(dataset, 1)["image format"]


def test_wrong_mode_and_size(dataset):
    Image.new("L", (64, 64)).save(dataset / "training/Bread/image.png")
    assert validate_dataset(dataset, 1)["image format"]


def test_class_structure_and_counts(dataset):
    (dataset / "training/extra").mkdir()
    (dataset / "validation/Bread/image.png").unlink()
    result = validate_dataset(dataset, 1)
    assert result["class folders"] and result["minimum counts"]


def test_class_balance(dataset):
    for i in range(10):
        Image.new("RGB", (128, 128), (240, i, 0)).save(dataset / f"training/Bread/{i}.png")
    assert validate_dataset(dataset, 1)["class balance"]


def test_missing_dataset(tmp_path):
    result = validate_dataset(tmp_path / "missing", 1)
    assert result["class folders"] and result["minimum counts"] and result["class balance"]
