import pytest
from PIL import Image

from src.food11.data import CLASSES, category_from_filename, resize_image


@pytest.mark.parametrize(
    "filename, expected",
    [("0_12.jpg", "Bread"), ("3_120.jpg", "Egg"), ("10_3.jpg", "Vegetable-Fruit")],
)
def test_category(filename, expected):
    assert category_from_filename(filename) == expected


@pytest.mark.parametrize("filename", ["missing.jpg", "11_0.jpg", "-1_0.jpg", "x_0.jpg"])
def test_invalid_category(filename):
    with pytest.raises(ValueError):
        category_from_filename(filename)


@pytest.mark.parametrize("mode", ["L", "RGBA", "RGB"])
def test_resize(mode):
    image = Image.new(mode, (512, 384))
    output = resize_image(image)
    assert output.size == (128, 128)
    assert output.mode == "RGB"
    assert image.size == (512, 384)
    assert image.mode == mode


def test_classes():
    assert len(CLASSES) == len(set(CLASSES)) == 11
