"""Integration checks for a running local or containerized Food-11 API.

Start MLflow and the API first, then run:
    python -m unittest discover -s tests -v
Override FOOD11_API_URL to check a different host or port.
"""

from io import BytesIO
import json
import os
from pathlib import Path
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from src.food11.data import CATEGORIES


BASE_URL = os.environ.get("FOOD11_API_URL", "http://127.0.0.1:8000").rstrip("/")
ROOT = Path(__file__).resolve().parents[1]


def upload(content: bytes, filename: str = "image.jpg") -> tuple[int, dict]:
    boundary = "food11-integration-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    request = Request(
        f"{BASE_URL}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


class Food11ApiTests(unittest.TestCase):
    def assert_prediction(self, status: int, result: dict) -> None:
        self.assertEqual(status, 200, result)
        self.assertIn(result["category"], CATEGORIES)
        self.assertGreaterEqual(result["confidence"], 1 / len(CATEGORIES) - 1e-6)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_health(self):
        with urlopen(f"{BASE_URL}/health", timeout=10) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.load(response), {"status": "ok"})

    def test_dataset_image_and_repeat_prediction(self):
        image = next((ROOT / "data/food11_processed_mini/validation/Bread").glob("*.jpg"))
        status, first = upload(image.read_bytes())
        self.assert_prediction(status, first)
        status, second = upload(image.read_bytes())
        self.assertEqual(status, 200)
        self.assertEqual(first, second)

    def test_grayscale_and_rgba_images(self):
        for mode in ("L", "RGBA"):
            with self.subTest(mode=mode):
                content = BytesIO()
                Image.new(mode, (72, 180)).save(content, format="PNG")
                self.assert_prediction(*upload(content.getvalue(), "image.png"))

    def test_invalid_and_empty_images(self):
        for content in (b"this is not an image", b""):
            with self.subTest(content=content):
                status, result = upload(content)
                self.assertEqual(status, 400, result)

    def test_missing_file(self):
        request = Request(f"{BASE_URL}/predict", data=b"", method="POST")
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 422)
        caught.exception.close()

    def test_oversized_upload(self):
        status, result = upload(b"x" * (10 * 1024 * 1024 + 1))
        self.assertEqual(status, 413, result)


if __name__ == "__main__":
    unittest.main()
