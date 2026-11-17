"""Run inside the frontend environment with unittest discovery."""

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import requests
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class FrontendTests(unittest.TestCase):
    def run_upload(self, response=None, error=None):
        image = BytesIO()
        Image.new("RGB", (32, 32)).save(image, format="PNG")
        image.seek(0)
        image.name = "food.png"
        image.type = "image/png"
        with patch("streamlit.file_uploader", return_value=image), patch(
            "requests.post", return_value=response, side_effect=error
        ) as post:
            app = AppTest.from_file(str(APP)).run()
            self.assertFalse(app.exception)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(post.call_args.kwargs["timeout"], (5, 60))
            self.assertEqual(post.call_args.kwargs["files"]["file"][0], "food.png")
        return app

    def test_prediction(self):
        response = Mock()
        response.json.return_value = {"category": "Bread", "confidence": 0.9}
        app = self.run_upload(response=response)
        self.assertIn("Bread (90.0%)", app.success[0].value)

    def test_unavailable_service(self):
        app = self.run_upload(error=requests.Timeout())
        self.assertIn("unavailable", app.error[0].value)

    def test_invalid_upload_response(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError()
        app = self.run_upload(response=response)
        self.assertIn("another JPG or PNG", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
