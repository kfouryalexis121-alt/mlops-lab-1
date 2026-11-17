"""Exercise the upload page against the running inference container."""

from io import BytesIO
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


@unittest.skipUnless(os.environ.get("FOOD11_LIVE_TEST_IMAGE"), "Needs a running API and sample image")
class LiveFrontendTests(unittest.TestCase):
    def test_upload_prediction(self):
        image = BytesIO(Path(os.environ["FOOD11_LIVE_TEST_IMAGE"]).read_bytes())
        image.name = "food.jpg"
        image.type = "image/jpeg"
        with patch("streamlit.file_uploader", return_value=image):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.run()
            app.button[0].click().run(timeout=90)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.success), 1)
        self.assertIn("Prediction:", app.success[0].value)
        print(app.success[0].value)


if __name__ == "__main__":
    unittest.main()
