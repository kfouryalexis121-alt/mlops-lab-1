from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.food11 import serve


class FakeModel:
    def predict(self, inputs):
        assert inputs.shape == (1, 3, 128, 128)
        assert inputs.dtype == np.float32
        assert np.isfinite(inputs).all()
        # The real model returns logits; softmax gives exactly 0.90 for Bread.
        return np.log([[0.90] + [0.01] * 10])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    remote = Mock(side_effect=AssertionError("Unit tests must not contact MLflow"))
    monkeypatch.setattr(serve.mlflow, "MlflowClient", remote)
    monkeypatch.setattr(serve.mlflow.pyfunc, "load_model", remote)
    with TestClient(serve.app) as client:
        yield client


def test_health_without_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/predict", files={"file": ("x.jpg", b"x")}).status_code == 503


def test_predict(client, monkeypatch):
    monkeypatch.setattr(serve, "model", FakeModel())
    image = (Path(__file__).parent / "fixtures/bread.jpg").read_bytes()
    response = client.post("/predict", files={"file": ("bread.jpg", image, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["category"] == "Bread"
    assert response.json()["confidence"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "content, status",
    [(b"", 400), (b"not an image", 400), (b"x" * (serve.MAX_UPLOAD_BYTES + 1), 413)],
)
def test_invalid_upload(client, monkeypatch, content, status):
    monkeypatch.setattr(serve, "model", FakeModel())
    assert client.post("/predict", files={"file": ("x.jpg", content)}).status_code == status


def test_missing_upload(client):
    assert client.post("/predict").status_code == 422
