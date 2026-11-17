"""Serve the Food-11 champion model from the MLflow Model Registry."""

import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from .data import CATEGORIES, IMAGE_SIZE

MODEL_URI = "models:/food11@champion"
# ImageFolder assigns class indices in alphabetical directory order.
CLASS_NAMES = tuple(sorted(CATEGORIES))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load once; fail startup if the registered model cannot be fetched."""
    global model
    model = None
    app.state.model_version = None
    if not os.environ.get("SKIP_MODEL"):
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        version = mlflow.MlflowClient().get_model_version_by_alias("food11", "champion")
        model = mlflow.pyfunc.load_model(f"models:/food11/{version.version}")
        app.state.model_version = version.version
        logging.getLogger("uvicorn.error").info(
            "Loaded %s as version %s", MODEL_URI, version.version
        )
    try:
        yield
    finally:
        model = None
        app.state.model_version = None


app = FastAPI(title="Food-11 prediction API", lifespan=lifespan)


class Prediction(BaseModel):
    category: str
    confidence: float


def preprocess_image(content: bytes) -> np.ndarray:
    """Match Lab 1 resizing and Lab 2 ImageNet normalization."""
    try:
        with Image.open(BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="Upload a valid image.") from error

    tensor = (pixels.transpose(2, 0, 1) - MEAN) / STD
    return np.ascontiguousarray(tensor[None, ...], dtype=np.float32)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(file: Annotated[UploadFile, File()]) -> Prediction:
    """Return the category and its softmax score for one uploaded image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be at most 10 MiB.")

    inputs = preprocess_image(content)
    logits = np.asarray(model.predict(inputs), dtype=np.float32)
    if logits.shape != (1, len(CLASS_NAMES)) or not np.isfinite(logits).all():
        raise RuntimeError("Model returned invalid Food-11 predictions")

    scores = np.exp(logits[0] - logits[0].max())
    probabilities = scores / scores.sum()
    index = int(probabilities.argmax())
    return Prediction(category=CLASS_NAMES[index], confidence=float(probabilities[index]))
