from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from api import model_definition  # noqa: F401 - registers custom Keras layers.


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "concrete_crack.keras"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"


class ModelNotFoundError(RuntimeError):
    pass


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True)
class PredictionResult:
    label: str
    has_crack: bool
    confidence: float
    crack_probability: float
    threshold: float


class CrackDetectionService:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata(metadata_path)
        self.model = self._load_model(model_path)

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image_batch = self.preprocess_image(image_bytes)
        prediction = self.model.predict(image_batch, verbose=0)
        crack_probability = float(np.asarray(prediction).reshape(-1)[0])
        threshold = float(self.metadata["threshold"])
        has_crack = crack_probability >= threshold
        label = (
            self.metadata["labels"]["1"]
            if has_crack
            else self.metadata["labels"]["0"]
        )
        confidence = crack_probability if has_crack else 1.0 - crack_probability

        return PredictionResult(
            label=label,
            has_crack=has_crack,
            confidence=round(confidence, 6),
            crack_probability=round(crack_probability, 6),
            threshold=threshold,
        )

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("Uploaded file is not a valid image.") from exc

        input_width, input_height = self.metadata["input_size"]
        image = image.resize((input_width, input_height))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)

    def _load_model(self, model_path: Path):
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model file not found at {model_path}. "
                "Run train_export_model.py first to create it."
            )
        return tf.keras.models.load_model(model_path, compile=False)

    def _load_metadata(self, metadata_path: Path) -> dict[str, Any]:
        if not metadata_path.exists():
            raise ModelNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Run train_export_model.py first to create it."
            )
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        required_keys = {"labels", "input_size", "threshold"}
        missing_keys = required_keys.difference(metadata)
        if missing_keys:
            raise ValueError(
                f"Metadata is missing required keys: {', '.join(sorted(missing_keys))}"
            )
        return metadata
