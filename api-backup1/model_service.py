from __future__ import annotations

import json
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, UnidentifiedImageError

from api import model_definition  # noqa: F401 - registers custom Keras layers.


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "concrete_crack.keras"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


class ModelNotFoundError(RuntimeError):
    pass


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True)
class EncodedImage:
    mime_type: str
    base64: str


@dataclass(frozen=True)
class Hotspot:
    x: int
    y: int
    width: int
    height: int
    score: float


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int


@dataclass(frozen=True)
class AnalysisResult:
    risk_level: str
    hotspot_count: int
    dominant_area: str


@dataclass(frozen=True)
class ExplainabilityResult:
    method: str
    overlay_image: EncodedImage
    heatmap_image: EncodedImage
    hotspots: list[Hotspot]


@dataclass(frozen=True)
class PredictionResult:
    label: str
    has_crack: bool
    confidence: float
    crack_probability: float
    threshold: float
    image_info: ImageInfo
    probabilities: dict[str, float]
    analysis: AnalysisResult
    explainability: ExplainabilityResult


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
        original_image = self._open_image(image_bytes)
        image_batch = self.preprocess_image(original_image)
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
        heatmap = self._build_saliency_heatmap(image_batch)
        hotspots = self._find_hotspots(heatmap, original_image.size, has_crack)
        explainability = self._build_explainability(
            original_image=original_image,
            heatmap=heatmap,
            hotspots=hotspots,
            has_crack=has_crack,
        )
        analysis = AnalysisResult(
            risk_level=self._risk_level(crack_probability, threshold),
            hotspot_count=len(hotspots),
            dominant_area=self._dominant_area(hotspots, original_image.size),
        )

        return PredictionResult(
            label=label,
            has_crack=has_crack,
            confidence=round(confidence, 6),
            crack_probability=round(crack_probability, 6),
            threshold=threshold,
            image_info=ImageInfo(
                width=original_image.width,
                height=original_image.height,
            ),
            probabilities={
                "cracked": round(crack_probability, 6),
                "non_cracked": round(1.0 - crack_probability, 6),
            },
            analysis=analysis,
            explainability=explainability,
        )

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        input_width, input_height = self.metadata["input_size"]
        resized_image = image.resize((input_width, input_height))
        image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)

    def _open_image(self, image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("Uploaded file is not a valid image.") from exc

    def _build_saliency_heatmap(self, image_batch: np.ndarray) -> np.ndarray:
        input_tensor = tf.convert_to_tensor(image_batch)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            prediction = self.model(input_tensor, training=False)
            crack_score = prediction[:, 0]

        gradients = tape.gradient(crack_score, input_tensor)
        if gradients is None:
            input_width, input_height = self.metadata["input_size"]
            return np.zeros((input_height, input_width), dtype=np.float32)

        heatmap = tf.reduce_mean(tf.abs(gradients), axis=-1)[0].numpy()
        heatmap = np.maximum(heatmap, 0)
        maximum = float(np.max(heatmap))
        if maximum <= 0:
            return np.zeros_like(heatmap, dtype=np.float32)
        return (heatmap / maximum).astype(np.float32)

    def _build_explainability(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        hotspots: list[Hotspot],
        has_crack: bool,
    ) -> ExplainabilityResult:
        heatmap_image = self._heatmap_to_image(heatmap, original_image.size)
        overlay_image = self._overlay_heatmap(original_image, heatmap_image, has_crack)

        if hotspots:
            draw = ImageDraw.Draw(overlay_image)
            for hotspot in hotspots:
                x1 = hotspot.x
                y1 = hotspot.y
                x2 = hotspot.x + hotspot.width
                y2 = hotspot.y + hotspot.height
                draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=3)
                draw.rectangle((x1, max(0, y1 - 20), min(original_image.width, x1 + 136), y1), fill=(180, 35, 24))
                draw.text((x1 + 6, max(1, y1 - 17)), "Likely crack area", fill=(255, 255, 255))

        return ExplainabilityResult(
            method="grad_cam_style_saliency",
            overlay_image=EncodedImage(
                mime_type="image/png",
                base64=self._image_to_base64(overlay_image),
            ),
            heatmap_image=EncodedImage(
                mime_type="image/png",
                base64=self._image_to_base64(heatmap_image),
            ),
            hotspots=hotspots,
        )

    def _heatmap_to_image(self, heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
        heatmap_uint8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        red = heatmap_uint8
        green = np.clip(255 - np.abs(heatmap_uint8.astype(np.int16) - 128) * 2, 0, 255).astype(np.uint8)
        blue = 255 - heatmap_uint8
        alpha = np.clip(heatmap_uint8, 45, 210).astype(np.uint8)
        rgba = np.dstack([red, green, blue, alpha])
        return Image.fromarray(rgba, mode="RGBA").resize(size, RESAMPLE_BILINEAR)

    def _overlay_heatmap(
        self,
        original_image: Image.Image,
        heatmap_image: Image.Image,
        has_crack: bool,
    ) -> Image.Image:
        base = original_image.convert("RGBA")
        alpha_scale = 0.48 if has_crack else 0.24
        heatmap = heatmap_image.copy()
        alpha = np.asarray(heatmap.getchannel("A"), dtype=np.float32)
        heatmap.putalpha(Image.fromarray(np.clip(alpha * alpha_scale, 0, 255).astype(np.uint8)))
        return Image.alpha_composite(base, heatmap)

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _find_hotspots(
        self,
        heatmap: np.ndarray,
        original_size: tuple[int, int],
        has_crack: bool,
    ) -> list[Hotspot]:
        if not has_crack or not np.any(heatmap):
            return []

        cutoff = max(0.58, float(np.percentile(heatmap, 88)))
        mask = heatmap >= cutoff
        visited = np.zeros(mask.shape, dtype=bool)
        height, width = mask.shape
        components: list[tuple[int, int, int, int, float, int]] = []

        for row in range(height):
            for column in range(width):
                if not mask[row, column] or visited[row, column]:
                    continue

                stack = [(row, column)]
                visited[row, column] = True
                rows: list[int] = []
                columns: list[int] = []
                scores: list[float] = []

                while stack:
                    current_row, current_column = stack.pop()
                    rows.append(current_row)
                    columns.append(current_column)
                    scores.append(float(heatmap[current_row, current_column]))

                    for next_row, next_column in (
                        (current_row - 1, current_column),
                        (current_row + 1, current_column),
                        (current_row, current_column - 1),
                        (current_row, current_column + 1),
                    ):
                        if (
                            0 <= next_row < height
                            and 0 <= next_column < width
                            and mask[next_row, next_column]
                            and not visited[next_row, next_column]
                        ):
                            visited[next_row, next_column] = True
                            stack.append((next_row, next_column))

                if len(rows) >= 3:
                    components.append(
                        (
                            min(columns),
                            min(rows),
                            max(columns),
                            max(rows),
                            float(np.mean(scores)),
                            len(rows),
                        )
                    )

        components.sort(key=lambda item: (item[5], item[4]), reverse=True)
        original_width, original_height = original_size
        x_scale = original_width / width
        y_scale = original_height / height
        hotspots: list[Hotspot] = []

        for min_x, min_y, max_x, max_y, score, _area in components[:3]:
            box_x = max(0, int(min_x * x_scale))
            box_y = max(0, int(min_y * y_scale))
            box_width = max(8, int((max_x - min_x + 1) * x_scale))
            box_height = max(8, int((max_y - min_y + 1) * y_scale))
            box_width = min(box_width, original_width - box_x)
            box_height = min(box_height, original_height - box_y)
            hotspots.append(
                Hotspot(
                    x=box_x,
                    y=box_y,
                    width=box_width,
                    height=box_height,
                    score=round(score, 4),
                )
            )

        return hotspots

    def _risk_level(self, crack_probability: float, threshold: float) -> str:
        if crack_probability < threshold:
            return "No Crack"
        if crack_probability < 0.75:
            return "Possible Crack"
        return "High Crack Likelihood"

    def _dominant_area(
        self,
        hotspots: list[Hotspot],
        image_size: tuple[int, int],
    ) -> str:
        if not hotspots:
            return "none"

        strongest = max(hotspots, key=lambda hotspot: hotspot.score)
        center_x = strongest.x + strongest.width / 2
        center_y = strongest.y + strongest.height / 2
        image_width, image_height = image_size
        horizontal = "left" if center_x < image_width / 3 else "right" if center_x > image_width * 2 / 3 else "center"
        vertical = "top" if center_y < image_height / 3 else "bottom" if center_y > image_height * 2 / 3 else "middle"

        if horizontal == "center" and vertical == "middle":
            return "center"
        if horizontal == "center":
            return vertical
        if vertical == "middle":
            return horizontal
        return f"{vertical}-{horizontal}"

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
