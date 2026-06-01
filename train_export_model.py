from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from api.model_definition import build_enhanced_densenet


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_SIZE = (64, 64)
THRESHOLD = 0.5


def load_images(non_cracked_dir: Path, cracked_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data: list[np.ndarray] = []
    labels: list[int] = []

    for label, folder in ((0, non_cracked_dir), (1, cracked_dir)):
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        for image_path in sorted(folder.rglob("*")):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                print(f"Skipping unreadable image: {image_path}")
                continue

            image = image.resize(INPUT_SIZE)
            data.append(np.asarray(image, dtype=np.float32))
            labels.append(label)

    if not data:
        raise ValueError("No supported images found in the provided dataset folders.")

    images = np.asarray(data, dtype=np.float32) / 255.0
    image_labels = np.asarray(labels, dtype=np.float32)
    return images, image_labels


def write_metadata(metadata_path: Path) -> None:
    metadata = {
        "labels": {
            "0": "Non-Cracked",
            "1": "Cracked",
        },
        "input_size": list(INPUT_SIZE),
        "channels": 3,
        "threshold": THRESHOLD,
        "preprocessing": {
            "color_mode": "RGB",
            "resize": list(INPUT_SIZE),
            "normalization": "pixel_value / 255.0",
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the concrete crack detection Keras model."
    )
    parser.add_argument(
        "--non-cracked-dir",
        required=True,
        type=Path,
        help="Folder containing non-cracked concrete images.",
    )
    parser.add_argument(
        "--cracked-dir",
        required=True,
        type=Path,
        help="Folder containing cracked concrete images.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet weights, or 'none' to train without downloading weights.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/concrete_crack.keras"),
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("models/metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, labels = load_images(args.non_cracked_dir, args.cracked_dir)
    stratify = labels if len(set(labels.tolist())) == 2 else None
    train_images, validation_images, train_labels, validation_labels = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=43,
        shuffle=True,
        stratify=stratify,
    )

    weights = None if args.weights == "none" else args.weights
    model = build_enhanced_densenet(weights=weights)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_images,
        train_labels,
        validation_data=(validation_images, validation_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    write_metadata(args.metadata_out)
    print(f"Saved model to {args.model_out}")
    print(f"Saved metadata to {args.metadata_out}")


if __name__ == "__main__":
    main()
