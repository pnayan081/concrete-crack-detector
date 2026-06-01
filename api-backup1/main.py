from __future__ import annotations

import base64
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.model_service import (
    CrackDetectionService,
    InvalidImageError,
    ModelNotFoundError,
)


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FRONTEND_INDEX = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
SUPPORTED_MIME_TYPES = {
    "application/octet-stream",
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/x-ms-bmp",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.crack_service = CrackDetectionService()
    except ModelNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    yield


app = FastAPI(
    title="Concrete Crack Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index():
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend page not found.")
    return FileResponse(FRONTEND_INDEX)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a jpg, jpeg, png, or bmp image.",
        )

    if file.content_type and file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported content type. Upload an image file.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = app.state.crack_service.predict(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mime_type = file.content_type
    if not mime_type or mime_type == "application/octet-stream":
        mime_type = mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"

    return {
        "filename": file.filename,
        "label": result.label,
        "has_crack": result.has_crack,
        "confidence": result.confidence,
        "crack_probability": result.crack_probability,
        "threshold": result.threshold,
        "image": {
            "mime_type": mime_type,
            "base64": base64.b64encode(image_bytes).decode("ascii"),
        },
        "image_info": {
            "width": result.image_info.width,
            "height": result.image_info.height,
            "file_size_kb": round(len(image_bytes) / 1024, 2),
            "mime_type": mime_type,
        },
        "probabilities": result.probabilities,
        "analysis": {
            "risk_level": result.analysis.risk_level,
            "hotspot_count": result.analysis.hotspot_count,
            "dominant_area": result.analysis.dominant_area,
        },
        "explainability": {
            "method": result.explainability.method,
            "overlay_image": {
                "mime_type": result.explainability.overlay_image.mime_type,
                "base64": result.explainability.overlay_image.base64,
            },
            "heatmap_image": {
                "mime_type": result.explainability.heatmap_image.mime_type,
                "base64": result.explainability.heatmap_image.base64,
            },
            "hotspots": [
                {
                    "x": hotspot.x,
                    "y": hotspot.y,
                    "width": hotspot.width,
                    "height": hotspot.height,
                    "score": hotspot.score,
                }
                for hotspot in result.explainability.hotspots
            ],
        },
    }
