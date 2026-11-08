"""HTTP API for short, cropped printed Javanese script lines."""
from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import secrets

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

from app.inference import ocr_service
from app.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail startup if model files cannot be downloaded or loaded.
    await run_in_threadpool(ocr_service.load)
    yield


app = FastAPI(title="Hanacaraka OCR API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.model_id}


@app.post("/v1/ocr")
async def recognize(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    if not settings.api_key or not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Expected a JPEG, PNG, or WebP image")

    data = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds upload limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
        with Image.open(BytesIO(data)) as img:
            if img.width * img.height > settings.max_image_pixels:
                raise HTTPException(status_code=413, detail="Image resolution too large")
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
            rgb.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Invalid image") from None

    script = await run_in_threadpool(ocr_service.recognize, rgb)
    return {"script": script, "model": settings.model_id}
