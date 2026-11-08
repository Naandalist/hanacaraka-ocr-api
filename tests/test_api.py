"""API unit tests with mocked model inference (no downloaded weights)."""
from dataclasses import replace
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.inference import OCRService, ocr_service
from app.main import app
from app.settings import settings


def test_pad_to_square_preserves_aspect_ratio():
    image = Image.new("RGB", (320, 50), "white")
    padded = OCRService.pad_to_square(image)
    assert padded.size == (320, 320)
    assert padded.getpixel((0, 135)) == (255, 255, 255)


def test_ocr_authentication(monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "settings", replace(settings, api_key="test-secret"))
    monkeypatch.setattr(ocr_service, "recognize", lambda image: "ꦲ")

    image = Image.new("RGB", (128, 32), "white")
    output = BytesIO()
    image.save(output, format="PNG")
    payload = {"file": ("line.png", output.getvalue(), "image/png")}

    # No lifespan context manager: mock inference and avoid model downloads.
    client = TestClient(app)
    assert client.post("/v1/ocr", files=payload).status_code == 401
    response = client.post("/v1/ocr", files=payload, headers={"x-api-key": "test-secret"})
    assert response.status_code == 200
    assert response.json()["script"] == "ꦲ"
