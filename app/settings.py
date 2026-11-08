"""Runtime configuration."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("MODEL_ID", "thesimonharms/trocr-javanese-synthetic-v6")
    api_key: str = os.getenv("OCR_API_KEY", "")
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "12000000"))
    generation_max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "24"))
    num_beams: int = int(os.getenv("NUM_BEAMS", "4"))


settings = Settings()
