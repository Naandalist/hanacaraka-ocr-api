"""Load and run TrOCR v6 against a cropped printed Javanese text line."""
from __future__ import annotations

import threading

from PIL import Image

from app.settings import settings


class OCRService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.processor = None
        self.model = None
        self.device = None

    def load(self) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(settings.model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(settings.model_id)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def pad_to_square(image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width == height:
            return image
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), image.getpixel((0, 0)))
        canvas.paste(image, (0, (side - height) // 2))
        return canvas

    def recognize(self, image: Image.Image) -> str:
        import torch

        if self.processor is None or self.model is None:
            raise RuntimeError("OCR model is not loaded")
        prepared = self.pad_to_square(image)
        pixel_values = self.processor(images=prepared, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        tokenizer = self.processor.tokenizer
        eos_id = tokenizer.sep_token_id
        if eos_id is None:
            eos_id = tokenizer.eos_token_id

        # One inference at a time for predictable memory usage on a small VPS.
        with self._lock, torch.inference_mode():
            ids = self.model.generate(
                pixel_values,
                max_new_tokens=settings.generation_max_new_tokens,
                num_beams=settings.num_beams,
                do_sample=False,
                decoder_start_token_id=tokenizer.cls_token_id,
                eos_token_id=eos_id,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=0,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


ocr_service = OCRService()
