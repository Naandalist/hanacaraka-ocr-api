# Hanacaraka OCR API

FastAPI + Hugging Face [TrOCR Javanese Synthetic v6](https://huggingface.co/thesimonharms/trocr-javanese-synthetic-v6).

Recognizes **one short, cropped line of printed Aksara Jawa** per request and returns Unicode Javanese. It does not detect lines, read full pages or handwriting, or transliterate to Latin. The publisher's 96% exact-match score comes from held-out **synthetic** images, not camera photos.

## Start locally (Python 3.11)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export OCR_API_KEY='replace-with-long-random-secret'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

At first startup the model is downloaded from Hugging Face (~247 MB safetensors, ~505 MB repository total). It is cached after that. Use one Uvicorn worker so the model is loaded only once. CPU inference may take several seconds or longer, depending on hardware.

### Docker Compose

```bash
cp .env.example .env
# Change OCR_API_KEY in .env first
docker compose up --build -d
```

The Compose service binds only to 127.0.0.1:8000. Put an HTTPS reverse proxy with rate limiting and proper client authentication in front of it.

## API

- `GET /health` — status when model has loaded.
- `POST /v1/ocr` — multipart image field `file`, header `x-api-key`. Accepts PNG/JPEG/WebP, max 5 MiB / 12 MP.

```bash
curl http://127.0.0.1:8000/v1/ocr \
  -H 'x-api-key: YOUR_KEY' \
  -F 'file=@cropped-line.png'
```

Response shape (example, not a guarantee of model accuracy):

```json
{"script":"ꦲꦤꦕꦫꦏ","model":"thesimonharms/trocr-javanese-synthetic-v6"}
```

Input should be cropped to one short text line. Preprocessing pads to a square using background sampled from the top-left pixel and centers the crop vertically before TrOCR's 384×384 resize. The model uses `cls_token_id` to start decoding, per the model card.

### React Native

Send `FormData` with an image upload named `file`; receive `script` as Javanese Unicode. Handle Latin transliteration in the existing app. Do **not** treat a hard-coded API key in a public app as secure authentication; implement per-user tokens or a backend gateway for public deployment.

## Tests

```bash
pip install pytest httpx
python -m pytest -q
```

Tests stub inference and do not validate the real checkpoint. Perform a live cropped-line test after installing dependencies and downloading model weights. The repository does **not** contain a pretrained model binary.

## Model

Author: Simon Harms; license: MIT. The v6 model is distinct from the older `aksara-ts` CRNN ONNX model.
