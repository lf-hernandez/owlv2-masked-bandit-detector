import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from detector import Detector

detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = Detector()

    yield

    if detector and hasattr(detector, "model"):
        del detector.model
        del detector.processor
        if detector.device == "cuda":
            import torch

            torch.cuda.empty_cache()


app = FastAPI(
    title="Masked Bandit Detector API",
    description="Zero-shot raccoon detection using OWLv2",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Masked Bandit Detector API",
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST with image file)",
            "detect_and_draw": "/detect-and-draw (POST with image file)",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": detector.device if detector else "not initialized",
    }


@app.post("/detect")
def detect_raccoons(file: UploadFile = File(...), threshold: float = 0.1):
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))

        detections = detector.detect(image, threshold=threshold)

        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "image_size": list(image.size),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-and-draw")
def detect_and_draw(file: UploadFile = File(...), threshold: float = 0.1):
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))

        result = detector.detect_and_draw(image, threshold=threshold)

        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
