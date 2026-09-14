# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2

from face_matcher import FaceMatcher  # your existing file

app = FastAPI()

# Load model ONCE at startup, not per-request — insightface init is slow
matcher = FaceMatcher(use_gpu=False)

async def read_image(file: UploadFile) -> np.ndarray:
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

@app.post("/verify-face")
async def verify_face(
    doc_image: UploadFile = File(...),
    live_image: UploadFile = File(...)
):
    doc_img = await read_image(doc_image)
    live_img = await read_image(live_image)

    if doc_img is None or live_img is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode one or both images"})

    result = matcher.verify(doc_img, live_img)
    return result
