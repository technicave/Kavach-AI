"""
SIH26188 - Suraksha Kavach
Combined document screening endpoint: OCR/MRZ validation + tamper detection + face verification
"""

import os
import sys
import tempfile

# backend/app.py lives one level down from the project root, and face_matcher.py
# lives in face_analysis/ — add both to the path so the plain imports below work
# regardless of where uvicorn is launched from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "face_analysis"))
sys.path.append(os.path.join(PROJECT_ROOT, "aadhar_verify"))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2

from passport_ocr import analyze_document
from aadhaar_ocr import analyze_aadhaar
from aadhaar_tamper import predict_aadhaar_tamper
from predict import predict as predict_tamper
from face_matcher import FaceMatcher

app = FastAPI(title="Suraksha Kavach - Document Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[STARTUP] Loading FaceMatcher (insightface)...")
matcher = FaceMatcher(use_gpu=False)
print("[STARTUP] Ready.")


async def save_upload(file: UploadFile, suffix: str = ".jpg") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()
    return tmp.name


def compute_overall_risk(ocr_result: dict, tamper_result: dict, face_result: dict) -> dict:
    score = 0.0
    reasons = []

    ocr_issues = ocr_result.get("ocr_issues", [])
    if ocr_issues:
        score += 35
        reasons.extend(ocr_issues)
    if not ocr_result.get("mrz_parsed_successfully"):
        score += 15
        reasons.append("MRZ could not be parsed")

    if "error" not in tamper_result:
        tampered_prob = tamper_result.get("tampered_probability", 0)
        score += tampered_prob * 0.4
        if tampered_prob >= 70:
            reasons.append("High photo tamper probability")
    else:
        score += 20
        reasons.append(f"Tamper check failed: {tamper_result['error']}")

    if not face_result.get("match", False):
        score += 25
        reasons.append(face_result.get("message", "Face verification failed"))

    score = round(min(score, 100), 2)
    if score >= 70:
        level = "High Risk"
    elif score >= 35:
        level = "Medium Risk"
    else:
        level = "Low Risk"

    return {"overall_risk_score": score, "risk_level": level, "reasons": reasons}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/screen-document")
async def screen_document(
    doc_image: UploadFile = File(...),
    live_image: UploadFile = File(...)
):
    doc_path = await save_upload(doc_image)
    live_path = await save_upload(live_image)

    try:
        ocr_result = analyze_document(doc_path)
        tamper_result = predict_tamper(doc_path)

        doc_cv = cv2.imread(doc_path)
        live_cv = cv2.imread(live_path)
        face_result = matcher.verify(doc_cv, live_cv)

        overall = compute_overall_risk(ocr_result, tamper_result, face_result)

        return {
            "ocr_validation": ocr_result,
            "tamper_detection": tamper_result,
            "face_verification": face_result,
            "overall": overall,
        }
    finally:
        os.remove(doc_path)
        os.remove(live_path)


def compute_aadhaar_risk(aadhaar_result: dict, tamper_result: dict, face_result: dict) -> dict:
    score = 0.0
    reasons = []

    ocr_issues = aadhaar_result.get("ocr_issues", [])
    if ocr_issues:
        score += 30
        reasons.extend(ocr_issues)

    if not aadhaar_result.get("number_validation", {}).get("valid"):
        score += 25
        reasons.append("Aadhaar number failed Verhoeff validation")

    if "error" not in tamper_result:
        tampered_prob = tamper_result.get("tampered_probability", 0)
        score += tampered_prob * 0.4
        if tampered_prob >= 70:
            reasons.append("High photo tamper probability")
    else:
        score += 20
        reasons.append(f"Tamper check failed: {tamper_result['error']}")

    if not face_result.get("match", False):
        score += 25
        reasons.append(face_result.get("message", "Face verification failed"))

    score = round(min(score, 100), 2)
    level = "High Risk" if score >= 70 else "Medium Risk" if score >= 35 else "Low Risk"
    return {"overall_risk_score": score, "risk_level": level, "reasons": reasons}


@app.post("/screen-aadhaar")
async def screen_aadhaar(
    doc_image: UploadFile = File(...),
    live_image: UploadFile = File(...)
):
    doc_path = await save_upload(doc_image)
    live_path = await save_upload(live_image)

    try:
        aadhaar_result = analyze_aadhaar(doc_path)
        tamper_result = predict_aadhaar_tamper(doc_path)

        doc_cv = cv2.imread(doc_path)
        live_cv = cv2.imread(live_path)
        face_result = matcher.verify(doc_cv, live_cv)

        overall = compute_aadhaar_risk(aadhaar_result, tamper_result, face_result)

        return {
            "aadhaar_validation": aadhaar_result,
            "tamper_detection": tamper_result,
            "face_verification": face_result,
            "overall": overall,
        }
    finally:
        os.remove(doc_path)
        os.remove(live_path)
