import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os
from typing import Tuple, Optional, Dict


def get_photo_region(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return tuple(faces[0])

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h else 0
        area = w * h
        if 0.65 < aspect < 0.95 and area > (h_img * w_img * 0.03):
            candidates.append((x, y, w, h, area))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda c: c[4], reverse=True)
    x, y, w, h, _ = candidates[0]
    return (x, y, w, h)


def ela_score_from_array(image_bgr: np.ndarray, quality: int = 90, scale: int = 25) -> float:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original = Image.fromarray(rgb)
    temp_path = "_temp_ela.jpg"
    original.save(temp_path, 'JPEG', quality=quality)
    compressed = Image.open(temp_path)
    ela = ImageChops.difference(original, compressed)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela = ImageEnhance.Brightness(ela).enhance(scale / max_diff)
    ela_np = np.array(ela)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
    return round(min(100.0, float(np.mean(gray) * 2.2)), 2)


REFERENCE_SIZE = (150, 150)  # fixed size all crops are normalized to before feature
                              # computation, so image/photo resolution stops being a
                              # hidden confound. Confirmed necessary: real test images
                              # ranged from 290x444 to 1748x1240 px, and Laplacian
                              # variance / edge density are inherently resolution-
                              # sensitive, which was causing false positives on
                              # otherwise-genuine documents simply because they were
                              # a different resolution than the 2 base images used
                              # to build the training set.


def _normalize_crop(region: 'np.ndarray') -> 'np.ndarray':
    if region.size == 0:
        return region
    return cv2.resize(region, REFERENCE_SIZE, interpolation=cv2.INTER_AREA)


def edge_score(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    pad = 12
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = _normalize_crop(gray)
    edges = cv2.Canny(gray, 40, 120)
    density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1] + 1e-5)
    return round(float(min(100, density * 380)), 2)


def noise_texture_score(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    photo = _normalize_crop(gray[y:y + h, x:x + w])
    pad = 18
    h_img, w_img = gray.shape
    y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
    x1, x2 = max(0, x - pad), min(w_img, x + w + pad)
    surrounding = gray[y1:y2, x1:x2].copy()
    surrounding[y - y1:y - y1 + h, x - x1:x - x1 + w] = 0
    surrounding = _normalize_crop(surrounding)

    def noise_std(region, mask=None):
        denoised = cv2.medianBlur(region, 3)
        residual = cv2.absdiff(region, denoised).astype(np.float32)
        vals = residual[mask] if mask is not None else residual
        return float(np.std(vals)) if vals.size >= 50 else 0.0

    p_noise = noise_std(photo)
    s_noise = noise_std(surrounding, surrounding > 0)
    if s_noise == 0:
        return 0.0
    return round(float(min(100.0, abs(p_noise - s_noise) / s_noise * 70)), 2)


def sharpness_score(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    photo = _normalize_crop(gray[y:y + h, x:x + w])
    pad = 20
    h_img, w_img = gray.shape
    y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
    x1, x2 = max(0, x - pad), min(w_img, x + w + pad)
    surrounding = gray[y1:y2, x1:x2].copy()
    surrounding[y - y1:y - y1 + h, x - x1:x - x1 + w] = 0
    surrounding = _normalize_crop(surrounding)

    def lap_var(region):
        mask = region > 0
        return cv2.Laplacian(region, cv2.CV_64F).var() if np.sum(mask) >= 50 else 0

    p_sharp = lap_var(photo)
    s_sharp = lap_var(surrounding)
    if s_sharp == 0:
        return 0.0
    return round(float(min(100.0, abs(p_sharp - s_sharp) / s_sharp * 60)), 2)


def extract_features(image_bgr: np.ndarray) -> Optional[Dict[str, float]]:
    """Returns None if no photo region could be located."""
    bbox = get_photo_region(image_bgr)
    if bbox is None:
        return None
    return {
        "ela": ela_score_from_array(image_bgr),
        "edge": edge_score(image_bgr, bbox),
        "noise_texture": noise_texture_score(image_bgr, bbox),
        "sharpness": sharpness_score(image_bgr, bbox),
    }
