import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os
from typing import Dict, Tuple, Optional


class PhotoReplacementDetector:
    def __init__(self, ela_quality: int = 90, ela_scale: int = 25):
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale

    def _perform_ela(self, image_path: str) -> Tuple[np.ndarray, float]:
        original = Image.open(image_path).convert('RGB')

        temp_path = "temp_ela.jpg"
        original.save(temp_path, 'JPEG', quality=self.ela_quality)
        compressed = Image.open(temp_path)

        ela = ImageChops.difference(original, compressed)

        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        scale_factor = self.ela_scale / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale_factor)

        ela_np = np.array(ela)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
        ela_score = min(100, float(np.mean(gray) * 2.2))

        # Save ELA for debugging
        cv2.imwrite("ela_result.jpg", cv2.cvtColor(ela_np, cv2.COLOR_RGB2BGR))

        return ela_np, round(ela_score, 2)

    def _detect_face_haar(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        return (x, y, w, h)

    def _detect_photo_rectangle(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Fallback: Detect the rectangular photo area on passport"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_img, w_img = gray.shape
        candidates = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)
            area = w * h

            # Passport photo is usually portrait and reasonably large
            if 0.65 < aspect < 0.95 and area > (h_img * w_img * 0.03):
                candidates.append((x, y, w, h, area))

        if not candidates:
            return None

        # Take the largest reasonable rectangle
        candidates = sorted(candidates, key=lambda c: c[4], reverse=True)
        x, y, w, h, _ = candidates[0]
        return (x, y, w, h)

    def _get_photo_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # Try face first
        bbox = self._detect_face_haar(image)
        if bbox is not None:
            return bbox

        # Fallback to rectangle detection
        return self._detect_photo_rectangle(image)

    def _edge_irregularity_score(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        x, y, w, h = bbox
        pad = 12
        h_img, w_img = image.shape[:2]

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)

        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1] + 1e-5)
        score = min(100, edge_density * 380)
        return round(float(score), 2)

 def _texture_difference_score(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """Compares noise-residual texture (not raw content variance) between photo
    and surrounding document. Raw pixel std is useless here — a face always has
    higher variance than blank background, genuine or not. Noise residual reflects
    the capture/print/scan process, which should be consistent across a genuine
    document regardless of what's pictured in each region."""
    x, y, w, h = bbox
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    photo = gray[y:y + h, x:x + w]

    pad = 18
    h_img, w_img = gray.shape
    y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
    x1, x2 = max(0, x - pad), min(w_img, x + w + pad)
    surrounding = gray[y1:y2, x1:x2].copy()
    surrounding[y - y1:y - y1 + h, x - x1:x - x1 + w] = 0

    def noise_std(region, mask=None):
        denoised = cv2.medianBlur(region, 3)
        residual = cv2.absdiff(region, denoised).astype(np.float32)
        vals = residual[mask] if mask is not None else residual
        if vals.size < 50:
            return 0.0
        return float(np.std(vals))

    photo_noise = noise_std(photo)
    surround_noise = noise_std(surrounding, surrounding > 0)

    if surround_noise == 0:
        return 0.0
    ratio_diff = abs(photo_noise - surround_noise) / surround_noise
    return round(min(100.0, ratio_diff * 70), 2)


    def _sharpness_mismatch_score(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Compares Laplacian-variance sharpness of the photo region vs the surrounding
        document. A photo composited in from a different source (e.g. a different scan,
        a different camera, or a design tool export) very often has a different sharpness
        /resolution profile than the printed document around it, even when the compression
        history looks identical (which is what defeats plain ELA on a re-rendered fake)."""
        x, y, w, h = bbox
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        photo = gray[y:y + h, x:x + w]

        pad = 20
        h_img, w_img = gray.shape
        y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
        x1, x2 = max(0, x - pad), min(w_img, x + w + pad)
        surrounding = gray[y1:y2, x1:x2].copy()
        surrounding[y - y1:y - y1 + h, x - x1:x - x1 + w] = 0

        def laplacian_var(region):
            mask = region > 0
            if np.sum(mask) < 50:
                return 0
            return cv2.Laplacian(region, cv2.CV_64F).var()

        photo_sharp = laplacian_var(photo)
        surround_sharp = laplacian_var(surrounding)
        if surround_sharp == 0:
            return 0.0
        ratio = abs(photo_sharp - surround_sharp) / surround_sharp
        return round(min(100.0, ratio * 60), 2)

    def detect(self, image_path: str) -> Dict:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not read image → {image_path}"}

        ela_img, ela_score = self._perform_ela(image_path)

        bbox = self._get_photo_region(image)

        edge_score = 0.0
        texture_score = 0.0
        sharpness_score = 0.0
        region_found = False
        method = "None"

        if bbox is not None:
            region_found = True
            edge_score = self._edge_irregularity_score(image, bbox)
            texture_score = self._texture_difference_score(image, bbox)
            sharpness_score = self._sharpness_mismatch_score(image, bbox)

        # Check which method found it
        if self._detect_face_haar(image) is not None:
            method = "Face Detection"
        else:
            method = "Photo Rectangle"

        if region_found:
            weighted = (ela_score * 0.30) + (edge_score * 0.25) + (texture_score * 0.25) + (sharpness_score * 0.20)
            signals_above = sum(1 for s in [ela_score, edge_score, texture_score, sharpness_score] if s > 40)
            final_score = max(weighted, 58) if signals_above >= 2 else weighted
        else:
            final_score = ela_score * 0.85

        final_score = round(min(100.0, final_score), 2)

        if final_score >= 55:
            decision = "High Risk - Possible Photo Replacement"
        elif final_score >= 32:
            decision = "Medium Risk - Needs Manual Review"
        else:
            decision = "Low Risk - Likely Genuine"

        reasons = []
        if ela_score > 40:
            reasons.append(f"ELA compression inconsistency (score: {ela_score})")
        if edge_score > 35:
            reasons.append(f"Unusual edge patterns around photo (score: {edge_score})")
        if texture_score > 35:
            reasons.append(f"Texture difference between photo & paper (score: {texture_score})")
        if sharpness_score > 35:
            reasons.append(f"Sharpness/resolution mismatch vs document (score: {sharpness_score})")
        if not reasons:
            reasons.append("No strong signs of photo replacement detected")

        return {
            "final_score": final_score,
            "decision": decision,
            "ela_score": ela_score,
            "edge_score": edge_score,
            "texture_score": texture_score,
            "sharpness_score": sharpness_score,
            "region_found": region_found,
            "detection_method": method,
            "reasons": reasons
        }


if __name__ == "__main__":
    detector = PhotoReplacementDetector()

    image_path = input("\nEnter the full path of the document image: ").strip()

    result = detector.detect(image_path)

    if "error" in result:
        print("Error:", result["error"])
    else:
        print("\n" + "=" * 45)
        print("   PHOTO REPLACEMENT DETECTION REPORT")
        print("=" * 45)
        print(f"Final Score       : {result['final_score']} / 100")
        print(f"Decision          : {result['decision']}")
        print(f"ELA Score         : {result['ela_score']}")
        print(f"Edge Score        : {result['edge_score']}")
        print(f"Texture Score     : {result['texture_score']}")
        print(f"Sharpness Score   : {result['sharpness_score']}")
        print(f"Photo Region Found: {result['region_found']}")
        print(f"Detection Method  : {result['detection_method']}")
        print("\nReasons:")
        for r in result['reasons']:
            print(f"  • {r}")
        print("\nELA visualization saved as → ela_result.jpg")
