"""
SIH26188 - Face Detection & Matching Module
Improved version for low-quality document crops
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[WARNING] InsightFace not installed.")


class FaceMatcher:
    def __init__(self, use_gpu: bool = False):
        self.insight_app = None

        if INSIGHTFACE_AVAILABLE:
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
                self.insight_app = FaceAnalysis(name='buffalo_l', providers=providers)
                self.insight_app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
                print("[INFO] InsightFace (buffalo_l) loaded")
            except Exception as e:
                print(f"[WARNING] InsightFace failed to load: {e}")
                self.insight_app = None

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        """Strong enhancement for low-quality / noisy face crops"""
        if image is None:
            return None

        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if too small (very important for cropped faces)
        h, w = image.shape[:2]
        if max(h, w) < 300:
            scale = 400 / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)

        # CLAHE contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

        return image

    def get_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        if image is None or self.insight_app is None:
            return None

        # Try original first
        try:
            faces = self.insight_app.get(image)
            if faces:
                face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                return {
                    "bbox": face.bbox.astype(int).tolist(),
                    "embedding": face.embedding,
                    "det_score": float(face.det_score),
                    "source": "insightface"
                }
        except Exception:
            pass

        # Try with strong enhancement
        try:
            enhanced = self._enhance(image)
            faces = self.insight_app.get(enhanced)
            if faces:
                face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                return {
                    "bbox": face.bbox.astype(int).tolist(),
                    "embedding": face.embedding,
                    "det_score": float(face.det_score),
                    "source": "insightface_enhanced"
                }
        except Exception as e:
            print(f"[ERROR] Detection failed even after enhancement: {e}")

        return None

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if emb1 is None or emb2 is None:
            return 0.0
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))

    def verify(self, doc_image: np.ndarray, live_image: np.ndarray, threshold: float = 0.42) -> Dict[str, Any]:
        result = {
            "match": False,
            "similarity": 0.0,
            "threshold": threshold,
            "risk_level": "High",
            "message": "",
            "doc_face": None,
            "live_face": None
        }

        doc_face = self.get_face(doc_image)
        if doc_face is None:
            result["message"] = "No face detected in document"
            return result

        result["doc_face"] = {
            "bbox": doc_face["bbox"],
            "score": doc_face["det_score"],
            "source": doc_face["source"]
        }

        live_face = self.get_face(live_image)
        if live_face is None:
            result["message"] = "No face detected in live image"
            return result

        result["live_face"] = {
            "bbox": live_face["bbox"],
            "score": live_face["det_score"],
            "source": live_face["source"]
        }

        if doc_face["embedding"] is not None and live_face["embedding"] is not None:
            sim = self.cosine_similarity(doc_face["embedding"], live_face["embedding"])
            result["similarity"] = round(sim, 4)
            result["match"] = sim >= threshold

            if sim >= 0.55:
                result["risk_level"] = "Low"
                result["message"] = "Strong match"
            elif sim >= threshold:
                result["risk_level"] = "Medium"
                result["message"] = "Possible match - recommend manual check"
            else:
                result["risk_level"] = "High"
                result["message"] = "Faces do not match"
        else:
            result["message"] = "Could not extract embeddings"
            result["risk_level"] = "Medium"

        return result
