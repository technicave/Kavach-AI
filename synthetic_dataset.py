import cv2
import numpy as np
import random
import glob
import csv
from features import get_photo_region, extract_features

random.seed(42)


def jpeg_recompress(img, quality):
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def make_genuine_variant(img):
    """Realistic non-tampered variation: different scan/photo quality, nothing pasted."""
    variant = img.copy()
    q = random.choice([70, 80, 85, 90, 95])
    variant = jpeg_recompress(variant, q)
    if random.random() < 0.5:
        alpha = random.uniform(0.9, 1.1)  # brightness/contrast jitter
        beta = random.uniform(-10, 10)
        variant = cv2.convertScaleAbs(variant, alpha=alpha, beta=beta)
    return variant


def make_tampered_variant(img, donor_face_crop):
    """Paste a foreign face crop into this document's photo region, then resave —
    simulates a real photo-replacement forgery."""
    bbox = get_photo_region(img)
    if bbox is None:
        return None
    x, y, w, h = bbox
    variant = img.copy()

    resized_face = cv2.resize(donor_face_crop, (w, h))

    tamper_mode = random.choice(["swap", "swap_blur", "swap_sharpen"])
    if tamper_mode == "swap_blur":
        resized_face = cv2.GaussianBlur(resized_face, (3, 3), 0)
    elif tamper_mode == "swap_sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        resized_face = cv2.filter2D(resized_face, -1, kernel)

    variant[y:y + h, x:x + w] = resized_face
    variant = jpeg_recompress(variant, random.choice([75, 85, 92, 97]))
    return variant


def build_dataset(sample_paths, n_genuine_per_base=8, n_tampered_per_base=8, out_csv="dataset.csv"):
    images = [cv2.imread(p) for p in sample_paths]
    images = [img for img in images if img is not None]
    if len(images) < 2:
        raise ValueError("Need at least 2 base images (for donor faces to swap between them)")

    rows = []

    for i, base in enumerate(images):
        bbox = get_photo_region(base)
        if bbox is None:
            continue

        # Genuine variants: label 0
        for _ in range(n_genuine_per_base):
            variant = make_genuine_variant(base)
            feats = extract_features(variant)
            if feats:
                rows.append({**feats, "label": 0})

        # Tampered variants: label 1, donor face pulled from a DIFFERENT base image
        donors = [img for j, img in enumerate(images) if j != i]
        for _ in range(n_tampered_per_base):
            donor = random.choice(donors)
            dbbox = get_photo_region(donor)
            if dbbox is None:
                continue
            dx, dy, dw, dh = dbbox
            donor_face = donor[dy:dy + dh, dx:dx + dw]
            variant = make_tampered_variant(base, donor_face)
            if variant is None:
                continue
            feats = extract_features(variant)
            if feats:
                rows.append({**feats, "label": 1})

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ela", "edge", "noise_texture", "sharpness", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows ({sum(r['label']==0 for r in rows)} genuine, "
          f"{sum(r['label']==1 for r in rows)} tampered) to {out_csv}")
    return rows


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "sample"
    paths = sorted(glob.glob(f"{folder}/*.jpg") + glob.glob(f"{folder}/*.jpeg") + glob.glob(f"{folder}/*.png"))
    print(f"Using base images from '{folder}/':", paths)
    build_dataset(paths)
