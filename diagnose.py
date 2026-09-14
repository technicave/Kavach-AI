import cv2
import os
from features import get_photo_region

print("=== Haar Cascade Check ===")
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
print("Cascade path:", cascade_path)
print("File exists:", os.path.exists(cascade_path))

cascade = cv2.CascadeClassifier(cascade_path)
print("Cascade loaded successfully:", not cascade.empty())

print("\n=== Region Detection Check ===")
image_path = input("Enter image path to diagnose: ").strip()
image = cv2.imread(image_path)
if image is None:
    print(f"ERROR: could not read {image_path}")
else:
    print("Image shape:", image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
    print(f"Faces detected by Haar cascade: {len(faces)}")
    if len(faces) > 0:
        print("Face boxes (x, y, w, h):", faces.tolist())

    bbox = get_photo_region(image)
    print("\nFinal bbox chosen by get_photo_region:", bbox)

    if bbox is not None:
        x, y, w, h = bbox
        crop = image[y:y+h, x:x+w]
        cv2.imwrite("_diagnostic_detected_region.jpg", crop)
        print(f"\nSaved the detected region as '_diagnostic_detected_region.jpg'")
        print("OPEN THAT FILE AND LOOK AT IT — it should show the passport photo.")
        print("If it shows text, a table border, or anything other than the face photo,")
        print("that confirms the detector is locking onto the wrong region.")
    else:
        print("No region detected at all.")
