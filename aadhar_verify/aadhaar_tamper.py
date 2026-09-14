import sys
import os
import cv2
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "aadhaar_tamper_model.joblib")


def predict_aadhaar_tamper(image_path, model_path=MODEL_PATH):
    clf = joblib.load(model_path)
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image → {image_path}"}

    feats = extract_features(img)
    if feats is None:
        return {"error": "Could not locate a photo region on this document"}

    X = [[feats["ela"], feats["edge"], feats["noise_texture"], feats["sharpness"]]]
    proba = clf.predict_proba(X)[0]
    tampered_prob = round(float(proba[1]) * 100, 2)

    rule_triggered = feats["ela"] > 40 or feats["edge"] > 60
    if rule_triggered:
        tampered_prob = max(tampered_prob, 75.0)

    if tampered_prob >= 70:
        decision = "High Risk - Possible Photo Replacement"
    elif tampered_prob >= 35:
        decision = "Medium Risk - Needs Manual Review"
    else:
        decision = "Low Risk - Likely Genuine"

    return {
        "tampered_probability": tampered_prob,
        "decision": decision,
        "rule_override_triggered": rule_triggered,
        "features": feats,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ").strip()
    result = predict_aadhaar_tamper(path)
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("\n" + "=" * 45)
        print("   AADHAAR PHOTO REPLACEMENT DETECTION (ML)")
        print("=" * 45)
        print(f"Tampered Probability : {result['tampered_probability']}%")
        print(f"Decision              : {result['decision']}")
        print(f"Rule Override Fired   : {result['rule_override_triggered']}")
        print(f"Raw signals           : {result['features']}")
