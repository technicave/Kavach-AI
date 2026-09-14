"""
One-off debug script. Run this from your project root (same place you run
passport_ocr.py from), with a real passport image path.

It does NOT modify passport_ocr.py. It just prints the raw structure PaddleOCR
gives back, so we know exactly which keys hold the text strings and which
hold the box coordinates for your installed PaddleOCR version.
"""
import sys
from paddleocr import PaddleOCR

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "tampered_passports/fake3.png"

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
result = ocr.predict(IMAGE_PATH)

for i, res in enumerate(result):
    data = getattr(res, 'json', {}) or {}
    print(f"--- result[{i}] top-level keys ---")
    print(list(data.keys()))

    inner = data.get('res', data)
    print(f"--- inner keys ---")
    print(list(inner.keys()) if isinstance(inner, dict) else type(inner))

    if isinstance(inner, dict):
        for key in inner.keys():
            val = inner[key]
            if isinstance(val, list):
                print(f"  {key}: list of {len(val)} items")
                if val:
                    print(f"    first item: {val[0]!r}")
            else:
                print(f"  {key}: {type(val)}")
