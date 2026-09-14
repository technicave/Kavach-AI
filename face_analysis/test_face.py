"""
Quick test script
Usage: python test_face.py document.jpg live.jpg
"""

import sys
import cv2
from face_matcher import FaceMatcher   # change this import if needed


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_face.py <document_image> <live_image>")
        sys.exit(1)

    doc_path = sys.argv[1]
    live_path = sys.argv[2]

    print("Loading FaceMatcher...")
    matcher = FaceMatcher(use_gpu=False)

    doc_img = cv2.imread(doc_path)
    live_img = cv2.imread(live_path)

    if doc_img is None:
        print(f"Could not read document image: {doc_path}")
        sys.exit(1)
    if live_img is None:
        print(f"Could not read live image: {live_path}")
        sys.exit(1)

    print("Running verification...")
    result = matcher.verify(doc_img, live_img, threshold=0.42)

    print("\n" + "="*40)
    print("FACE VERIFICATION RESULT")
    print("="*40)
    print(f"Match        : {'YES' if result['match'] else 'NO'}")
    print(f"Similarity   : {result['similarity']:.4f}")
    print(f"Threshold    : {result['threshold']}")
    print(f"Risk Level   : {result['risk_level']}")
    print(f"Message      : {result['message']}")
    print("-"*40)
    print(f"Document Face: {result['doc_face']}")
    print(f"Live Face    : {result['live_face']}")
    print("="*40)


if __name__ == "__main__":
    main()
