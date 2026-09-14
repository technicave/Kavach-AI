"""
UIDAI Secure QR decode + cross-check against OCR-extracted visual fields.
We do NOT verify the digital signature (needs UIDAI's public key) -- we only
read the demographic fields and compare them against what OCR read off the
printed card. A mismatch is a strong tamper signal: the QR is far harder to
forge than printed text, so a disagreement means the printed side is suspect.
"""

import zlib
import sys
import cv2
from typing import Optional, Dict, Any


def read_qr_raw(image_path: str) -> Optional[bytes]:
    img = cv2.imread(image_path)
    if img is None:
        return None
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if not data:
        return None
    # QR payload is often binary (zlib-compressed), not clean UTF-8 text.
    # cv2 returns a decoded str, so round-trip through latin-1 to recover raw bytes
    # without corrupting them -- standard trick for this exact case.
    return data.encode('latin-1', errors='ignore')


# Field order for UIDAI Secure QR v2, separated by byte 0xFF after zlib decompression.
# If field order looks wrong once tested on real samples, _raw_fields_debug below
# shows the actual split so the mapping can be corrected.
SECURE_QR_FIELDS = [
    "email_mobile_status", "reference_id", "name", "dob", "gender",
    "care_of", "district", "landmark", "house", "location", "pincode",
    "post_office", "state", "street", "sub_district", "vtc",
]


def parse_secure_qr(raw: bytes) -> Optional[Dict[str, Any]]:
    try:
        decompressed = zlib.decompress(raw)
    except zlib.error:
        return None  # not zlib-compressed -- might be an older plain-XML QR instead

    parts = decompressed.split(b'\xff')
    decoded_parts = [p.decode('utf-8', errors='ignore') for p in parts]

    data = {}
    for i, field_name in enumerate(SECURE_QR_FIELDS):
        if i < len(decoded_parts):
            data[field_name] = decoded_parts[i]

    # reference_id's first 4 chars are the Aadhaar number's last 4 digits
    ref_id = data.get("reference_id", "")
    if len(ref_id) >= 4 and ref_id[:4].isdigit():
        data["aadhaar_last4"] = ref_id[:4]

    data["_raw_field_count"] = len(decoded_parts)
    data["_raw_fields_debug"] = decoded_parts[:20]
    return data


def parse_plain_xml_qr(raw: bytes) -> Optional[Dict[str, Any]]:
    """Older pre-2019 Aadhaar QR was plain XML, not compressed."""
    try:
        text = raw.decode('utf-8', errors='ignore')
        if 'PrintLetterBarcodeData' not in text:
            return None
        import xml.etree.ElementTree as ET
        attrs = ET.fromstring(text).attrib
        return {
            "name": attrs.get("name"),
            "dob": attrs.get("dob") or attrs.get("yob"),
            "gender": attrs.get("gender"),
            "aadhaar_last4": (attrs.get("uid") or "")[-4:] or None,
        }
    except Exception:
        return None


def decode_aadhaar_qr(image_path: str) -> Optional[Dict[str, Any]]:
    raw = read_qr_raw(image_path)
    if raw is None:
        return None

    result = parse_plain_xml_qr(raw)
    if result:
        result["qr_format"] = "plain_xml"
        return result

    result = parse_secure_qr(raw)
    if result:
        result["qr_format"] = "secure_qr_v2"
        return result

    return {"qr_format": "unknown", "raw_length": len(raw)}


def cross_check_qr_vs_ocr(qr_data: Optional[Dict[str, Any]], ocr_fields: Dict[str, Any]) -> list:
    issues = []
    if not qr_data:
        issues.append("QR code not found on document")
        return issues
    if qr_data.get("qr_format") == "unknown":
        issues.append("QR code found but could not be decoded")
        return issues

    qr_name = (qr_data.get("name") or "").strip().upper()
    ocr_name = (ocr_fields.get("name") or "").strip().upper()
    if qr_name and ocr_name and qr_name != ocr_name:
        issues.append(f"Name mismatch: QR='{qr_name}' | OCR='{ocr_name}'")

    qr_gender = (qr_data.get("gender") or "").strip().upper()[:1]
    ocr_gender = (ocr_fields.get("gender") or "").strip().upper()[:1]
    if qr_gender and ocr_gender and qr_gender != ocr_gender:
        issues.append(f"Gender mismatch: QR='{qr_gender}' | OCR='{ocr_gender}'")

    qr_last4 = qr_data.get("aadhaar_last4")
    ocr_number = ocr_fields.get("aadhaar_number", "")
    if qr_last4 and ocr_number and ocr_number[-4:] != qr_last4:
        issues.append(f"Aadhaar last-4 mismatch: QR='{qr_last4}' | OCR='{ocr_number[-4:]}'")

    return issues


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ").strip()
    print(decode_aadhaar_qr(path))
