import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from passport_ocr import ocr, get_all_texts  # reuse the already-loaded PaddleOCR engine
from aadhaar_validator import verhoeff_checksum
from aadhaar_qr import decode_aadhaar_qr, cross_check_qr_vs_ocr


def extract_aadhaar_fields(texts):
    full = " ".join(texts)
    data = {}

    # Aadhaar number
    m = re.search(r'\b(\d{4}\s?\d{4}\s?\d{4})\b', full)
    if m:
        data['aadhaar_number'] = m.group(1).replace(' ', '')

    # DOB — full date form
    m = re.search(r'DOB\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})', full, re.IGNORECASE)
    if m:
        data['dob'] = m.group(1)
    else:
        # Year-only forms: "Year of Birth: 1987" or abbreviated "YoB:1977"
        m = re.search(r'(?:Year\s*of\s*Birth|YoB)\s*[:\-]?\s*(\d{4})', full, re.IGNORECASE)
        if m:
            data['year_of_birth'] = m.group(1)

    # Gender
    if re.search(r'\bFEMALE\b', full, re.IGNORECASE):
        data['gender'] = 'Female'
    elif re.search(r'\bMALE\b', full, re.IGNORECASE):
        data['gender'] = 'Male'

    # Name — take the LAST name-like line before hitting the demographic block
    # (DOB/YoB/gender). Aadhaar prints native-script name then English name back to
    # back, and PaddleOCR sometimes misreads the native-script line into garbled
    # ASCII that still passes a loose name regex — that garbled line always comes
    # BEFORE the real English name, so "last match wins" picks the correct one.
    LABEL_WORDS = ['GOVERNMENT', 'INDIA', 'DOB', 'BIRTH', 'YEAR', 'YOB', 'MALE', 'FEMALE',
                   'AADHAAR', 'ADHAAR', 'UIDAI']
    STOP_WORDS = ['DOB', 'YOB', 'MALE', 'FEMALE', 'YEAR']
    candidate = None
    for t in texts:
        clean = t.strip()
        upper = clean.upper()
        if any(w in upper for w in STOP_WORDS):
            break  # hit the demographic block — stop looking, keep last candidate found
        if re.match(r'^[A-Za-z]+(\s[A-Za-z]+){1,3}$', clean) and 6 <= len(clean) <= 40:
            if not any(w in upper for w in LABEL_WORDS):
                candidate = clean
    if candidate:
        data['name'] = candidate

    return data

def validate_aadhaar_number(aadhaar_number: str) -> dict:
    if not aadhaar_number or len(aadhaar_number) != 12 or not aadhaar_number.isdigit():
        return {"valid": False, "reason": "Not a valid 12-digit number"}
    valid = verhoeff_checksum(aadhaar_number)
    return {
        "valid": valid,
        "reason": None if valid else "Verhoeff checksum failed — invalid or tampered number"
    }


def analyze_aadhaar(image_path):
    result = ocr.predict(image_path)
    texts = get_all_texts(result)

    fields = extract_aadhaar_fields(texts)
    number_check = validate_aadhaar_number(fields.get('aadhaar_number', ''))

    issues = []
    if not fields.get('aadhaar_number'):
        issues.append("Aadhaar number not detected")
    elif not number_check['valid']:
        issues.append(number_check['reason'])
    if not fields.get('name'):
        issues.append("Name not detected")
    if not fields.get('dob') and not fields.get('year_of_birth'):
        issues.append("Date of birth / year of birth not detected")

    qr_data = decode_aadhaar_qr(image_path)
    qr_issues = cross_check_qr_vs_ocr(qr_data, fields)
    issues.extend(qr_issues)

    return {
        "fields": fields,
        "number_validation": number_check,
        "qr_data": qr_data,
        "ocr_issues": issues,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ").strip()

    result = ocr.predict(path)
    texts = get_all_texts(result)

    print("\n==== RAW OCR TEXTS ====")
    for t in texts:
        print(repr(t))

    print("\n==== PARSED RESULT ====")
    print(analyze_aadhaar(path))
