"""
Module: Image Metadata Analysis (SIH26188 - Module 3 bullet 4)

Checks a document image's file-level metadata for tamper indicators that
sit "underneath" the visible content — separate from OCR/visual checks.

Signals, most to least reliable:
  1. EXIF Software/ProcessingSoftware tag naming a known image editor
     -> STRONG signal if present. Absence proves nothing (most scans/phone
        photos strip this anyway, or never had it).
  2. DateTimeOriginal vs DateTime (modify date) mismatch
     -> MEDIUM signal. A modify date after capture date suggests the file
        was re-saved by something after the original camera/scan wrote it.
  3. Missing EXIF entirely on a JPEG -> INFORMATIONAL ONLY. Very common
     for innocent reasons (messaging apps, cloud uploads, some scanners
     strip metadata by default). Reported but never scored as a risk on
     its own.

  NOTE: An earlier version of this module attempted double-JPEG-compression
  detection by counting quantization tables in the file. That approach was
  tested and found NOT to work — re-saving a decoded image writes a fresh
  table set rather than "stacking" onto the original, so table-counting
  doesn't actually detect re-compression. Real double-compression detection
  requires DCT coefficient histogram analysis, which is out of scope for
  now. Removed rather than shipped non-functional.

Usage:
    python3 metadata_check.py path/to/image.jpg
"""
import sys
from PIL import Image
from PIL.ExifTags import TAGS

KNOWN_EDITORS = [
    "photoshop", "gimp", "snapseed", "pixlr", "lightroom", "affinity",
    "paint.net", "photopea", "picsart", "canva", "corel", "illustrator",
    "facetune", "remini", "inpaint",
]


def get_exif_dict(path: str) -> dict:
    try:
        img = Image.open(path)
        raw = img._getexif() if hasattr(img, "_getexif") else None
        if not raw:
            return {}
        return {TAGS.get(tag_id, tag_id): value for tag_id, value in raw.items()}
    except Exception:
        return {}


def check_software_tag(exif: dict) -> list:
    issues = []
    for key in ("Software", "ProcessingSoftware", "HostComputer"):
        val = exif.get(key)
        if val and isinstance(val, str):
            val_lower = val.lower()
            for editor in KNOWN_EDITORS:
                if editor in val_lower:
                    issues.append(f"EXIF '{key}' names an image editor: '{val}'")
                    break
    return issues


def check_timestamp_consistency(exif: dict) -> list:
    issues = []
    original = exif.get("DateTimeOriginal")
    modified = exif.get("DateTime")
    if original and modified and original != modified:
        issues.append(
            f"Capture time ({original}) differs from file modify time ({modified}) "
            f"— file was re-saved after the original capture"
        )
    return issues


def run_metadata_check(path: str) -> dict:
    exif = get_exif_dict(path)
    is_jpeg = path.lower().endswith((".jpg", ".jpeg"))

    software_issues = check_software_tag(exif)
    timestamp_issues = check_timestamp_consistency(exif)

    all_issues = software_issues + timestamp_issues

    info_notes = []
    if is_jpeg and not exif:
        info_notes.append(
            "No EXIF metadata found. Not necessarily suspicious — common for "
            "images that passed through messaging apps, cloud storage, or "
            "certain scanners, all of which strip EXIF by default."
        )

    if software_issues:
        decision = "HIGH RISK — editing software fingerprint found in metadata"
    elif timestamp_issues:
        decision = "MEDIUM RISK — metadata inconsistency, needs manual review"
    else:
        decision = "LOW RISK — no metadata tamper indicators found"

    return {
        "decision": decision,
        "issues": all_issues,
        "info_notes": info_notes,
        "exif_present": bool(exif),
        "exif_sample": {k: v for k, v in exif.items()
                         if k in ("Make", "Model", "Software", "DateTimeOriginal", "DateTime")},
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "sample/original.jpg"
    report = run_metadata_check(path)

    print(f"\n{'=' * 50}")
    print("  IMAGE METADATA ANALYSIS REPORT")
    print(f"{'=' * 50}")
    print(f"Decision: {report['decision']}\n")

    print("-- Relevant EXIF fields found --")
    if report["exif_sample"]:
        for k, v in report["exif_sample"].items():
            print(f"  {k:20s}: {v}")
    else:
        print("  none")

    print("\n-- Issues --")
    if report["issues"]:
        for issue in report["issues"]:
            print("  ⚠️ ", issue)
    else:
        print("  none")

    if report["info_notes"]:
        print("\n-- Notes (informational, not risk indicators) --")
        for note in report["info_notes"]:
            print("  ℹ️ ", note)


if __name__ == "__main__":
    main()
