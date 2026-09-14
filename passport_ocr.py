from paddleocr import PaddleOCR
import re
from datetime import date

# ISO 3166-1 alpha-3 codes (plus a few common MRZ-specific non-country codes
# like UNK/UNO/XXX/EUE used on some travel documents). Used to validate and
# auto-correct the MRZ nationality/issuing-country fields, since a single
# fixed digit<->letter mapping isn't enough — some OCR confusions (like 0
# looking like O *or* D depending on the font) are genuinely ambiguous and
# need to be checked against real codes rather than guessed blindly.
ISO3_COUNTRY_CODES = {
    "AFG","ALB","DZA","ASM","AND","AGO","AIA","ATA","ATG","ARG","ARM","ABW",
    "AUS","AUT","AZE","BHS","BHR","BGD","BRB","BLR","BEL","BLZ","BEN","BMU",
    "BTN","BOL","BIH","BWA","BRA","BRN","BGR","BFA","BDI","CPV","KHM","CMR",
    "CAN","CYM","CAF","TCD","CHL","CHN","COL","COM","COG","COD","COK","CRI",
    "CIV","HRV","CUB","CUW","CYP","CZE","DNK","DJI","DMA","DOM","ECU","EGY",
    "SLV","GNQ","ERI","EST","SWZ","ETH","FJI","FIN","FRA","GUF","PYF","GAB",
    "GMB","GEO","DEU","GHA","GIB","GRC","GRL","GRD","GLP","GUM","GTM","GIN",
    "GNB","GUY","HTI","HND","HKG","HUN","ISL","IND","IDN","IRN","IRQ","IRL",
    "ISR","ITA","JAM","JPN","JOR","KAZ","KEN","KIR","PRK","KOR","KWT","KGZ",
    "LAO","LVA","LBN","LSO","LBR","LBY","LIE","LTU","LUX","MAC","MDG","MWI",
    "MYS","MDV","MLI","MLT","MHL","MTQ","MRT","MUS","MEX","FSM","MDA","MCO",
    "MNG","MNE","MSR","MAR","MOZ","MMR","NAM","NRU","NPL","NLD","NCL","NZL",
    "NIC","NER","NGA","NIU","MKD","NOR","OMN","PAK","PLW","PAN","PNG","PRY",
    "PER","PHL","POL","PRT","PRI","QAT","ROU","RUS","RWA","KNA","LCA","VCT",
    "WSM","SMR","STP","SAU","SEN","SRB","SYC","SLE","SGP","SVK","SVN","SLB",
    "SOM","ZAF","SSD","ESP","LKA","SDN","SUR","SWE","CHE","SYR","TWN","TJK",
    "TZA","THA","TLS","TGO","TON","TTO","TUN","TUR","TKM","TUV","UGA","UKR",
    "ARE","GBR","USA","URY","UZB","VUT","VEN","VNM","YEM","ZMB","ZWE",
    "UNK","UNO","XXX",
}

AMBIGUOUS_MRZ_CHARS = {
    '0': ['O', 'D', 'Q'],
    '1': ['I', 'L'],
    '5': ['S'],
    '8': ['B'],
    '2': ['Z'],
    '6': ['G'],
    '4': ['A'],
}


def correct_country_code(raw3: str):
    """Return (code, was_corrected). Tries the raw value first, then searches
    plausible OCR-confusion substitutions for a match against real ISO3
    codes, rather than applying one fixed digit->letter rule. This is what
    catches cases like '1N0' -> 'IND' where the '0' really meant 'D', which a
    single-mapping fixer (0 always -> O) would get wrong."""
    raw3 = raw3.upper()
    if raw3 in ISO3_COUNTRY_CODES:
        return raw3, False

    candidates = {raw3}
    for i, ch in enumerate(raw3):
        if ch in AMBIGUOUS_MRZ_CHARS:
            new_candidates = set()
            for cand in candidates:
                for alt in AMBIGUOUS_MRZ_CHARS[ch]:
                    new_candidates.add(cand[:i] + alt + cand[i + 1:])
            candidates |= new_candidates

    valid = sorted(c for c in candidates if c in ISO3_COUNTRY_CODES)
    if valid:
        return valid[0], True

    return raw3, False

IMAGE_PATH = "sample/7.jpg"   # ← change if needed

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

CHECKSUM_VALUES = {str(i): i for i in range(10)}
CHECKSUM_VALUES.update({chr(65 + i): i + 10 for i in range(26)})
CHECKSUM_VALUES['<'] = 0
WEIGHTS = [7, 3, 1]


def mrz_checksum(data: str) -> int:
    total = sum(CHECKSUM_VALUES.get(c, 0) * WEIGHTS[i % 3] for i, c in enumerate(data))
    return total % 10


def fix_digit_confusions(s: str) -> str:
    """For fields that must be all-digits (DOB, expiry)."""
    return (s.replace('O', '0').replace('Q', '0')
             .replace('I', '1').replace('L', '1')
             .replace('S', '5').replace('B', '8')
             .replace('Z', '2').replace('G', '6'))


def fix_letter_confusions(s: str) -> str:
    """For fields that must be all-letters (nationality)."""
    return (s.replace('0', 'O').replace('1', 'I').replace('5', 'S')
             .replace('8', 'B').replace('2', 'Z').replace('6', 'G'))


def get_all_texts(result):
    texts = []
    for res in result:
        data = getattr(res, 'json', {}) or {}
        if 'rec_texts' in data:
            texts.extend(data['rec_texts'])
        elif isinstance(data.get('res'), dict) and 'rec_texts' in data['res']:
            texts.extend(data['res']['rec_texts'])
    return [t.strip() for t in texts if t and t.strip()]


def extract_fields(texts):
    full = " ".join(texts).upper()
    data = {}

    # Passport Number — this field is usually large, clear print and OCRs
    # reliably even when the small MRZ line drops characters, so it's our
    # best cross-reference source for fixing the MRZ below.
    m = re.search(r'\b([A-Z]?\d{7})\b', full)
    if m:
        data['passport_number'] = m.group(1)

    # Name (Visual Zone fallback only — parse_mrz()/main() prefer the MRZ-derived
    # name since it's structured and far less prone to grabbing a stray label).
    LABEL_WORDS = ['REPUBLIC', 'INDIA', 'NATIONALITY', 'BENGALURU', 'SURNAME',
                   'GIVEN', 'NAME', 'NAMES', 'TYPE', 'SEX', 'CODE', 'DATE',
                   'ISSUE', 'EXPIRY', 'PLACE', 'BIRTH', 'PASSPORT', 'COUNTRY',
                   'SIGNATURE', 'AUTHORITY']
    for t in texts:
        clean = t.strip().upper()
        if re.match(r'^[A-Z]+[-\s]?[A-Z]+$', clean) and 6 <= len(clean) <= 30:
            if not any(x in clean for x in LABEL_WORDS):
                data['name'] = t.strip().replace('-', ' ')
                break

    # Sex
    if re.search(r'\bM\b', full):
        data['sex'] = 'M'
    elif re.search(r'\bF\b', full):
        data['sex'] = 'F'

    # Dates
    dates = re.findall(r'\b(\d{2}/\d{2}/\d{4})\b', full)
    if len(dates) >= 1:
        data['date_of_birth'] = dates[0]
    if len(dates) >= 2:
        data['date_of_issue'] = dates[1]
    if len(dates) >= 3:
        data['date_of_expiry'] = dates[2]

    if "INDIAN" in full:
        data['nationality'] = "INDIAN"

    for t in texts:
        if "BENGALURU" in t.upper() or "KARNATAKA" in t.upper():
            data['place'] = t.strip()
            break

    return data


def recover_line2(raw_candidate: str, visual_passport_number: str = None) -> str:
    """PaddleOCR very commonly clips the first character of the MRZ's second
    line — it's small, tightly kerned text, and a leading letter (rather than
    a digit) is especially prone to being dropped or merged with the line
    start. When that happens every subsequent field parses one position off,
    which is exactly what produced the garbage nationality/sex/DOB values.
    We detect the dropped-character case by length (43 instead of the
    standard TD3 44) and recover the missing character from the Visual Zone
    OCR of the passport number, which is printed much larger and rarely has
    this problem."""
    candidate = raw_candidate
    if len(candidate) == 43 and visual_passport_number:
        candidate = visual_passport_number[0] + candidate
    return candidate.ljust(44, '<')


def parse_name_from_mrz(line1: str):
    """Parse surname/given names directly from MRZ line 1, which encodes
    them in a strict 'P<CCCSURNAME<<GIVEN<NAMES<<<...' format. This is far
    more reliable than scraping the Visual Zone with a generic regex, which
    can accidentally pick up a printed label (e.g. 'Surname') instead of the
    actual name if the label happens to OCR as a clean all-caps word."""
    if not line1 or len(line1) < 6 or not line1.startswith('P'):
        return None, None
    body = line1[5:]  # skip "P<CCC"
    parts = body.split('<<')
    surname = parts[0].replace('<', ' ').strip() if parts and parts[0] else None
    given = parts[1].replace('<', ' ').strip() if len(parts) > 1 and parts[1] else None
    return surname or None, given or None


def build_corrected_line2(line2: str) -> str:
    """The raw MRZ line often has genuine letter/digit OCR confusions baked
    into it (e.g. nationality 'IND' misread as '1ND' because I and 1 look
    almost identical in the small MRZ font). parse_mrz() already corrects
    these confusions field-by-field when it builds nationality_mrz, dob_mrz,
    etc — but previously the *raw* line2 string printed to the user still
    showed the uncorrected characters, which is misleading and inconsistent
    with the parsed results shown right below it. This rebuilds a cleaned
    version of the line using the same per-field rules, so what you see
    printed matches what was actually parsed."""
    if len(line2) < 28:
        return line2
    passport_field = line2[0:10]                              # digits/letters + check digit, left as-is
    # Nationality must be corrected against the RAW segment, not a
    # single-mapping letter-fixer's output — fix_letter_confusions always
    # maps '0'->'O', which would already destroy the '0'->'D' possibility
    # needed to recover 'IND' from a misread like '1N0'. correct_country_code
    # tries all plausible substitutions and checks against real ISO3 codes.
    nationality, _ = correct_country_code(line2[10:13])
    dob = fix_digit_confusions(line2[13:20])                  # must be digits (+check digit)
    sex = line2[20]
    expiry = fix_digit_confusions(line2[21:28])               # must be digits (+check digit)
    rest = line2[28:]
    return passport_field + nationality + dob + sex + expiry + rest


def parse_mrz(texts, visual_passport_number=None):
    candidates = []
    for t in texts:
        cleaned = re.sub(r'[^A-Z0-9<]', '', t.upper())
        if len(cleaned) >= 30:
            candidates.append(cleaned)

    if len(candidates) < 1:
        return None

    line1 = None
    line2_raw = None

    for c in candidates:
        if c.startswith('P<') or (c.startswith('P') and '<<' in c):
            line1 = c
        elif re.match(r'^[A-Z0-9]{7,9}<', c):
            line2_raw = c

    if not line2_raw:
        for c in sorted(candidates, key=lambda x: sum(ch.isdigit() for ch in x[:10]), reverse=True):
            if not c.startswith('P'):
                line2_raw = c
                break

    if not line1:
        for c in candidates:
            if c.startswith('P'):
                line1 = c
                break

    if not line1:
        # Same dropped-leading-character issue as line 2, just here the missing
        # char is always 'P' (document type), since every candidate we're
        # scanning is a passport MRZ line 1. A genuine line 1 minus its
        # leading 'P' looks like "<IND<SURNAME<<GIVEN<<<...".
        for c in candidates:
            if c.startswith('<') and re.match(r'^<[A-Z]{3}', c) and c is not line2_raw:
                line1 = 'P' + c
                break

    if not line2_raw:
        return None

    line2 = recover_line2(line2_raw, visual_passport_number)
    line2_corrected = build_corrected_line2(line2)

    passport_num = line2_corrected[0:9].replace('<', '')
    passport_check = line2_corrected[9] if len(line2_corrected) > 9 else ''
    # build_corrected_line2() already ran the nationality segment through
    # correct_country_code() against the RAW (pre-fixed) characters, so
    # line2_corrected[10:13] is already the resolved code. We compare it
    # against the untouched raw slice from `line2` just to know whether a
    # correction actually happened, for the log note below.
    nationality = line2_corrected[10:13]
    nationality_corrected = (nationality != line2[10:13].upper())
    dob = line2_corrected[13:19]
    dob_check = line2_corrected[19] if len(line2_corrected) > 19 else ''
    sex = line2_corrected[20] if len(line2_corrected) > 20 else ''
    expiry = line2_corrected[21:27] if len(line2_corrected) > 27 else ''
    expiry_check = line2_corrected[27] if len(line2_corrected) > 27 else ''

    if not re.match(r'^\d{6}$', dob):
        dob = ''
    if not re.match(r'^\d{6}$', expiry):
        expiry = ''

    # Checksums are computed on the corrected line, since the goal is to
    # verify the *document's* encoded data against itself, not to penalize
    # an OCR misread that's already been resolved with high confidence.
    checksum_valid = {
        'passport_number': mrz_checksum(line2_corrected[0:9]) == int(passport_check) if passport_check.isdigit() else None,
        'dob': mrz_checksum(dob) == int(dob_check) if dob and dob_check.isdigit() else None,
        'expiry': mrz_checksum(expiry) == int(expiry_check) if expiry and expiry_check.isdigit() else None,
    }

    issuing_country_raw = line1[2:5] if line1 and len(line1) > 5 else ''
    issuing_country, issuing_country_corrected = (
        correct_country_code(issuing_country_raw) if issuing_country_raw else ('', False)
    )

    surname, given_names = parse_name_from_mrz(line1)

    return {
        'mrz_line1': line1 or '',
        'mrz_line2_raw': line2,
        'mrz_line2': line2_corrected,
        'document_type': line1[0] if line1 else '',
        'issuing_country': issuing_country,
        'issuing_country_corrected': issuing_country_corrected,
        'surname': surname,
        'given_names': given_names,
        'name_mrz': f"{given_names} {surname}".strip() if (surname or given_names) else None,
        'passport_number_mrz': passport_num,
        'nationality_mrz': nationality,
        'nationality_corrected': nationality_corrected,
        'dob_mrz': dob,
        'sex_mrz': sex,
        'expiry_mrz': expiry,
        'checksum_valid': checksum_valid,
        'was_recovered': len(line2_raw) == 43,
    }


def format_mrz_date(yymmdd, field_type="dob", reference_year=None):
    """field_type='dob' assumes the year must be in the past.
    field_type='issue'/'expiry' instead picks whichever century puts the
    year closest to 'now', since passport validity windows can legitimately
    roll from e.g. 2024 into 2034 — a fixed '>30 => 19xx' cutoff (the
    original bug) would wrongly turn an expiry of '34' into 1934."""
    if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
        return "Invalid"
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
    year = int(yy)
    current_year = reference_year or date.today().year

    if field_type == "dob":
        full_year = 1900 + year if (2000 + year) > current_year else 2000 + year
    else:
        candidate_2000 = 2000 + year
        candidate_1900 = 1900 + year
        full_year = (candidate_2000
                     if abs(candidate_2000 - current_year) <= abs(candidate_1900 - current_year)
                     else candidate_1900)

    return f"{dd}/{mm}/{full_year}"


def check_tamper(visual, mrz):
    issues = []

    if not mrz:
        issues.append("MRZ not detected properly")
        return issues

    # Trust checksum results first — if a field's checksum is invalid, that's
    # a much stronger signal than a raw visual-vs-MRZ text mismatch, which can
    # just be an OCR error on either side.
    for field, valid in mrz.get('checksum_valid', {}).items():
        if valid is False:
            issues.append(f"MRZ checksum failed for {field} — possible tampering or unrecovered OCR error")

    vis_num = visual.get('passport_number', '').upper().replace('Z', '2')
    mrz_num = mrz.get('passport_number_mrz', '')
    if vis_num and mrz_num and vis_num[-7:] != mrz_num[-7:]:
        issues.append(f"Passport Number mismatch: Visual={visual.get('passport_number')} | MRZ={mrz_num}")

    if visual.get('sex') and mrz.get('sex_mrz') and visual['sex'] != mrz['sex_mrz']:
        issues.append(f"Sex mismatch: Visual={visual['sex']} | MRZ={mrz['sex_mrz']}")

    if mrz.get('nationality_mrz') not in ISO3_COUNTRY_CODES:
        issues.append(f"Suspicious nationality in MRZ: {mrz.get('nationality_mrz')} (not a recognized ISO3 country code, even after OCR-confusion correction)")

    if visual.get('date_of_birth') and mrz.get('dob_mrz'):
        vis_dob = visual['date_of_birth'].replace('/', '')
        if len(vis_dob) == 8:
            vis_yymmdd = vis_dob[6:8] + vis_dob[2:4] + vis_dob[0:2]
            if vis_yymmdd != mrz['dob_mrz']:
                issues.append(f"Date of Birth mismatch: Visual={visual['date_of_birth']} | MRZ={format_mrz_date(mrz['dob_mrz'], 'dob')}")

    return issues


def main():
    print("Running OCR...\n")
    result = ocr.predict(IMAGE_PATH)
    texts = get_all_texts(result)

    print("==== All Detected Texts ====")
    for t in texts:
        print(t)

    print("\n==== Extracted Fields (Visual Zone) ====")
    visual = extract_fields(texts)
    for k, v in visual.items():
        print(f"{k:22}: {v}")

    print("\n==== MRZ Parsing ====")
    mrz = parse_mrz(texts, visual_passport_number=visual.get('passport_number'))
    if not mrz:
        print("Failed to parse MRZ")
    else:
        # Name: prefer the MRZ-derived name (structured, reliable) over the
        # Visual Zone regex guess, which can mistakenly grab a label word.
        if mrz.get('name_mrz'):
            visual['name'] = mrz['name_mrz']
            print(f"(name corrected using MRZ: {mrz['name_mrz']})")

        if mrz.get('was_recovered'):
            print("(Note: MRZ line 2 was 1 character short — recovered using Visual Zone passport number)")
        if mrz['mrz_line2_raw'] != mrz['mrz_line2']:
            print(f"mrz_line2 (raw OCR) : {mrz['mrz_line2_raw']}  (contains OCR letter/digit confusion)")
        if mrz.get('nationality_corrected'):
            print(f"(nationality corrected via ISO3 lookup: raw MRZ segment did not match a real country code as-is)")
        print(f"mrz_line1           : {mrz['mrz_line1']}")
        print(f"mrz_line2           : {mrz['mrz_line2']}")
        print(f"passport_number_mrz : {mrz['passport_number_mrz']}")
        print(f"nationality_mrz     : {mrz['nationality_mrz']}")
        print(f"sex_mrz             : {mrz['sex_mrz']}")
        print(f"dob_mrz             : {mrz['dob_mrz']}  → {format_mrz_date(mrz['dob_mrz'], 'dob')}")
        print(f"expiry_mrz          : {mrz['expiry_mrz']}  → {format_mrz_date(mrz['expiry_mrz'], 'expiry')}")
        print(f"checksum_valid      : {mrz['checksum_valid']}")

    print("\n==== Tamper Check ====")
    issues = check_tamper(visual, mrz)
    if not issues:
        print("✅ No major tamper indicators found")
    else:
        print("⚠️ Potential issues detected:")
        for issue in issues:
            print("   -", issue)


if __name__ == "__main__":
    main()
