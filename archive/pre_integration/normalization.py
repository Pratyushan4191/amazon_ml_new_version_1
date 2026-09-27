"""
Amazon ML Challenge 2026 - Business Entity Resolution
Module: Data Normalization (Member 2 Deliverable)

Provides conservative, robust normalization for:
- Business Names (Unicode, case, punctuation, legal suffix stripping, token sorting, trigrams)
- Addresses (Landmark extraction, street abbreviations, house numbers, postal codes, city)
- Countries (Hard-gate country normalization for US, India, France, and open-set labels)
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional

# ============================================================================
# Country Mapping & Normalization
# ============================================================================

COUNTRY_CANONICAL_MAP = {
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "in": "india",
    "ind": "india",
    "india": "india",
    "fr": "france",
    "fra": "france",
    "france": "france",
}

def normalize_country(raw: Any) -> str:
    """
    Standardize country string to canonical label.
    Used exclusively as a hard blocking gate (s1_country == cand_country).
    For unseen open-set countries (e.g. at test time), strips and lowercases cleanly.
    """
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    if not s or s == "nan" or s == "null":
        return ""
    # Unicode NFKC standardization
    s = unicodedata.normalize("NFKC", s)
    return COUNTRY_CANONICAL_MAP.get(s, s)


# ============================================================================
# Name Normalization & Legal Suffixes
# ============================================================================

# Sorted by descending token length to guarantee longest-match stripping first.
# Includes US, Indian, French forms and transliterations.
LEGAL_SUFFIXES = [
    # Multi-token phrases
    "private limited",
    "limited liability partnership",
    "société à responsabilité limitée",
    "entreprise unipersonnelle à responsabilité limitée",
    "société par actions simplifiée",
    "société en nom collectif",
    "société anonyme",
    "pvt ltd",
    "pvt. ltd.",
    "pvt. ltd",
    "pvt limited",
    "co. ltd.",
    "co. ltd",
    "co ltd",
    "co. inc.",
    "co. inc",
    "co inc",
    "co. corp.",
    "co. corp",
    "co corp",
    "ltd. liab. co.",
    "limited partnership",
    "trading as",
    "t/a",
    "d.b.a.",
    "dba",
    # Single tokens
    "corporation",
    "incorporated",
    "limited",
    "company",
    "private",
    "corp.",
    "corp",
    "inc.",
    "inc",
    "ltd.",
    "ltd",
    "co.",
    "co",
    "pvt.",
    "pvt",
    "llc.",
    "llc",
    "llp.",
    "llp",
    "lp.",
    "lp",
    "plc.",
    "plc",
    "sarl",
    "eurl",
    "sas",
    "snc",
    "sa",
    # Devanagari forms
    "प्राइवेट लिमिटेड",
    "प्राइवेट",
    "लिमिटेड",
]

# Compile regex pattern to match suffixes at word boundaries at the END of string
# Example: matches "Corp" at end of "Acme Corp", but NOT "Corporation" in "Corporation Bank"
_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\s*$",
    re.IGNORECASE
)

# DBA / Trade Name pattern
_DBA_PATTERN = re.compile(r"\b(dba|d\.b\.a\.|t/a|trading as)\b", re.IGNORECASE)


def strip_legal_suffixes(name: str) -> str:
    """
    Remove terminal legal suffixes iteratively (e.g., 'co ltd' -> empty).
    Preserves core entity keywords that happen to contain suffix words mid-phrase.
    """
    prev = None
    cur = name.strip()
    while prev != cur:
        prev = cur
        cur = _SUFFIX_PATTERN.sub("", cur).strip()
    return cur


def compute_char_trigrams(text: str, n: int = 3) -> str:
    """
    Compute sorted character trigram signature for fuzzy blocking (Rule C).
    Deterministic, robust to minor misspellings and character transpositions.
    """
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < n:
        return cleaned
    trigrams = {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}
    # Pick top 3 alphabetically sorted trigrams to keep key compact and discriminative
    sorted_tri = sorted(trigrams)
    signature = "_".join(sorted_tri[:3])
    return signature


def soundex(token: str) -> str:
    """
    Standard Soundex implementation (dependency-free phonetic key).
    Returns a 4-character phonetic code (e.g., 'A120').
    """
    if not token:
        return ""
    token = unicodedata.normalize("NFKC", token).upper()
    token = re.sub(r"[^A-Z]", "", token)
    if not token:
        return ""

    first_char = token[0]
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    }

    encoded = [first_char]
    prev_code = mapping.get(first_char, "")

    for char in token[1:]:
        code = mapping.get(char, "")
        if code:
            if code != prev_code:
                encoded.append(code)
            prev_code = code
        else:
            prev_code = ""

    result = "".join(encoded)
    return (result + "000")[:4]


def phonetic_encode(token: str) -> str:
    """
    Phonetic encoder for Rule D.
    Uses jellyfish.metaphone if installed; otherwise falls back to Soundex cleanly.
    """
    if not token:
        return ""
    try:
        import jellyfish
        return jellyfish.metaphone(token)
    except (ImportError, Exception):
        return soundex(token)


def normalize_name(raw: Any) -> Dict[str, Any]:
    """
    Normalize business name while preserving raw input and variants.

    Returns dict containing:
      - raw: Original unedited input
      - clean: Lowercased, NFKC Unicode, punctuation to space, & normalized
      - no_suffix: Legal entity suffixes stripped
      - sorted_tokens: Alphabetically sorted tokens without suffixes (Rule A)
      - prefix_4: First 4 characters of clean name (Rule E)
      - trigram_sig: Top trigrams signature (Rule C)
      - first_token: First meaningful name token
      - phonetic_key: Soundex of first name token (Rule D component)
    """
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        raw_str = ""
    else:
        raw_str = str(raw)

    if not raw_str.strip():
        return {
            "raw": raw_str,
            "clean": "",
            "no_suffix": "",
            "sorted_tokens": "",
            "prefix_4": "",
            "trigram_sig": "",
            "first_token": "",
            "phonetic_key": "",
        }

    # Step 1: Unicode NFKC standardization
    s = unicodedata.normalize("NFKC", raw_str)

    # Step 2: Lowercase
    s = s.lower()

    # Step 3: Handle '&' -> 'and'
    s = re.sub(r"&", " and ", s)

    # Step 4: Punctuation normalization (replace non-alphanumeric with space)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Step 5: DBA / Trading As split
    dba_split = _DBA_PATTERN.split(s, maxsplit=1)
    primary_name = dba_split[0].strip() if len(dba_split) > 1 else s

    # Step 6: Strip legal suffixes
    no_suffix = strip_legal_suffixes(primary_name)
    if not no_suffix:  # Fallback if name was ONLY a suffix
        no_suffix = primary_name

    # Step 7: Sorted name tokens (Rule A key)
    tokens = [t for t in no_suffix.split() if t]
    sorted_tokens = " ".join(sorted(tokens))

    # Step 8: Prefix 4 (Rule E key)
    prefix_4 = primary_name.replace(" ", "")[:4] if primary_name else ""

    # Step 9: Trigrams signature (Rule C key)
    trigram_sig = compute_char_trigrams(no_suffix, n=3)

    # Step 10: Phonetic key for first token (Rule D component)
    first_tok = tokens[0] if tokens else ""
    phonetic_k = phonetic_encode(first_tok) if first_tok else ""

    return {
        "raw": raw_str,
        "clean": primary_name,
        "no_suffix": no_suffix,
        "sorted_tokens": sorted_tokens,
        "prefix_4": prefix_4,
        "trigram_sig": trigram_sig,
        "first_token": first_tok,
        "phonetic_key": phonetic_k,
    }


# ============================================================================
# Address Normalization & Component Extractors
# ============================================================================

STREET_ABBREVIATIONS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "highway": "hwy",
    "expressway": "expy",
    # India common patterns
    "nagar": "ngr",
    "colony": "col",
    "sector": "sec",
    "phase": "ph",
    "marg": "marg",
    "cross": "cross",
    "main": "main",
}

LANDMARK_PATTERNS = [
    re.compile(r"\b(near|opp|opposite|behind|next\s+to|adjacent\s+to)\s+[^,;]+", re.IGNORECASE),
]

_POSTAL_REGEX = re.compile(r"\b\d{5,6}\b")
_HOUSE_NUM_REGEX = re.compile(r"^\d+[\w\-/]*$")


def extract_house_number(address_str: str) -> str:
    """
    Extract first token that represents a building/house/plot number.
    Avoids 5-6 digit postal codes / PINs.
    Examples: '123 Main St' -> '123', '#45-B, Sector 5' -> '45-b'.
    """
    if not address_str:
        return ""
    clean = re.sub(r"^[#\s]+", "", address_str)
    tokens = clean.split()
    for t in tokens:
        cleaned_tok = re.sub(r"[^\w\-/]", "", t).lower()
        if cleaned_tok and _HOUSE_NUM_REGEX.match(cleaned_tok):
            # Skip 5 or 6 digit numbers (these are postal codes/ZIPs)
            if len(cleaned_tok) in (5, 6) and cleaned_tok.isdigit():
                continue
            return cleaned_tok
    return ""


def extract_postal_code(address_str: str) -> str:
    """
    Extract 5-digit US ZIP, 6-digit Indian PIN, or 5-digit French postal code.
    """
    if not address_str:
        return ""
    match = _POSTAL_REGEX.search(address_str)
    return match.group(0) if match else ""


def extract_city(address_str: str) -> str:
    """
    Extract city heuristic from address string.
    In comma-separated addresses, city is typically the second to last component.
    """
    if not address_str:
        return ""
    parts = [p.strip().lower() for p in address_str.split(",") if p.strip()]
    if len(parts) >= 2:
        candidate = parts[-2]
        # Remove postal codes if embedded
        candidate = _POSTAL_REGEX.sub("", candidate).strip()
        candidate = re.sub(r"[^\w\s]", "", candidate).strip()
        return candidate
    return ""


ADDRESS_COMMON_STOPWORDS = {
    # Generic road/street suffixes and abbreviations
    "st", "street", "rd", "road", "ave", "avenue", "blvd", "boulevard",
    "dr", "drive", "ln", "lane", "ct", "court", "pl", "place", "way",
    "hwy", "highway", "expy", "expressway", "circle", "cir",
    # Building / floor / unit tokens
    "no", "num", "number", "fl", "floor", "suite", "ste", "apt", "apartment",
    "unit", "bldg", "building", "plot", "shop", "office", "room", "blk", "block",
    # Common geographic / administrative / Indian address generic words
    "nagar", "ngr", "colony", "col", "sector", "sec", "phase", "ph", "marg",
    "cross", "main", "near", "opp", "opposite", "behind", "beside", "adjacent",
    "city", "town", "dist", "district", "state", "post", "po", "new", "old",
    "east", "west", "north", "south", "central",
}


def normalize_address(raw: Any) -> Dict[str, Any]:
    """
    Normalize business address while isolating landmarks and extracting key components.

    Returns dict containing:
      - raw: Original input
      - clean: Lowercased, NFKC, stripped punctuation
      - no_landmark: Address string with landmark clauses removed
      - landmarks: Extracted landmark phrases
      - house_number: House/Plot/Building number (Rule B & G component)
      - street_token: First substantial street token (Rule B component)
      - postal_code: 5-6 digit postal code / PIN
      - city: Extracted city name (Rule D component)
      - tokens: Complete token list
      - informative_tokens: Non-stopword tokens (Rule F & G components)
    """
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        raw_str = ""
    else:
        raw_str = str(raw)

    if not raw_str.strip():
        return {
            "raw": raw_str,
            "clean": "",
            "no_landmark": "",
            "landmarks": [],
            "house_number": "",
            "street_token": "",
            "postal_code": "",
            "city": "",
            "tokens": [],
            "informative_tokens": [],
        }

    # Unicode NFKC & lowercase
    s = unicodedata.normalize("NFKC", raw_str).lower()

    # Extract landmarks
    landmarks = []
    no_landmark = s
    for pat in LANDMARK_PATTERNS:
        matches = pat.findall(s)
        if matches:
            landmarks.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
            no_landmark = pat.sub(" ", no_landmark)

    # Standardize street abbreviations
    for full, abbr in STREET_ABBREVIATIONS.items():
        no_landmark = re.sub(r"\b" + re.escape(full) + r"\b", abbr, no_landmark)

    # Clean punctuation
    clean_no_landmark = re.sub(r"[^\w\s]", " ", no_landmark)
    clean_no_landmark = re.sub(r"\s+", " ", clean_no_landmark).strip()

    # Extract components
    house_num = extract_house_number(clean_no_landmark)
    postal_code = extract_postal_code(s)
    city = extract_city(raw_str)

    tokens = clean_no_landmark.split()
    # Find first street token after house number (len >= 3 to avoid single-char noise)
    street_token = ""
    for t in tokens:
        if t != house_num and len(t) >= 3 and not t.isdigit():
            street_token = t
            break

    # Extract order-independent informative tokens for multi-token address rules (Rules F and G)
    informative_tokens = []
    seen_inf = set()
    for t in tokens:
        if (
            len(t) >= 3
            and not t.isdigit()
            and t != house_num
            and t not in ADDRESS_COMMON_STOPWORDS
            and t not in seen_inf
        ):
            informative_tokens.append(t)
            seen_inf.add(t)

    return {
        "raw": raw_str,
        "clean": clean_no_landmark,
        "no_landmark": clean_no_landmark,
        "landmarks": landmarks,
        "house_number": house_num,
        "street_token": street_token,
        "postal_code": postal_code,
        "city": city,
        "tokens": tokens,
        "informative_tokens": informative_tokens,
    }
