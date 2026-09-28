"""
Normalization module for Business Entity Resolution Blocking.

Provides clean, deterministic, and safe text normalization:
- Unicode decomposition (handles accents like é -> e, ä -> a)
- Transliteration / standardization of common Indic business terms
- Ordinal number standardization (e.g. 165th -> 165, 1st -> 1)
- Street abbreviation standardization (st -> street, rd -> road, ave -> avenue, etc.)
- Lowercasing and whitespace collapsing
- Safe symbol replacements (& -> and, + -> plus, @ -> at)
- Punctuation removal and missing value safety
"""

import re
import unicodedata
from typing import Optional, Any, Dict

# Common Indic / Devanagari business keywords mapped to standard Latin tokens
INDIC_MAP = {
    'प्राइवेट': 'private', 'प्रा': 'pvt', 'लिमिटेड': 'limited', 'लि': 'ltd',
    'लॉजिस्टिक्स': 'logistics', 'इंटरप्राइजेज': 'enterprises', 'सॉल्यूशंस': 'solutions',
    'टेक्नोलॉजीज': 'technologies', 'सर्विसेज': 'services', 'ट्रेडर्स': 'traders',
    'उद्योग': 'udyog', 'इन्वेस्टमेंट्स': 'investments', 'कंसल्टेंसी': 'consultancy'
}

# Standard street suffix abbreviation mapping
STREET_ABBRS: Dict[str, str] = {
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "cir": "circle", "ste": "suite", "apt": "apartment",
    "fl": "floor", "bldg": "building", "sq": "square", "ctr": "center"
}


def safe_str(val: Any) -> str:
    """Converts a value to string safely handling None, NaN, and whitespace."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none", "null", ""):
        return ""
    return s


def normalize_text(text: Optional[str]) -> str:
    """
    Normalizes a business name or address string:
    1. Handles None / NaN / empty values safely.
    2. Maps known Indic business terms to Latin equivalents.
    3. Unicode normalization (NFKD) to strip diacritics and accents.
    4. Lowercases all characters.
    5. Normalizes '&' -> ' and ', '+' -> ' plus ', '@' -> ' at '.
    6. Strips ordinal suffixes from numeric tokens (e.g. 1st -> 1, 165th -> 165).
    7. Replaces punctuation and special characters with whitespace.
    8. Collapses multiple whitespace characters into a single space.
    """
    s = safe_str(text)
    if not s:
        return ""
    
    # Map common Indic terms
    for k, v in INDIC_MAP.items():
        if k in s:
            s = s.replace(k, f" {v} ")
            
    # Unicode NFKD normalization to remove accents (e.g. 'Wélfare' -> 'Welfare', 'Dräxkor' -> 'Draxkor')
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ASCII", "ignore").decode("utf-8")
    
    # Lowercase
    s = s.lower()
    
    # Expand common symbols safely
    s = s.replace("&", " and ").replace("+", " plus ").replace("@", " at ")
    
    # Standardize ordinals (1st -> 1, 2nd -> 2, 3rd -> 3, 4th -> 4, 165th -> 165, 165rd -> 165)
    s = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", s)
    
    # Replace all non-alphanumeric characters with space
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    
    # Collapse multiple whitespaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def standardize_address_text(text: Optional[str]) -> str:
    """
    Standardizes address street types and unit abbreviations.
    """
    norm = normalize_text(text)
    if not norm:
        return ""
    words = norm.split()
    standardized = [STREET_ABBRS.get(w, w) for w in words]
    return " ".join(standardized)


def normalize_country(country: Optional[str]) -> str:
    """
    Normalizes country strings dynamically and consistently.
    Does not hardcode specific countries.
    Missing/empty values return 'UNKNOWN'.
    """
    c = safe_str(country)
    if not c:
        return "UNKNOWN"
    norm = normalize_text(c)
    if not norm:
        return "UNKNOWN"
    return norm.upper()
