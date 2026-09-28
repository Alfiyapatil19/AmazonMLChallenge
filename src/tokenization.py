"""
Tokenization module for Business Entity Resolution Blocking.

Extracts:
1. Business Name Tokens (with generic stopword filtering and acronym detection)
2. Address Tokens (street names, house numbers, compound 2-token shingles, postal codes)
3. Character N-Grams for business names (for fuzzy/typo robustness)
"""

import re
from typing import List, Set, Optional
from .normalization import normalize_text

# Generic legal form and high-frequency company type words that lack individual discriminative power
LEGAL_STOPWORDS: Set[str] = {
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation',
    'co', 'company', 'llp', 'pllc', 'lp', 'pc', 'pa', 'gmbh', 'sa', 'sarl', 'sas', 'spa',
    'the', 'and', 'of', 'in', 'at', 'for', 'on', 'with', 'to', 'by', 'an', 'a',
    'services', 'service', 'enterprise', 'enterprises', 'group', 'holdings', 'holding',
    'solutions', 'technologies', 'technology', 'consulting', 'management', 'international'
}

# Generic address keywords, structural labels, and country/state designators
ADDR_STOPWORDS: Set[str] = {
    'st', 'street', 'rd', 'road', 'ave', 'avenue', 'dr', 'drive', 'blvd', 'boulevard',
    'ln', 'lane', 'ct', 'court', 'cir', 'circle', 'way', 'pkwy', 'parkway', 'hwy', 'highway',
    'suite', 'ste', 'unit', 'apt', 'apartment', 'floor', 'fl', 'bldg', 'building', 'no', 'room',
    'rm', 'near', 'opp', 'opposite', 'behind', 'bh', 'cross', 'main', 'sector', 'plot',
    'north', 'south', 'east', 'west', 'n', 's', 'e', 'w', 'box', 'pobox', 'post', 'office',
    'the', 'and', 'of', 'in', 'at', 'for', 'on', 'with', 'to', 'by', 'an', 'a',
    'house', 'flat', 'tower', 'block', 'phase', 'nagar', 'colony', 'marg', 'bhavan',
    'residency', 'enclave', 'complex', 'arcade', 'plaza', 'market', 'lane',
    # Common region indicators
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia',
    'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
    'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt',
    'va', 'wa', 'wv', 'wi', 'wy', 'usa', 'india', 'france'
}


def extract_name_tokens(name: Optional[str], min_length: int = 2) -> List[str]:
    """
    Extracts meaningful business name tokens:
    1. Distinctive words (length >= min_length, not in legal stopwords).
    2. Acronyms & concatenated short words (e.g. "j u" -> "ju", "b plus" -> "bplus").
    3. Fallback to longest token if all tokens were legal stopwords.
    """
    norm = normalize_text(name)
    if not norm:
        return []
    words = norm.split()
    tokens: List[str] = []
    
    for w in words:
        if len(w) >= min_length and w not in LEGAL_STOPWORDS:
            tokens.append(f"n_{w}")
            
    # Capture acronyms / concatenated short initials (e.g., J/U -> "ju", A&B -> "ab")
    if len(words) >= 2:
        short_combo = "".join(w for w in words if len(w) <= 2)
        if len(short_combo) >= 2 and short_combo not in LEGAL_STOPWORDS:
            tokens.append(f"n_acr_{short_combo}")
            
    if not tokens and words:
        tokens.append(f"n_{max(words, key=len)}")
        
    return tokens


def extract_address_tokens(address: Optional[str]) -> List[str]:
    """
    Extracts meaningful address blocking tokens:
    1. Distinctive street / locality names (length >= 3, not in address stopwords).
    2. Normalized numeric tokens with leading zeros stripped (e.g. 01111 -> 1111).
    3. 2-token compound shingles (e.g. "1795_westchester", "lucknow_gomtinagar").
    """
    norm = normalize_text(address)
    if not norm:
        return []
    words = norm.split()
    tokens: List[str] = []
    clean_words: List[str] = []
    
    for w in words:
        if w.isdigit():
            clean_num = w.lstrip("0")
            if clean_num:
                tokens.append(f"num_{clean_num}")
                clean_words.append(clean_num)
        elif len(w) >= 3 and w not in ADDR_STOPWORDS:
            tokens.append(f"w_{w}")
            clean_words.append(w)
            
    # Compound 2-token address shingles
    for i in range(len(clean_words) - 1):
        w1, w2 = clean_words[i], clean_words[i + 1]
        tokens.append(f"sh_{w1}_{w2}")
        
    return tokens


def extract_name_ngrams(name: Optional[str], n: int = 4) -> List[str]:
    """
    Extracts character n-grams from the normalized business name without spaces.
    Provides robustness against typos and word concatenation.
    """
    norm = normalize_text(name).replace(" ", "")
    if not norm:
        return []
    if len(norm) < n:
        return [f"ng_{norm}"]
    return [f"ng_{norm[i:i + n]}" for i in range(len(norm) - n + 1)]
