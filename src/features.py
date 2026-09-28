"""
Enhanced Pairwise Feature Engineering Module for Business Entity Resolution (Stage 3).

Extracts 34 dense discriminative features comparing Source 1 and candidate records:
- Lexical & String Edit Similarities (Levenshtein, Jaro-Winkler, Token Sort, LCS, N-grams)
- Phonetic & Acronym Similarities (Soundex, Acronym overlap, Prefix matching)
- Token & Shingle Overlaps (Jaccard, Containment, 2-token compound shingles, word-only Jaccard)
- House Number Logic (Match, Conflict flag, Overlap count)
- Cross-Field & Source Interactions (Name-in-Address, Harmonic Mean JW, Source 3 flag)
"""

from typing import Dict, List, Set, Optional, Tuple, Any
import numpy as np
from .normalization import normalize_text, normalize_country, standardize_address_text
from .tokenization import (
    extract_name_tokens,
    extract_address_tokens,
    extract_name_ngrams
)


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def containment_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes containment similarity: intersection / min(len(a), len(b))."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    min_len = min(len(set_a), len(set_b))
    return intersection / min_len if min_len > 0 else 0.0


def char_ngram_jaccard(str_a: str, str_b: str, n: int = 3) -> float:
    """Computes Jaccard similarity over character n-grams."""
    if not str_a or not str_b:
        return 0.0
    ng_a = set(str_a[i:i + n] for i in range(len(str_a) - n + 1)) if len(str_a) >= n else {str_a}
    ng_b = set(str_b[i:i + n] for i in range(len(str_b) - n + 1)) if len(str_b) >= n else {str_b}
    return jaccard_similarity(ng_a, ng_b)


def normalized_edit_similarity(str_a: str, str_b: str) -> float:
    """
    Computes normalized Levenshtein edit similarity in [0.0, 1.0]
    with common prefix and suffix trimming for maximum throughput.
    """
    if str_a == str_b:
        return 1.0
    if not str_a or not str_b:
        return 0.0
    len_a, len_b = len(str_a), len(str_b)
    max_len = max(len_a, len_b)
    if max_len == 0:
        return 1.0
        
    start = 0
    min_len = min(len_a, len_b)
    while start < min_len and str_a[start] == str_b[start]:
        start += 1
    if start == min_len:
        return max(0.0, 1.0 - (abs(len_a - len_b) / max_len))
        
    end_a, end_b = len_a, len_b
    while end_a > start and end_b > start and str_a[end_a - 1] == str_b[end_b - 1]:
        end_a -= 1
        end_b -= 1
        
    sub_a = str_a[start:end_a]
    sub_b = str_b[start:end_b]
    sub_len_a = len(sub_a)
    sub_len_b = len(sub_b)
    
    if sub_len_a == 0:
        return max(0.0, 1.0 - (sub_len_b / max_len))
    if sub_len_b == 0:
        return max(0.0, 1.0 - (sub_len_a / max_len))
        
    prev_row = list(range(sub_len_b + 1))
    for i, c_a in enumerate(sub_a):
        curr_row = [i + 1] * (sub_len_b + 1)
        for j, c_b in enumerate(sub_b):
            cost = 0 if c_a == c_b else 1
            curr_row[j + 1] = min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost
            )
        prev_row = curr_row
    dist = prev_row[sub_len_b]
    return max(0.0, 1.0 - (dist / max_len))


def token_sort_similarity(str_a: str, str_b: str) -> float:
    """Computes edit similarity on alphabetically sorted word tokens."""
    sorted_a = " ".join(sorted(str_a.split()))
    sorted_b = " ".join(sorted(str_b.split()))
    return normalized_edit_similarity(sorted_a, sorted_b)


def longest_common_subsequence(s1: str, s2: str) -> int:
    """Computes length of longest common subsequence."""
    if not s1 or not s2:
        return 0
    m, n = len(s1), len(s2)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def lcs_ratio(s1: str, s2: str) -> float:
    """Computes LCS similarity ratio: (2 * LCS) / (len(s1) + len(s2))."""
    if not s1 or not s2:
        return 0.0
    lcs_len = longest_common_subsequence(s1, s2)
    return (2.0 * lcs_len) / (len(s1) + len(s2))


def jaro_similarity(s1: str, s2: str) -> float:
    """Computes standard Jaro similarity."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    match1 = [False] * len1
    match2 = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if match2[j] or s1[i] != s2[j]:
                continue
            match1[i] = True
            match2[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len1):
        if not match1[i]:
            continue
        while not match2[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Computes Jaro-Winkler similarity with prefix bonus."""
    jaro = jaro_similarity(s1, s2)
    if jaro < 0.7:
        return jaro
    prefix_len = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    return jaro + prefix_len * prefix_weight * (1.0 - jaro)


def soundex(name: str) -> str:
    """Computes standard Soundex code for phonetic matching."""
    if not name:
        return ""
    name = name.upper()
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    soundex_digits = [name[0]]
    prev = mapping.get(name[0], '0')
    for char in name[1:]:
        digit = mapping.get(char, '0')
        if digit != '0' and digit != prev:
            soundex_digits.append(digit)
        prev = digit
    soundex_code = "".join(soundex_digits).replace('0', '')
    return (soundex_code + "0000")[:4]


def extract_house_numbers(addr_or_tokens: Any) -> Set[str]:
    """Extracts numeric house/building numbers."""
    if isinstance(addr_or_tokens, set):
        return {t.replace("num_", "") for t in addr_or_tokens if t.startswith("num_")}
    tokens = extract_address_tokens(addr_or_tokens)
    return {t.replace("num_", "") for t in tokens if t.startswith("num_")}


FEATURE_NAMES: List[str] = [
    # Business Name Features (13)
    "name_exact_match",
    "name_jaccard",
    "name_containment",
    "name_ngram_jaccard_3",
    "name_ngram_jaccard_4",
    "name_edit_sim",
    "name_token_sort_sim",
    "name_jaro_winkler",
    "name_lcs_ratio",
    "name_len_ratio",
    "name_acronym_match",
    "name_prefix_match",
    "name_soundex_match",
    # Address Features (13)
    "addr_null",
    "addr_exact_match",
    "addr_jaccard",
    "addr_containment",
    "addr_word_jaccard",
    "addr_word_containment",
    "addr_edit_sim",
    "addr_jaro_winkler",
    "addr_house_num_match",
    "addr_house_num_conflict",
    "addr_house_num_overlap_count",
    "addr_shingle_overlap_count",
    "addr_len_ratio",
    # Cross & Joint Features (5)
    "name_in_addr_cross_match",
    "addr_in_name_cross_match",
    "joint_name_addr_sim",
    "harmonic_mean_jw",
    "shared_tokens_total_count",
    # Source & Country Features (2)
    "country_match",
    "is_source3"
]


def extract_pair_features(
    s1_name: Optional[str],
    s1_addr: Optional[str],
    s1_country: Optional[str],
    cand_name: Optional[str],
    cand_addr: Optional[str],
    cand_country: Optional[str],
    cand_id: Optional[str] = None
) -> List[float]:
    """
    Extracts dense 34-dimensional feature vector comparing an S1 entity
    and a candidate S2/S3 entity.
    """
    n1 = normalize_text(s1_name)
    n2 = normalize_text(cand_name)
    a1 = standardize_address_text(s1_addr)
    a2 = standardize_address_text(cand_addr)
    c1 = normalize_country(s1_country)
    c2 = normalize_country(cand_country)
    
    # 1. Business Name Features (13)
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0
    n1_toks = set(extract_name_tokens(n1))
    n2_toks = set(extract_name_tokens(n2))
    name_jacc = jaccard_similarity(n1_toks, n2_toks)
    name_contain = containment_similarity(n1_toks, n2_toks)
    name_ng3 = char_ngram_jaccard(n1, n2, n=3)
    name_ng4 = char_ngram_jaccard(n1, n2, n=4)
    name_edit = normalized_edit_similarity(n1, n2)
    name_sort = token_sort_similarity(n1, n2)
    name_jw = jaro_winkler_similarity(n1, n2)
    name_lcs = lcs_ratio(n1, n2)
    
    len1, len2 = len(n1), len(n2)
    name_len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0
    
    acr1 = {t.replace("n_acr_", "") for t in n1_toks if t.startswith("n_acr_")}
    acr2 = {t.replace("n_acr_", "") for t in n2_toks if t.startswith("n_acr_")}
    name_acr = 1.0 if (acr1 and acr2 and bool(acr1 & acr2)) else 0.0
    
    name_pref = 1.0 if (n1 and n2 and (n1.startswith(n2[:4]) or n2.startswith(n1[:4]))) else 0.0
    sx1 = soundex(n1.split()[0]) if n1 else ""
    sx2 = soundex(n2.split()[0]) if n2 else ""
    soundex_match = 1.0 if (sx1 and sx2 and sx1 == sx2) else 0.0
    
    # 2. Address Features (13)
    addr_null = 1.0 if not a2 else 0.0
    addr_exact = 1.0 if (a1 and a2 and a1 == a2) else 0.0
    a1_toks = set(extract_address_tokens(a1))
    a2_toks = set(extract_address_tokens(a2))
    addr_jacc = jaccard_similarity(a1_toks, a2_toks)
    addr_contain = containment_similarity(a1_toks, a2_toks)
    
    s1_words = {t for t in a1_toks if t.startswith("w_")}
    c_words = {t for t in a2_toks if t.startswith("w_")}
    addr_word_jacc = jaccard_similarity(s1_words, c_words)
    addr_word_contain = containment_similarity(s1_words, c_words)
    
    addr_edit = normalized_edit_similarity(a1, a2) if (a1 and a2) else 0.0
    addr_jw = jaro_winkler_similarity(a1, a2) if (a1 and a2) else 0.0
    
    nums1 = extract_house_numbers(a1_toks)
    nums2 = extract_house_numbers(a2_toks)
    num_overlap = len(nums1 & nums2)
    if nums1 and nums2:
        num_match = 1.0 if num_overlap > 0 else 0.0
        num_conflict = 1.0 if num_overlap == 0 else 0.0
    elif not nums1 and not nums2:
        num_match = 0.5
        num_conflict = 0.0
    else:
        num_match = 0.0
        num_conflict = 0.0
        
    sh1 = {t for t in a1_toks if t.startswith("sh_")}
    sh2 = {t for t in a2_toks if t.startswith("sh_")}
    shingle_overlap = float(len(sh1 & sh2))
    
    alen1, alen2 = len(a1), len(a2)
    addr_len_ratio = (min(alen1, alen2) / max(alen1, alen2)) if max(alen1, alen2) > 0 else 0.0
    
    # 3. Cross & Joint Features (5)
    name_in_addr = 1.0 if any(t in a2 for t in n1.split() if len(t) >= 4) else 0.0
    addr_in_name = 1.0 if any(t in n2 for t in a1.split() if len(t) >= 4) else 0.0
    joint_sim = (name_jw + addr_jw) / 2.0 if a2 else name_jw
    
    if a2 and (name_jw + addr_jw) > 0:
        harmonic_jw = 2.0 * (name_jw * addr_jw) / (name_jw + addr_jw)
    else:
        harmonic_jw = name_jw
        
    total_shared_tokens = float(len(n1_toks & n2_toks) + len(a1_toks & a2_toks))
    
    # 4. Source & Country Features (2)
    cntry_match = 1.0 if (c1 == c2 or c1 == "UNKNOWN" or c2 == "UNKNOWN") else 0.0
    is_s3 = 1.0 if (cand_id and str(cand_id).startswith("S3-")) else 0.0
    
    return [
        name_exact, name_jacc, name_contain, name_ng3, name_ng4, name_edit, name_sort, name_jw, name_lcs, name_len_ratio, name_acr, name_pref, soundex_match,
        addr_null, addr_exact, addr_jacc, addr_contain, addr_word_jacc, addr_word_contain, addr_edit, addr_jw, num_match, num_conflict, float(num_overlap), shingle_overlap, addr_len_ratio,
        name_in_addr, addr_in_name, joint_sim, harmonic_jw, total_shared_tokens,
        cntry_match, is_s3
    ]
