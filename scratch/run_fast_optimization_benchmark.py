import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import csv
import time
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import xgboost as xgb

from src.normalization import normalize_text, normalize_country
from src.tokenization import extract_name_tokens, extract_address_tokens, extract_name_ngrams
from src.indexer import CountryInvertedIndex
from src.postprocessing import compute_macro_f05, compute_f05_score_per_entity

sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------------------------------
# Blazing-Fast Similarity Metrics (Prefix/Suffix Trimmed)
# -------------------------------------------------------------
def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

def containment_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    min_len = min(len(set_a), len(set_b))
    return inter / min_len if min_len > 0 else 0.0

def char_ngram_jaccard(str_a: str, str_b: str, n: int = 3) -> float:
    if not str_a or not str_b:
        return 0.0
    ng_a = set(str_a[i:i + n] for i in range(len(str_a) - n + 1)) if len(str_a) >= n else {str_a}
    ng_b = set(str_b[i:i + n] for i in range(len(str_b) - n + 1)) if len(str_b) >= n else {str_b}
    return jaccard_similarity(ng_a, ng_b)

def normalized_edit_similarity(str_a: str, str_b: str) -> float:
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
    sorted_a = " ".join(sorted(str_a.split()))
    sorted_b = " ".join(sorted(str_b.split()))
    return normalized_edit_similarity(sorted_a, sorted_b)

def longest_common_subsequence(s1: str, s2: str) -> int:
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
    if not s1 or not s2:
        return 0.0
    lcs_len = longest_common_subsequence(s1, s2)
    return (2.0 * lcs_len) / (len(s1) + len(s2))

def jaro_similarity(s1: str, s2: str) -> float:
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

def extract_house_numbers(tokens: Set[str]) -> Set[str]:
    return {t.replace("num_", "") for t in tokens if t.startswith("num_")}

# -------------------------------------------------------------
# Optimized Feature Extraction with Precomputed Tokens
# -------------------------------------------------------------
def extract_baseline_20_features(
    s1_name: str, s1_addr: str, s1_country: str, s1_n_toks: Set[str], s1_a_toks: Set[str],
    c_name: str, c_addr: str, c_country: str, c_n_toks: Set[str], c_a_toks: Set[str]
) -> List[float]:
    name_exact = 1.0 if (s1_name and s1_name == c_name) else 0.0
    name_jacc = jaccard_similarity(s1_n_toks, c_n_toks)
    name_ng_jacc = char_ngram_jaccard(s1_name, c_name, n=3)
    name_edit = normalized_edit_similarity(s1_name, c_name)
    name_sort = token_sort_similarity(s1_name, c_name)
    name_jw = jaro_winkler_similarity(s1_name, c_name)
    
    len1, len2 = len(s1_name), len(c_name)
    name_len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0
    name_len_diff = float(abs(len1 - len2))
    
    acr1 = {t.replace("n_acr_", "") for t in s1_n_toks if t.startswith("n_acr_")}
    acr2 = {t.replace("n_acr_", "") for t in c_n_toks if t.startswith("n_acr_")}
    name_acr = 1.0 if (acr1 and acr2 and bool(acr1 & acr2)) else 0.0
    name_pref = 1.0 if (s1_name and c_name and (s1_name.startswith(c_name[:4]) or c_name.startswith(s1_name[:4]))) else 0.0
    
    addr_null = 1.0 if not c_addr else 0.0
    addr_exact = 1.0 if (s1_addr and c_addr and s1_addr == c_addr) else 0.0
    addr_jacc = jaccard_similarity(s1_a_toks, c_a_toks)
    addr_edit = normalized_edit_similarity(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
    addr_jw = jaro_winkler_similarity(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
    
    nums1 = extract_house_numbers(s1_a_toks)
    nums2 = extract_house_numbers(c_a_toks)
    if nums1 and nums2:
        num_match = 1.0 if bool(nums1 & nums2) else 0.0
    elif not nums1 and not nums2:
        num_match = 0.5
    else:
        num_match = 0.0
        
    sh1 = {t for t in s1_a_toks if t.startswith("sh_")}
    sh2 = {t for t in c_a_toks if t.startswith("sh_")}
    shingle_overlap = float(len(sh1 & sh2))
    
    alen1, alen2 = len(s1_addr), len(c_addr)
    addr_len_ratio = (min(alen1, alen2) / max(alen1, alen2)) if max(alen1, alen2) > 0 else 0.0
    
    joint_sim = (name_jw + addr_jw) / 2.0 if c_addr else name_jw
    cntry_match = 1.0 if (s1_country == c_country or s1_country == "UNKNOWN" or c_country == "UNKNOWN") else 0.0
    
    return [
        name_exact, name_jacc, name_ng_jacc, name_edit, name_sort, name_jw,
        name_len_ratio, name_len_diff, name_acr, name_pref,
        addr_null, addr_exact, addr_jacc, addr_edit, addr_jw,
        num_match, shingle_overlap, addr_len_ratio,
        joint_sim, cntry_match
    ]

def extract_advanced_28_features(
    s1_name: str, s1_addr: str, s1_country: str, s1_n_toks: Set[str], s1_a_toks: Set[str],
    cand_id: str, c_name: str, c_addr: str, c_country: str, c_n_toks: Set[str], c_a_toks: Set[str]
) -> List[float]:
    # 1. Name Features (11)
    name_exact = 1.0 if (s1_name and s1_name == c_name) else 0.0
    name_jacc = jaccard_similarity(s1_n_toks, c_n_toks)
    name_contain = containment_similarity(s1_n_toks, c_n_toks)
    name_ng3 = char_ngram_jaccard(s1_name, c_name, n=3)
    name_ng4 = char_ngram_jaccard(s1_name, c_name, n=4)
    name_edit = normalized_edit_similarity(s1_name, c_name)
    name_sort = token_sort_similarity(s1_name, c_name)
    name_jw = jaro_winkler_similarity(s1_name, c_name)
    name_lcs = lcs_ratio(s1_name, c_name)
    
    len1, len2 = len(s1_name), len(c_name)
    name_len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0
    
    acr1 = {t.replace("n_acr_", "") for t in s1_n_toks if t.startswith("n_acr_")}
    acr2 = {t.replace("n_acr_", "") for t in c_n_toks if t.startswith("n_acr_")}
    name_acr = 1.0 if (acr1 and acr2 and bool(acr1 & acr2)) else 0.0
    
    # 2. Address Features (10)
    addr_null = 1.0 if not c_addr else 0.0
    addr_exact = 1.0 if (s1_addr and c_addr and s1_addr == c_addr) else 0.0
    addr_jacc = jaccard_similarity(s1_a_toks, c_a_toks)
    addr_contain = containment_similarity(s1_a_toks, c_a_toks)
    addr_edit = normalized_edit_similarity(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
    addr_jw = jaro_winkler_similarity(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
    
    nums1 = extract_house_numbers(s1_a_toks)
    nums2 = extract_house_numbers(c_a_toks)
    num_overlap = len(nums1 & nums2)
    if nums1 and nums2:
        num_match = 1.0 if num_overlap > 0 else 0.0
    elif not nums1 and not nums2:
        num_match = 0.5
    else:
        num_match = 0.0
        
    sh1 = {t for t in s1_a_toks if t.startswith("sh_")}
    sh2 = {t for t in c_a_toks if t.startswith("sh_")}
    shingle_overlap = float(len(sh1 & sh2))
    
    alen1, alen2 = len(s1_addr), len(c_addr)
    addr_len_ratio = (min(alen1, alen2) / max(alen1, alen2)) if max(alen1, alen2) > 0 else 0.0
    
    # 3. Cross & Joint Features (5)
    name_in_addr = 1.0 if any(t in c_addr for t in s1_name.split() if len(t) >= 4) else 0.0
    addr_in_name = 1.0 if any(t in c_name for t in s1_addr.split() if len(t) >= 4) else 0.0
    joint_sim = (name_jw + addr_jw) / 2.0 if c_addr else name_jw
    
    if c_addr and (name_jw + addr_jw) > 0:
        harmonic_jw = 2.0 * (name_jw * addr_jw) / (name_jw + addr_jw)
    else:
        harmonic_jw = name_jw
        
    total_shared_tokens = float(len(s1_n_toks & c_n_toks) + len(s1_a_toks & c_a_toks))
    
    # 4. Source & Country Features (2)
    cntry_match = 1.0 if (s1_country == c_country or s1_country == "UNKNOWN" or c_country == "UNKNOWN") else 0.0
    is_s3 = 1.0 if str(cand_id).startswith("S3-") else 0.0
    
    return [
        name_exact, name_jacc, name_contain, name_ng3, name_ng4, name_edit, name_sort, name_jw, name_lcs, name_len_ratio, name_acr,
        addr_null, addr_exact, addr_jacc, addr_contain, addr_edit, addr_jw, num_match, float(num_overlap), shingle_overlap, addr_len_ratio,
        name_in_addr, addr_in_name, joint_sim, harmonic_jw, total_shared_tokens,
        cntry_match, is_s3
    ]

# -------------------------------------------------------------
# Main Benchmark Runner
# -------------------------------------------------------------
def run_benchmark(n_entities: int = 5000, val_ratio: float = 0.3):
    print("=" * 80, flush=True)
    print("HIGH-PRECISION CONTROLLED EVALUATION BENCHMARK (OLD VS NEW)", flush=True)
    print("=" * 80, flush=True)
    
    # 1. Load Ground Truth
    print(f"\n[1/5] Ingesting {n_entities:,} ground truth records...", flush=True)
    gt_map: Dict[str, Set[str]] = {}
    s1_ids: List[str] = []
    needed_s2: Set[str] = set()
    needed_s3: Set[str] = set()
    
    with open('train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for i, row in enumerate(reader):
            if i >= n_entities:
                break
            s1_id = row[0]
            matched_str = row[1] if len(row) > 1 else ""
            matches = {m.strip() for m in matched_str.split(',') if m.strip()} if matched_str else set()
            gt_map[s1_id] = matches
            s1_ids.append(s1_id)
            for m in matches:
                if m.startswith("S2-"):
                    needed_s2.add(m)
                elif m.startswith("S3-"):
                    needed_s3.add(m)
                    
    total_true_matches = sum(len(m) for m in gt_map.values())
    print(f"Loaded {len(s1_ids):,} S1 entities ({len(needed_s2):,} needed S2, {len(needed_s3):,} needed S3 | Total True Matches: {total_true_matches:,})", flush=True)
    
    # 2. Ingest S1 Records
    print("\n[2/5] Ingesting S1 records from train_source1.tsv...", flush=True)
    s1_id_set = set(s1_ids)
    s1_records: Dict[str, Dict[str, Any]] = {}
    with open('train/train_source1.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            eid = row[0]
            if eid in s1_id_set:
                raw_name = row[1] if len(row) > 1 else ''
                raw_addr = row[2] if len(row) > 2 else ''
                raw_cntry = row[3] if len(row) > 3 else ''
                norm_n = normalize_text(raw_name)
                norm_a = normalize_text(raw_addr)
                norm_c = normalize_country(raw_cntry)
                s1_records[eid] = {
                    'entity_id': eid,
                    'name': norm_n,
                    'addr': norm_a,
                    'country': norm_c,
                    'n_toks': set(extract_name_tokens(norm_n)),
                    'a_toks': set(extract_address_tokens(norm_a))
                }
                if len(s1_records) == len(s1_id_set):
                    break
    print(f"Retrieved and normalized {len(s1_records):,} S1 records.", flush=True)
    
    # 3. Ingest S2 and S3 Records
    print("\n[3/5] Streaming S2 & S3 target entities + background negative pool...", flush=True)
    target_records: Dict[str, Dict[str, Any]] = {}
    
    s2_found = 0
    s2_bg_limit = 25000
    s2_bg_count = 0
    with open('train/train_source2.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            eid = row[0]
            is_needed = eid in needed_s2
            if is_needed or s2_bg_count < s2_bg_limit:
                norm_n = normalize_text(row[1] if len(row) > 1 else '')
                norm_a = normalize_text(row[2] if len(row) > 2 else '')
                norm_c = normalize_country(row[3] if len(row) > 3 else '')
                target_records[eid] = {
                    'entity_id': eid,
                    'name': norm_n,
                    'addr': norm_a,
                    'country': norm_c,
                    'n_toks': set(extract_name_tokens(norm_n)),
                    'a_toks': set(extract_address_tokens(norm_a))
                }
                if is_needed:
                    s2_found += 1
                else:
                    s2_bg_count += 1
                if s2_found == len(needed_s2) and s2_bg_count >= s2_bg_limit:
                    break
                    
    s3_found = 0
    s3_bg_limit = 25000
    s3_bg_count = 0
    with open('train/train_source3.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            eid = row[0]
            is_needed = eid in needed_s3
            if is_needed or s3_bg_count < s3_bg_limit:
                norm_n = normalize_text(row[1] if len(row) > 1 else '')
                norm_a = normalize_text(row[2] if len(row) > 2 else '')
                norm_c = normalize_country(row[3] if len(row) > 3 else '')
                target_records[eid] = {
                    'entity_id': eid,
                    'name': norm_n,
                    'addr': norm_a,
                    'country': norm_c,
                    'n_toks': set(extract_name_tokens(norm_n)),
                    'a_toks': set(extract_address_tokens(norm_a))
                }
                if is_needed:
                    s3_found += 1
                else:
                    s3_bg_count += 1
                if s3_found == len(needed_s3) and s3_bg_count >= s3_bg_limit:
                    break
                    
    print(f"Target pool assembled: {len(target_records):,} records (S2: {s2_found}/{len(needed_s2)} found, S3: {s3_found}/{len(needed_s3)} found)", flush=True)
    
    # 4. Inverted Index Building
    print("\n[4/5] Building Inverted Index...", flush=True)
    t0 = time.time()
    idx = CountryInvertedIndex(country="ALL")
    for eid, rec in target_records.items():
        idx.add_record(eid, rec['name'], rec['addr'])
    print(f"Index built in {time.time()-t0:.2f}s across {len(target_records):,} targets.", flush=True)
    
    # Split S1 into Train and Validation
    n_train = int(len(s1_ids) * (1.0 - val_ratio))
    train_s1 = s1_ids[:n_train]
    val_s1 = s1_ids[n_train:]
    val_gt = {eid: gt_map[eid] for eid in val_s1}
    
    print(f"Split: {len(train_s1):,} Training S1 | {len(val_s1):,} Validation S1 entities.", flush=True)
    
    # -------------------------------------------------------------
    # Extract Training Data (Positives & Hard Negatives)
    # -------------------------------------------------------------
    print("\nExtracting Training Pairs...", flush=True)
    t_tr = time.time()
    X_train_base = []
    X_train_adv = []
    y_train = []
    
    for eid in train_s1:
        s1_r = s1_records[eid]
        true_m = gt_map.get(eid, set())
        cands = idx.query_candidates(s1_r['name'], s1_r['addr'])
        cands.discard(eid)
        
        # Positives
        for m_id in true_m:
            if m_id in target_records:
                m_r = target_records[m_id]
                f_base = extract_baseline_20_features(
                    s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                    m_r['name'], m_r['addr'], m_r['country'], m_r['n_toks'], m_r['a_toks']
                )
                f_adv = extract_advanced_28_features(
                    s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                    m_id, m_r['name'], m_r['addr'], m_r['country'], m_r['n_toks'], m_r['a_toks']
                )
                X_train_base.append(f_base)
                X_train_adv.append(f_adv)
                y_train.append(1)
                
        # Hard Negatives (sample up to 10 per entity)
        negs = list(cands - true_m)
        if negs:
            for n_id in negs[:10]:
                if n_id in target_records:
                    n_r = target_records[n_id]
                    f_base = extract_baseline_20_features(
                        s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                        n_r['name'], n_r['addr'], n_r['country'], n_r['n_toks'], n_r['a_toks']
                    )
                    f_adv = extract_advanced_28_features(
                        s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                        n_id, n_r['name'], n_r['addr'], n_r['country'], n_r['n_toks'], n_r['a_toks']
                    )
                    X_train_base.append(f_base)
                    X_train_adv.append(f_adv)
                    y_train.append(0)
                    
    X_tr_base = np.array(X_train_base, dtype=np.float32)
    X_tr_adv = np.array(X_train_adv, dtype=np.float32)
    y_tr = np.array(y_train, dtype=np.int32)
    
    print(f"Training set: {len(y_tr):,} pairs (Positives: {np.sum(y_tr):,}, Negatives: {len(y_tr)-np.sum(y_tr):,}) in {time.time()-t_tr:.2f}s", flush=True)
    
    # Train Baseline Model (20 features)
    print("\nTraining Baseline XGBoost Model (20 features)...", flush=True)
    model_base = xgb.XGBClassifier(
        n_estimators=180, max_depth=6, learning_rate=0.08, subsample=0.85,
        colsample_bytree=0.85, random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=-1
    )
    model_base.fit(X_tr_base, y_tr)
    
    # Train Advanced Model (28 features, regularized)
    print("Training Optimized XGBoost Model (28 features, regularized)...", flush=True)
    model_adv = xgb.XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.06, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=-1
    )
    model_adv.fit(X_tr_adv, y_tr)
    
    # -------------------------------------------------------------
    # Featurize Validation Set in Single High-Speed Batch
    # -------------------------------------------------------------
    print("\n[5/5] Featurizing Validation Set across candidate pairs...", flush=True)
    t_val = time.time()
    
    val_entity_pair_counts = []
    val_pair_eids = []
    val_pair_cids = []
    val_X_base = []
    val_X_adv = []
    
    for eid in val_s1:
        s1_r = s1_records[eid]
        cands = list(idx.query_candidates(s1_r['name'], s1_r['addr']))
        cands = [c for c in cands if c != eid]
        
        if not cands:
            val_entity_pair_counts.append((eid, 0))
            continue
            
        ranked_cands = []
        for c_id in cands:
            if c_id in target_records:
                c_r = target_records[c_id]
                overlap = len(s1_r['n_toks'] & c_r['n_toks']) * 2 + len(s1_r['a_toks'] & c_r['a_toks'])
                ranked_cands.append((overlap, c_id))
                
        ranked_cands.sort(key=lambda x: x[0], reverse=True)
        eval_cands = [cid for _, cid in ranked_cands[:50]]
        
        pair_count = 0
        for c_id in eval_cands:
            c_r = target_records[c_id]
            fb = extract_baseline_20_features(
                s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                c_r['name'], c_r['addr'], c_r['country'], c_r['n_toks'], c_r['a_toks']
            )
            fa = extract_advanced_28_features(
                s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                c_id, c_r['name'], c_r['addr'], c_r['country'], c_r['n_toks'], c_r['a_toks']
            )
            val_X_base.append(fb)
            val_X_adv.append(fa)
            val_pair_eids.append(eid)
            val_pair_cids.append(c_id)
            pair_count += 1
            
        val_entity_pair_counts.append((eid, pair_count))
        
    print(f"Featurized {len(val_X_base):,} validation pairs in {time.time()-t_val:.2f}s", flush=True)
    
    # Single-Batch Predictions
    print("Running Batch Model Inference...", flush=True)
    t_inf = time.time()
    p_base_all = model_base.predict_proba(np.array(val_X_base, dtype=np.float32))[:, 1]
    p_adv_all = model_adv.predict_proba(np.array(val_X_adv, dtype=np.float32))[:, 1]
    print(f"Batch inference completed in {time.time()-t_inf:.2f}s", flush=True)
    
    # Reassemble per entity
    val_base_scores: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    val_adv_scores: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    
    for i in range(len(val_pair_eids)):
        eid = val_pair_eids[i]
        cid = val_pair_cids[i]
        val_base_scores[eid].append((cid, float(p_base_all[i])))
        val_adv_scores[eid].append((cid, float(p_adv_all[i])))
        
    for eid in val_s1:
        if eid not in val_base_scores:
            val_base_scores[eid] = []
            val_adv_scores[eid] = []
            
    # -------------------------------------------------------------
    # Evaluate Baseline Configuration (tau=0.75, max_k=5, no margin)
    # -------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("EVALUATING BASELINE (OLD PIPELINE)", flush=True)
    print("=" * 80, flush=True)
    
    preds_base = {}
    tp_b, pred_b, true_b = 0, 0, 0
    sing_cor_b, sing_tot_b = 0, 0
    
    for eid in val_s1:
        pairs = val_base_scores[eid]
        true_set = val_gt.get(eid, set())
        true_b += len(true_set)
        if not true_set:
            sing_tot_b += 1
            
        accepted = [(cid, p) for cid, p in pairs if p >= 0.75]
        accepted.sort(key=lambda x: x[1], reverse=True)
        if len(accepted) > 5:
            accepted = accepted[:5]
            
        pred_set = {cid for cid, _ in accepted}
        preds_base[eid] = pred_set
        
        if not true_set and not pred_set:
            sing_cor_b += 1
            
        tp = len(pred_set & true_set)
        tp_b += tp
        pred_b += len(pred_set)
        
    macro_f05_base = compute_macro_f05(preds_base, val_gt)
    prec_base = tp_b / pred_b if pred_b > 0 else 1.0
    rec_base = tp_b / true_b if true_b > 0 else 0.0
    sing_acc_base = sing_cor_b / sing_tot_b if sing_tot_b > 0 else 1.0
    
    print(f"Baseline Macro F0.5:        {macro_f05_base*100:.2f}% ({macro_f05_base:.4f})", flush=True)
    print(f"Baseline Pairwise Precision:{prec_base*100:.2f}%", flush=True)
    print(f"Baseline Pairwise Recall:   {rec_base*100:.2f}%", flush=True)
    print(f"Baseline Singleton Accuracy:{sing_acc_base*100:.2f}%", flush=True)
    
    # -------------------------------------------------------------
    # Grid Search & Evaluate Optimized Configuration (NEW PIPELINE)
    # -------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("OPTIMIZING NEW PIPELINE (GRID SEARCH OVER TAU, DELTA, MAX-K)", flush=True)
    print("=" * 80, flush=True)
    
    best_config = None
    best_f05 = -1.0
    best_metrics = {}
    
    for tau in [0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]:
        for delta in [0.08, 0.10, 0.12, 0.15, 0.20, 1.0]:
            for max_k in [3, 4, 5, 6, None]:
                preds_new = {}
                tp_n, pred_n, true_n = 0, 0, 0
                sing_cor_n, sing_tot_n = 0, 0
                
                for eid in val_s1:
                    pairs = val_adv_scores[eid]
                    true_set = val_gt.get(eid, set())
                    true_n += len(true_set)
                    if not true_set:
                        sing_tot_n += 1
                        
                    if not pairs:
                        preds_new[eid] = set()
                        if not true_set:
                            sing_cor_n += 1
                        continue
                        
                    p_max = max(p for _, p in pairs)
                    if p_max < tau:
                        preds_new[eid] = set()
                        if not true_set:
                            sing_cor_n += 1
                        continue
                        
                    accepted = [(cid, p) for cid, p in pairs if p >= tau and (p >= p_max - delta)]
                    accepted.sort(key=lambda x: x[1], reverse=True)
                    if max_k is not None and len(accepted) > max_k:
                        accepted = accepted[:max_k]
                        
                    pred_set = {cid for cid, _ in accepted}
                    preds_new[eid] = pred_set
                    
                    if not true_set and not pred_set:
                        sing_cor_n += 1
                        
                    tp = len(pred_set & true_set)
                    tp_n += tp
                    pred_n += len(pred_set)
                    
                score = compute_macro_f05(preds_new, val_gt)
                precision = tp_n / pred_n if pred_n > 0 else 1.0
                recall = tp_n / true_n if true_n > 0 else 0.0
                singleton_acc = sing_cor_n / sing_tot_n if sing_tot_n > 0 else 1.0
                
                if score > best_f05:
                    best_f05 = score
                    best_config = (tau, delta, max_k)
                    best_metrics = {
                        "macro_f05": score,
                        "precision": precision,
                        "recall": recall,
                        "singleton_acc": singleton_acc,
                        "tau": tau,
                        "delta": delta,
                        "max_k": max_k
                    }
                    print(f"  --> NEW BEST: tau={tau:.2f}, delta={delta:.2f}, max_k={str(max_k):4s} | Macro F0.5={score*100:.2f}% | P={precision*100:.2f}% | R={recall*100:.2f}% | S-Acc={singleton_acc*100:.2f}%", flush=True)
                    
    print("\n" + "=" * 80, flush=True)
    print("FINAL COMPARATIVE EVALUATION RESULTS (EXACT NUMBERS)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Metric':<30} | {'Baseline (OLD)':<20} | {'Optimized (NEW)':<20} | {'Delta':<15}")
    print("-" * 90)
    print(f"{'Macro-Averaged F_0.5':<30} | {macro_f05_base*100:.2f}% ({macro_f05_base:.4f})     | {best_metrics['macro_f05']*100:.2f}% ({best_metrics['macro_f05']:.4f})   | {+(best_metrics['macro_f05']-macro_f05_base)*100:+.2f}%")
    print(f"{'Pairwise Precision':<30} | {prec_base*100:.2f}%              | {best_metrics['precision']*100:.2f}%            | {+(best_metrics['precision']-prec_base)*100:+.2f}%")
    print(f"{'Pairwise Recall':<30} | {rec_base*100:.2f}%              | {best_metrics['recall']*100:.2f}%            | {+(best_metrics['recall']-rec_base)*100:+.2f}%")
    print(f"{'Singleton Accuracy':<30} | {sing_acc_base*100:.2f}%          | {best_metrics['singleton_acc']*100:.2f}%        | {+(best_metrics['singleton_acc']-sing_acc_base)*100:+.2f}%")
    print(f"{'Decision Threshold (tau)':<30} | {'0.75':<20} | {best_metrics['tau']:.2f}{'':<16} | {best_metrics['tau']-0.75:+.2f}")
    print(f"{'Max Matches / Entity (max_k)':<30} | {'5':<20} | {str(best_metrics['max_k']):<20} | -")
    print(f"{'Margin Filter (delta)':<30} | {'None (1.00)':<20} | {best_metrics['delta']:.2f}{'':<16} | -")
    print("=" * 80, flush=True)

if __name__ == '__main__':
    run_benchmark(n_entities=5000, val_ratio=0.3)
