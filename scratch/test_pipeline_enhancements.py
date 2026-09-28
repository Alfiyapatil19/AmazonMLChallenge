import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import csv
import time
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np
import xgboost as xgb

from src.normalization import normalize_text, normalize_country
from src.tokenization import extract_name_tokens, extract_address_tokens, extract_name_ngrams
from src.indexer import CountryInvertedIndex
from src.postprocessing import compute_macro_f05, compute_f05_score_per_entity

sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------------------------------
# Address Suffix Standardizer
# -------------------------------------------------------------
STREET_ABBRS = {
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "pkwy": "parkway",
    "hwy": "highway", "cir": "circle", "ste": "suite", "apt": "apartment",
    "fl": "floor", "bldg": "building", "sq": "square", "ctr": "center"
}

def standardize_address_text(text: str) -> str:
    if not text:
        return ""
    words = text.split()
    norm_words = [STREET_ABBRS.get(w, w) for w in words]
    return " ".join(norm_words)

# -------------------------------------------------------------
# High-Speed Similarity Metrics
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

# Soundex implementation for phonetic similarity
def soundex(name: str) -> str:
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

# -------------------------------------------------------------
# Enhanced 34-Feature Vector Extractor
# -------------------------------------------------------------
def extract_enhanced_features(
    s1_name: str, s1_addr: str, s1_country: str, s1_n_toks: Set[str], s1_a_toks: Set[str],
    cand_id: str, c_name: str, c_addr: str, c_country: str, c_n_toks: Set[str], c_a_toks: Set[str]
) -> List[float]:
    # 1. Business Name Features (14)
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
    
    # Prefix match & soundex
    name_pref = 1.0 if (s1_name and c_name and (s1_name.startswith(c_name[:4]) or c_name.startswith(s1_name[:4]))) else 0.0
    sx1 = soundex(s1_name.split()[0]) if s1_name else ""
    sx2 = soundex(c_name.split()[0]) if c_name else ""
    soundex_match = 1.0 if (sx1 and sx2 and sx1 == sx2) else 0.0
    
    # 2. Address Features (13)
    addr_null = 1.0 if not c_addr else 0.0
    addr_exact = 1.0 if (s1_addr and c_addr and s1_addr == c_addr) else 0.0
    addr_jacc = jaccard_similarity(s1_a_toks, c_a_toks)
    addr_contain = containment_similarity(s1_a_toks, c_a_toks)
    
    # Street words only (excluding numbers and shingles)
    s1_words = {t for t in s1_a_toks if t.startswith("w_")}
    c_words = {t for t in c_a_toks if t.startswith("w_")}
    addr_word_jacc = jaccard_similarity(s1_words, c_words)
    addr_word_contain = containment_similarity(s1_words, c_words)
    
    addr_edit = normalized_edit_similarity(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
    addr_jw = jaro_winkler_similarity(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
    
    nums1 = {t.replace("num_", "") for t in s1_a_toks if t.startswith("num_")}
    nums2 = {t.replace("num_", "") for t in c_a_toks if t.startswith("num_")}
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
        name_exact, name_jacc, name_contain, name_ng3, name_ng4, name_edit, name_sort, name_jw, name_lcs, name_len_ratio, name_acr, name_pref, soundex_match,
        addr_null, addr_exact, addr_jacc, addr_contain, addr_word_jacc, addr_word_contain, addr_edit, addr_jw, num_match, num_conflict, float(num_overlap), shingle_overlap, addr_len_ratio,
        name_in_addr, addr_in_name, joint_sim, harmonic_jw, total_shared_tokens,
        cntry_match, is_s3
    ]

# -------------------------------------------------------------
# Investigation & Multi-Stage Benchmark Runner
# -------------------------------------------------------------
def run_comprehensive_experiment(n_entities: int = 8000, val_ratio: float = 0.3):
    print("=" * 80)
    print("COMPREHENSIVE PIPELINE ENHANCEMENT BENCHMARK")
    print("=" * 80)
    
    # 1. Load Ground Truth
    print(f"\n[1/5] Ingesting {n_entities:,} ground truth records...")
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
                    
    # 2. Ingest S1 Records with Street Standardization
    s1_id_set = set(s1_ids)
    s1_records: Dict[str, Dict[str, Any]] = {}
    with open('train/train_source1.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            eid = row[0]
            if eid in s1_id_set:
                raw_n = normalize_text(row[1] if len(row) > 1 else '')
                raw_a = standardize_address_text(normalize_text(row[2] if len(row) > 2 else ''))
                raw_c = normalize_country(row[3] if len(row) > 3 else '')
                s1_records[eid] = {
                    'entity_id': eid,
                    'name': raw_n,
                    'addr': raw_a,
                    'country': raw_c,
                    'n_toks': set(extract_name_tokens(raw_n)),
                    'a_toks': set(extract_address_tokens(raw_a))
                }
                if len(s1_records) == len(s1_id_set):
                    break
                    
    # 3. Ingest Target S2 & S3 Records
    target_records: Dict[str, Dict[str, Any]] = {}
    s2_found, s2_bg_count, s2_bg_limit = 0, 0, 35000
    with open('train/train_source2.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            eid = row[0]
            is_needed = eid in needed_s2
            if is_needed or s2_bg_count < s2_bg_limit:
                raw_n = normalize_text(row[1] if len(row) > 1 else '')
                raw_a = standardize_address_text(normalize_text(row[2] if len(row) > 2 else ''))
                raw_c = normalize_country(row[3] if len(row) > 3 else '')
                target_records[eid] = {
                    'entity_id': eid,
                    'name': raw_n,
                    'addr': raw_a,
                    'country': raw_c,
                    'n_toks': set(extract_name_tokens(raw_n)),
                    'a_toks': set(extract_address_tokens(raw_a))
                }
                if is_needed:
                    s2_found += 1
                else:
                    s2_bg_count += 1
                if s2_found == len(needed_s2) and s2_bg_count >= s2_bg_limit:
                    break
                    
    s3_found, s3_bg_count, s3_bg_limit = 0, 0, 35000
    with open('train/train_source3.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            eid = row[0]
            is_needed = eid in needed_s3
            if is_needed or s3_bg_count < s3_bg_limit:
                raw_n = normalize_text(row[1] if len(row) > 1 else '')
                raw_a = standardize_address_text(normalize_text(row[2] if len(row) > 2 else ''))
                raw_c = normalize_country(row[3] if len(row) > 3 else '')
                target_records[eid] = {
                    'entity_id': eid,
                    'name': raw_n,
                    'addr': raw_a,
                    'country': raw_c,
                    'n_toks': set(extract_name_tokens(raw_n)),
                    'a_toks': set(extract_address_tokens(raw_a))
                }
                if is_needed:
                    s3_found += 1
                else:
                    s3_bg_count += 1
                if s3_found == len(needed_s3) and s3_bg_count >= s3_bg_limit:
                    break
                    
    # 4. Inverted Index Building
    print("\n[2/5] Building Inverted Index...")
    t0 = time.time()
    idx = CountryInvertedIndex(country="ALL")
    for eid, rec in target_records.items():
        idx.add_record(eid, rec['name'], rec['addr'])
    print(f"Index built in {time.time()-t0:.2f}s across {len(target_records):,} target records.")
    
    n_train = int(len(s1_ids) * (1.0 - val_ratio))
    train_s1 = s1_ids[:n_train]
    val_s1 = s1_ids[n_train:]
    val_gt = {eid: gt_map[eid] for eid in val_s1}
    
    print(f"Train Entities: {len(train_s1):,} | Validation Entities: {len(val_s1):,} (Singletons in Val: {sum(1 for v in val_gt.values() if not v):,})")
    
    # 5. Feature Extraction for Training
    print("\n[3/5] Extracting 34 Enhanced Features for Training Set...")
    t_tr = time.time()
    X_train = []
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
                feat = extract_enhanced_features(
                    s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                    m_id, m_r['name'], m_r['addr'], m_r['country'], m_r['n_toks'], m_r['a_toks']
                )
                X_train.append(feat)
                y_train.append(1)
                
        # Hard Negatives
        negs = list(cands - true_m)
        if negs:
            for n_id in negs[:10]:
                if n_id in target_records:
                    n_r = target_records[n_id]
                    feat = extract_enhanced_features(
                        s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                        n_id, n_r['name'], n_r['addr'], n_r['country'], n_r['n_toks'], n_r['a_toks']
                    )
                    X_train.append(feat)
                    y_train.append(0)
                    
    X_train_np = np.array(X_train, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.int32)
    print(f"Training dataset: {len(y_train_np):,} pairs (Pos: {np.sum(y_train_np):,}, Neg: {len(y_train_np)-np.sum(y_train_np):,}) in {time.time()-t_tr:.2f}s")
    
    # Train Regularized XGBoost Model
    print("\n[4/5] Training Optimized XGBoost Model (34 features)...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=3, reg_alpha=0.15, reg_lambda=1.2,
        random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=-1
    )
    model.fit(X_train_np, y_train_np)
    
    # 6. Featurize Validation Set
    print("\n[5/5] Featurizing Validation Set & Running Inference...")
    t_val = time.time()
    val_pair_eids = []
    val_pair_cids = []
    val_X = []
    
    for eid in val_s1:
        s1_r = s1_records[eid]
        cands = list(idx.query_candidates(s1_r['name'], s1_r['addr']))
        cands = [c for c in cands if c != eid]
        if not cands:
            continue
            
        ranked = []
        for c_id in cands:
            if c_id in target_records:
                c_r = target_records[c_id]
                ov = len(s1_r['n_toks'] & c_r['n_toks']) * 2 + len(s1_r['a_toks'] & c_r['a_toks'])
                ranked.append((ov, c_id))
        ranked.sort(key=lambda x: x[0], reverse=True)
        
        for cid in [x[1] for x in ranked[:60]]:
            c_r = target_records[cid]
            feat = extract_enhanced_features(
                s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                cid, c_r['name'], c_r['addr'], c_r['country'], c_r['n_toks'], c_r['a_toks']
            )
            val_X.append(feat)
            val_pair_eids.append(eid)
            val_pair_cids.append(cid)
            
    p_val_all = model.predict_proba(np.array(val_X, dtype=np.float32))[:, 1]
    
    # Reassemble validation candidates
    val_cand_scores: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i in range(len(val_pair_eids)):
        eid = val_pair_eids[i]
        cid = val_pair_cids[i]
        prob = float(p_val_all[i])
        feat = val_X[i]
        num_conflict = feat[22] # house number conflict
        val_cand_scores[eid].append({
            'cid': cid,
            'prob': prob,
            'num_conflict': num_conflict,
            'is_s3': cid.startswith("S3-")
        })
        
    for eid in val_s1:
        if eid not in val_cand_scores:
            val_cand_scores[eid] = []
            
    print(f"Validation inference completed in {time.time()-t_val:.2f}s")
    
    # -------------------------------------------------------------
    # Evaluate Grid Search & Find Optimal Operating Point
    # -------------------------------------------------------------
    def evaluate(preds):
        tot_tp, tot_pred, tot_true = 0, 0, 0
        s_cor, s_tot = 0, 0
        for eid in val_s1:
            true_set = val_gt.get(eid, set())
            pred_set = preds.get(eid, set())
            tot_true += len(true_set)
            tot_pred += len(pred_set)
            tp = len(pred_set & true_set)
            tot_tp += tp
            if not true_set:
                s_tot += 1
                if not pred_set:
                    s_cor += 1
        score = compute_macro_f05(preds, val_gt)
        p = tot_tp / tot_pred if tot_pred > 0 else 1.0
        r = tot_tp / tot_true if tot_true > 0 else 0.0
        s_acc = s_cor / s_tot if s_tot > 0 else 1.0
        return score, p, r, s_acc

    # Evaluate Baseline rule (tau=0.96, delta=0.03, max_k=4)
    preds_baseline = {}
    for eid in val_s1:
        cands = val_cand_scores[eid]
        if not cands:
            preds_baseline[eid] = set()
            continue
        p_max = max(c['prob'] for c in cands)
        if p_max < 0.96:
            preds_baseline[eid] = set()
            continue
        acc = [c['cid'] for c in cands if c['prob'] >= 0.96 and (c['prob'] >= p_max - 0.03)]
        preds_baseline[eid] = set(acc[:4])
        
    b_f05, b_p, b_r, b_s = evaluate(preds_baseline)
    print("\n" + "=" * 80)
    print("VALIDATION EVALUATION: PREVIOUS BASELINE VS ENHANCED PIPELINE")
    print("=" * 80)
    print(f"Baseline on this split: Macro F0.5 = {b_f05*100:.2f}% | P = {b_p*100:.2f}% | R = {b_r*100:.2f}% | S-Acc = {b_s*100:.2f}%")
    
    # Grid search over tau, delta, conflict filter, and max_k
    best_f05 = b_f05
    best_config = None
    best_metrics = {}
    
    for tau in [0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98]:
        for delta in [0.015, 0.02, 0.025, 0.03, 0.04]:
            for filter_conflicts in [False, True]:
                for max_k in [3, 4, 5]:
                    preds = {}
                    for eid in val_s1:
                        cands = val_cand_scores[eid]
                        if not cands:
                            preds[eid] = set()
                            continue
                        p_max = max(c['prob'] for c in cands)
                        if p_max < tau:
                            preds[eid] = set()
                            continue
                            
                        valid = [
                            c for c in cands
                            if c['prob'] >= tau and (c['prob'] >= p_max - delta)
                        ]
                        if filter_conflicts:
                            valid = [c for c in valid if c['num_conflict'] == 0.0]
                            
                        valid.sort(key=lambda x: x['prob'], reverse=True)
                        preds[eid] = {c['cid'] for c in valid[:max_k]}
                        
                    f05, p, r, s_acc = evaluate(preds)
                    if f05 > best_f05:
                        best_f05 = f05
                        best_config = (tau, delta, filter_conflicts, max_k)
                        best_metrics = {"f05": f05, "p": p, "r": r, "s_acc": s_acc}
                        print(f"  --> NEW BEST: tau={tau:.3f}, delta={delta:.3f}, filter_conflict={filter_conflicts}, max_k={max_k} | Macro F0.5={f05*100:.2f}% | P={p*100:.2f}% | R={r*100:.2f}% | S-Acc={s_acc*100:.2f}%")
                        
    print("\n" + "=" * 80)
    print("FINAL EXPERIMENT RESULTS")
    print("=" * 80)
    if best_config:
        print(f"Best Configuration: tau={best_config[0]:.3f}, delta={best_config[1]:.3f}, filter_conflict={best_config[2]}, max_k={best_config[3]}")
        print(f"Macro-Averaged F_0.5:      {b_f05*100:.2f}% -> {best_metrics['f05']*100:.2f}% ({best_metrics['f05']-b_f05:+.2f}%)")
        print(f"Pairwise Precision:        {b_p*100:.2f}% -> {best_metrics['p']*100:.2f}% ({best_metrics['p']-b_p:+.2f}%)")
        print(f"Pairwise Recall:           {b_r*100:.2f}% -> {best_metrics['r']*100:.2f}% ({best_metrics['r']-b_r:+.2f}%)")
        print(f"Singleton Accuracy:        {b_s*100:.2f}% -> {best_metrics['s_acc']*100:.2f}% ({best_metrics['s_acc']-b_s:+.2f}%)")
    else:
        print("Previous baseline remains the best validated configuration.")
    print("=" * 80)

if __name__ == '__main__':
    run_comprehensive_experiment(n_entities=8000, val_ratio=0.3)
