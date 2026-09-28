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
from src.features import extract_pair_features, FEATURE_NAMES
from src.postprocessing import compute_macro_f05, compute_f05_score_per_entity

sys.stdout.reconfigure(encoding='utf-8')

def run_stage5_investigation(n_entities: int = 6000, val_ratio: float = 0.35):
    print("=" * 80)
    print("STAGE 5 DEEP DIVE: SINGLETON FALSE POSITIVE GATING & POST-PROCESSING")
    print("=" * 80)
    
    # 1. Load Ground Truth
    print(f"\n[1/4] Ingesting {n_entities:,} ground truth records...")
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
                    
    # 2. Ingest S1 Records
    s1_id_set = set(s1_ids)
    s1_records: Dict[str, Dict[str, Any]] = {}
    with open('train/train_source1.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            eid = row[0]
            if eid in s1_id_set:
                norm_n = normalize_text(row[1] if len(row) > 1 else '')
                norm_a = normalize_text(row[2] if len(row) > 2 else '')
                norm_c = normalize_country(row[3] if len(row) > 3 else '')
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
                    
    # 3. Ingest S2 & S3 Records
    target_records: Dict[str, Dict[str, Any]] = {}
    s2_found, s2_bg_count, s2_bg_limit = 0, 0, 30000
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
                    
    s3_found, s3_bg_count, s3_bg_limit = 0, 0, 30000
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
                    
    # 4. Inverted Index Building
    idx = CountryInvertedIndex(country="ALL")
    for eid, rec in target_records.items():
        idx.add_record(eid, rec['name'], rec['addr'])
        
    n_train = int(len(s1_ids) * (1.0 - val_ratio))
    train_s1 = s1_ids[:n_train]
    val_s1 = s1_ids[n_train:]
    val_gt = {eid: gt_map[eid] for eid in val_s1}
    
    print(f"Train Entities: {len(train_s1):,} | Validation Entities: {len(val_s1):,} (Singletons in Val: {sum(1 for v in val_gt.values() if not v):,})")
    
    # Extract Training Pairs
    print("\n[2/4] Extracting Training Pairs & Fitting Regularized XGBoost...")
    X_train, y_train = [], []
    for eid in train_s1:
        s1_r = s1_records[eid]
        true_m = gt_map.get(eid, set())
        cands = idx.query_candidates(s1_r['name'], s1_r['addr'])
        cands.discard(eid)
        for m_id in true_m:
            if m_id in target_records:
                m_r = target_records[m_id]
                feat = extract_pair_features(s1_r['name'], s1_r['addr'], s1_r['country'], m_r['name'], m_r['addr'], m_r['country'], cand_id=m_id)
                X_train.append(feat)
                y_train.append(1)
        negs = list(cands - true_m)
        if negs:
            for n_id in negs[:10]:
                if n_id in target_records:
                    n_r = target_records[n_id]
                    feat = extract_pair_features(s1_r['name'], s1_r['addr'], s1_r['country'], n_r['name'], n_r['addr'], n_r['country'], cand_id=n_id)
                    X_train.append(feat)
                    y_train.append(0)
                    
    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.06, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=-1
    )
    model.fit(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32))
    print(f"XGBoost Model fitted on {len(X_train):,} training pairs.")
    
    # Featurize Validation Set
    print("\n[3/4] Featurizing Validation Set...")
    val_pair_eids, val_pair_cids, val_X = [], [], []
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
        for cid in [x[1] for x in ranked[:50]]:
            c_r = target_records[cid]
            feat = extract_pair_features(s1_r['name'], s1_r['addr'], s1_r['country'], c_r['name'], c_r['addr'], c_r['country'], cand_id=cid)
            val_X.append(feat)
            val_pair_eids.append(eid)
            val_pair_cids.append(cid)
            
    p_val_all = model.predict_proba(np.array(val_X, dtype=np.float32))[:, 1]
    
    # Store candidate tuples: (cid, prob, name_sim, addr_sim, is_s3)
    val_cand_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i in range(len(val_pair_eids)):
        eid = val_pair_eids[i]
        cid = val_pair_cids[i]
        prob = float(p_val_all[i])
        feat = val_X[i]
        name_jw = feat[7]  # name_jaro_winkler
        addr_jw = feat[15] # addr_jaro_winkler
        val_cand_map[eid].append({
            'cid': cid,
            'prob': prob,
            'name_jw': name_jw,
            'addr_jw': addr_jw,
            'is_s3': cid.startswith("S3-")
        })
        
    for eid in val_s1:
        if eid not in val_cand_map:
            val_cand_map[eid] = []
            
    # Baseline Current (tau=0.90, delta=0.08, max_k=4)
    print("\n[4/4] Testing Advanced Post-Processing Strategies...")
    
    def eval_predictions(preds: Dict[str, Set[str]]):
        total_tp, total_pred, total_true = 0, 0, 0
        sing_cor, sing_tot = 0, 0
        for eid in val_s1:
            true_set = val_gt.get(eid, set())
            pred_set = preds.get(eid, set())
            total_true += len(true_set)
            total_pred += len(pred_set)
            tp = len(pred_set & true_set)
            total_tp += tp
            if not true_set:
                sing_tot += 1
                if not pred_set:
                    sing_cor += 1
        f05 = compute_macro_f05(preds, val_gt)
        p = total_tp / total_pred if total_pred > 0 else 1.0
        r = total_tp / total_true if total_true > 0 else 0.0
        s_acc = sing_cor / sing_tot if sing_tot > 0 else 1.0
        return f05, p, r, s_acc

    # Baseline Evaluation
    preds_base = {}
    for eid in val_s1:
        cands = val_cand_map[eid]
        if not cands:
            preds_base[eid] = set()
            continue
        p_max = max(c['prob'] for c in cands)
        if p_max < 0.90:
            preds_base[eid] = set()
            continue
        acc = [c['cid'] for c in cands if c['prob'] >= 0.90 and (c['prob'] >= p_max - 0.08)]
        acc = acc[:4]
        preds_base[eid] = set(acc)
        
    base_f05, base_p, base_r, base_s = eval_predictions(preds_base)
    print(f"Current Baseline: Macro F0.5 = {base_f05*100:.2f}% | Precision = {base_p*100:.2f}% | Recall = {base_r*100:.2f}% | Singleton Acc = {base_s*100:.2f}%")
    
    # -------------------------------------------------------------
    # Grid Search: Source-Partitioned Gating + High Precision Margin
    # -------------------------------------------------------------
    print("\nEvaluating Source-Aware Gating (At most 1 per Source partition: max 1 S2, max 1 S3 unless P >= 0.98):")
    best_exp_f05 = base_f05
    best_config = None
    best_metrics = {}
    
    for tau in [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96]:
        for delta in [0.03, 0.05, 0.08, 0.10]:
            for min_name_jw in [0.0, 0.60, 0.70, 0.80]:
                for source_partition_mode in [False, True]:
                    preds = {}
                    for eid in val_s1:
                        cands = val_cand_map[eid]
                        if not cands:
                            preds[eid] = set()
                            continue
                        p_max = max(c['prob'] for c in cands)
                        if p_max < tau:
                            preds[eid] = set()
                            continue
                        
                        # Filter by tau, delta, and min_name_jw
                        valid = [
                            c for c in cands
                            if c['prob'] >= tau and (c['prob'] >= p_max - delta) and (c['name_jw'] >= min_name_jw)
                        ]
                        valid.sort(key=lambda x: x['prob'], reverse=True)
                        
                        if source_partition_mode:
                            # Keep best S2 and best S3
                            s2_chosen = []
                            s3_chosen = []
                            for c in valid:
                                if not c['is_s3']:
                                    if len(s2_chosen) == 0 or c['prob'] >= 0.98:
                                        s2_chosen.append(c['cid'])
                                else:
                                    if len(s3_chosen) == 0 or c['prob'] >= 0.98:
                                        s3_chosen.append(c['cid'])
                            preds[eid] = set(s2_chosen[:2] + s3_chosen[:2])
                        else:
                            preds[eid] = {c['cid'] for c in valid[:4]}
                            
                    f05, p, r, s_acc = eval_predictions(preds)
                    if f05 > best_exp_f05:
                        best_exp_f05 = f05
                        best_config = (tau, delta, min_name_jw, source_partition_mode)
                        best_metrics = {"f05": f05, "p": p, "r": r, "s_acc": s_acc}
                        print(f"  --> NEW BEST: tau={tau:.2f}, delta={delta:.2f}, name_jw>={min_name_jw:.2f}, src_partition={source_partition_mode} | Macro F0.5={f05*100:.2f}% | P={p*100:.2f}% | R={r*100:.2f}% | S-Acc={s_acc*100:.2f}%")
                        
    print("\n" + "=" * 80)
    print("STAGE 5 OPTIMIZATION COMPARATIVE RESULTS")
    print("=" * 80)
    if best_config:
        print(f"Best Config: tau={best_config[0]:.2f}, delta={best_config[1]:.2f}, min_name_jw={best_config[2]:.2f}, src_partition={best_config[3]}")
        print(f"Macro F0.5:      {base_f05*100:.2f}% -> {best_metrics['f05']*100:.2f}% ({best_metrics['f05']-base_f05:+.2f}%)")
        print(f"Precision:       {base_p*100:.2f}% -> {best_metrics['p']*100:.2f}% ({best_metrics['p']-base_p:+.2f}%)")
        print(f"Recall:          {base_r*100:.2f}% -> {best_metrics['r']*100:.2f}% ({best_metrics['r']-base_r:+.2f}%)")
        print(f"Singleton Acc:   {base_s*100:.2f}% -> {best_metrics['s_acc']*100:.2f}% ({best_metrics['s_acc']-base_s:+.2f}%)")
    else:
        print("Current baseline remains optimal.")
    print("=" * 80)

if __name__ == '__main__':
    run_stage5_investigation(n_entities=6000, val_ratio=0.35)
