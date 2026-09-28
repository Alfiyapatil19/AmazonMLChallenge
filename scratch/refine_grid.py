import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import csv
import numpy as np

from scratch.test_pipeline_enhancements import (
    normalize_text, normalize_country, standardize_address_text,
    extract_name_tokens, extract_address_tokens, CountryInvertedIndex,
    extract_enhanced_features, compute_macro_f05, compute_f05_score_per_entity, xgb
)

sys.stdout.reconfigure(encoding='utf-8')

def run_fine_tuning(n_entities: int = 10000, val_ratio: float = 0.3):
    print("=" * 80)
    print("FINE-GRAINED MACRO F_0.5 OPTIMIZATION (10,000 ENTITY SPLIT)")
    print("=" * 80)
    
    # 1. Load Ground Truth
    gt_map = {}
    s1_ids = []
    needed_s2 = set()
    needed_s3 = set()
    
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
                    
    # 2. Ingest S1
    s1_id_set = set(s1_ids)
    s1_records = {}
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
                    
    # 3. Ingest Targets
    target_records = {}
    s2_found, s2_bg_count, s2_bg_limit = 0, 0, 40000
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
                    
    s3_found, s3_bg_count, s3_bg_limit = 0, 0, 40000
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
                    
    # 4. Inverted Index
    idx = CountryInvertedIndex(country="ALL")
    for eid, rec in target_records.items():
        idx.add_record(eid, rec['name'], rec['addr'])
        
    n_train = int(len(s1_ids) * (1.0 - val_ratio))
    train_s1 = s1_ids[:n_train]
    val_s1 = s1_ids[n_train:]
    val_gt = {eid: gt_map[eid] for eid in val_s1}
    
    # 5. Extract Train
    X_train, y_train = [], []
    for eid in train_s1:
        s1_r = s1_records[eid]
        true_m = gt_map.get(eid, set())
        cands = idx.query_candidates(s1_r['name'], s1_r['addr'])
        cands.discard(eid)
        for m_id in true_m:
            if m_id in target_records:
                m_r = target_records[m_id]
                feat = extract_enhanced_features(
                    s1_r['name'], s1_r['addr'], s1_r['country'], s1_r['n_toks'], s1_r['a_toks'],
                    m_id, m_r['name'], m_r['addr'], m_r['country'], m_r['n_toks'], m_r['a_toks']
                )
                X_train.append(feat)
                y_train.append(1)
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
                    
    # Train Model
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=3, reg_alpha=0.15, reg_lambda=1.2,
        random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=-1
    )
    model.fit(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32))
    
    # Extract Val
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
    
    val_cand_scores = {}
    for i in range(len(val_pair_eids)):
        eid = val_pair_eids[i]
        cid = val_pair_cids[i]
        prob = float(p_val_all[i])
        if eid not in val_cand_scores:
            val_cand_scores[eid] = []
        val_cand_scores[eid].append((cid, prob))
        
    for eid in val_s1:
        if eid not in val_cand_scores:
            val_cand_scores[eid] = []
            
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

    print("\nFine-Grained Grid Search:")
    best_f05 = -1.0
    best_config = None
    best_metrics = {}
    
    for tau in [0.950, 0.960, 0.965, 0.970, 0.975, 0.980, 0.982, 0.985]:
        for delta in [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025]:
            for max_k in [4, 5, 6]:
                preds = {}
                for eid in val_s1:
                    pairs = val_cand_scores[eid]
                    if not pairs:
                        preds[eid] = set()
                        continue
                    p_max = max(p for _, p in pairs)
                    if p_max < tau:
                        preds[eid] = set()
                        continue
                    accepted = [(cid, p) for cid, p in pairs if p >= tau and (p >= p_max - delta)]
                    accepted.sort(key=lambda x: x[1], reverse=True)
                    preds[eid] = {cid for cid, _ in accepted[:max_k]}
                    
                f05, p, r, s_acc = evaluate(preds)
                if f05 > best_f05:
                    best_f05 = f05
                    best_config = (tau, delta, max_k)
                    best_metrics = {"f05": f05, "p": p, "r": r, "s_acc": s_acc}
                    print(f"  --> BEST: tau={tau:.3f}, delta={delta:.3f}, max_k={max_k} | Macro F0.5={f05*100:.2f}% | P={p*100:.2f}% | R={r*100:.2f}% | S-Acc={s_acc*100:.2f}%")
                    
    print("\n" + "=" * 80)
    print("FINAL ULTRA-HIGH PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Optimal Configuration:     tau={best_config[0]:.3f}, delta={best_config[1]:.3f}, max_k={best_config[2]}")
    print(f"Macro-Averaged F_0.5:      {best_metrics['f05']*100:.2f}% ({best_metrics['f05']:.4f})")
    print(f"Pairwise Precision:        {best_metrics['p']*100:.2f}%")
    print(f"Pairwise Recall:           {best_metrics['r']*100:.2f}%")
    print(f"Singleton Accuracy:        {best_metrics['s_acc']*100:.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    run_fine_tuning(n_entities=10000, val_ratio=0.3)
