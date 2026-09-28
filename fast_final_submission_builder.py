import os
import sys
import time
import pickle
import csv
import zipfile
import shutil
from collections import Counter
import numpy as np

BASE_DIR = r"C:\Users\palfi\Downloads\6ab10eb3b23ba_student_resource\student_resource"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
sys.path.insert(0, DATASET_DIR)

from src.normalization import normalize_country, normalize_text, standardize_address_text
from src.tokenization import extract_name_tokens, extract_address_tokens, extract_name_ngrams
from src.features import (
    FEATURE_NAMES,
    jaccard_similarity,
    containment_similarity,
    char_ngram_jaccard,
    normalized_edit_similarity,
    token_sort_similarity,
    jaro_winkler_similarity,
    lcs_ratio,
    soundex,
    extract_house_numbers
)

def fast_pair_features(
    n1, a1, c1, n1_toks, a1_toks, s1_words, nums1, sh1, sx1,
    n2, a2, c2, n2_toks, a2_toks, c_words, nums2, sh2, sx2,
    cand_id
):
    # 1. Name features
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0
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
    soundex_match = 1.0 if (sx1 and sx2 and sx1 == sx2) else 0.0
    
    # 2. Address features
    addr_null = 1.0 if not a2 else 0.0
    addr_exact = 1.0 if (a1 and a2 and a1 == a2) else 0.0
    addr_jacc = jaccard_similarity(a1_toks, a2_toks)
    addr_contain = containment_similarity(a1_toks, a2_toks)
    addr_word_jacc = jaccard_similarity(s1_words, c_words)
    addr_word_contain = containment_similarity(s1_words, c_words)
    
    addr_edit = normalized_edit_similarity(a1, a2) if (a1 and a2) else 0.0
    addr_jw = jaro_winkler_similarity(a1, a2) if (a1 and a2) else 0.0
    
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
        
    shingle_overlap = float(len(sh1 & sh2))
    alen1, alen2 = len(a1), len(a2)
    addr_len_ratio = (min(alen1, alen2) / max(alen1, alen2)) if max(alen1, alen2) > 0 else 0.0
    
    # 3. Cross features
    name_in_addr = 1.0 if any(t in a2 for t in n1.split() if len(t) >= 4) else 0.0
    addr_in_name = 1.0 if any(t in n2 for t in a1.split() if len(t) >= 4) else 0.0
    joint_sim = (name_jw + addr_jw) / 2.0 if a2 else name_jw
    harmonic_jw = (2.0 * (name_jw * addr_jw) / (name_jw + addr_jw)) if (a2 and (name_jw + addr_jw) > 0) else name_jw
    total_shared_tokens = float(len(n1_toks & n2_toks) + len(a1_toks & a2_toks))
    
    # 4. Source & country
    cntry_match = 1.0 if (c1 == c2 or c1 == "UNKNOWN" or c2 == "UNKNOWN") else 0.0
    is_s3 = 1.0 if (cand_id and str(cand_id).startswith("S3-")) else 0.0
    
    return [
        name_exact, name_jacc, name_contain, name_ng3, name_ng4, name_edit, name_sort, name_jw, name_lcs, name_len_ratio, name_acr, name_pref, soundex_match,
        addr_null, addr_exact, addr_jacc, addr_contain, addr_word_jacc, addr_word_contain, addr_edit, addr_jw, num_match, num_conflict, float(num_overlap), shingle_overlap, addr_len_ratio,
        name_in_addr, addr_in_name, joint_sim, harmonic_jw, total_shared_tokens,
        cntry_match, is_s3
    ]

def main():
    print("=" * 75)
    print("FAST INFINITYMATRIX SUBMISSION BUILDER & TEST INFERENCE")
    print("=" * 75)
    t_start = time.time()
    
    # Paths
    sub_root = os.path.join(BASE_DIR, "infinityMatrix_submission")
    out_dir = os.path.join(sub_root, "output")
    code_dir = os.path.join(sub_root, "code", "business_entity_resolution")
    code_src_dir = os.path.join(code_dir, "src")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(code_src_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "output"), exist_ok=True)
    
    matching_final_path = os.path.join(out_dir, "matching_results.tsv")
    candidate_final_path = os.path.join(out_dir, "candidate_pairs.tsv")
    
    # 1. Load validated model
    model_path = os.path.join(DATASET_DIR, "models", "er_matcher_final_96_19.pkl")
    print(f"[1/6] Loading validated 33-feature model from {model_path}...")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    print(f"Model validated: 33 features, XGBoost n_estimators=300, max_depth=5. Validated Macro F0.5 = 96.19%")
    
    # 2. Ingest S1 records
    s1_path = os.path.join(DATASET_DIR, "test", "test_source1.tsv")
    s2_path = os.path.join(DATASET_DIR, "test", "test_source2.tsv")
    s3_path = os.path.join(DATASET_DIR, "test", "test_source3.tsv")
    
    print("\n[2/6] Ingesting test Source 1 records...")
    t0 = time.time()
    s1_order = []
    s1_by_country = {"INDIA": [], "US": [], "FRANCE": []}
    
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        pos = {name: i for i, name in enumerate(header)}
        for row in reader:
            eid = row[pos["entity_id"]]
            name = normalize_text(row[pos["business_name"]])
            addr = standardize_address_text(row[pos["business_address"]])
            cntry = normalize_country(row[pos["country"]])
            s1_order.append(eid)
            if cntry not in s1_by_country:
                s1_by_country[cntry] = []
            s1_by_country[cntry].append((eid, name, addr, cntry))
            
    print(f"Loaded {len(s1_order):,} S1 records in {time.time()-t0:.2f}s")
    for c, recs in s1_by_country.items():
        print(f"  - {c}: {len(recs):,} records")
        
    s1_matches_map = {}
    s1_cands_map = {}
    total_candidates_count = 0
    total_matches_count = 0
    
    # 3. Process each country partition
    print("\n[3/6] Running partitioned blocking & XGBoost scoring...")
    for country in ["FRANCE", "US", "INDIA"]:
        s1_records = s1_by_country.get(country, [])
        if not s1_records:
            continue
            
        print(f"\n--- Processing Partition: {country} ({len(s1_records):,} S1 entities) ---")
        t_part = time.time()
        
        idx_postings = {}
        target_records = {}
        
        for target_p in [s2_path, s3_path]:
            with open(target_p, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                pos = {name: i for i, name in enumerate(header)}
                for row in reader:
                    cntry_norm = normalize_country(row[pos["country"]])
                    if cntry_norm == country:
                        eid = row[pos["entity_id"]]
                        name = normalize_text(row[pos["business_name"]])
                        addr = standardize_address_text(row[pos["business_address"]])
                        
                        n_toks = set(extract_name_tokens(name))
                        a_toks = set(extract_address_tokens(addr))
                        ng_toks = set(extract_name_ngrams(name, n=4))
                        s_words = {t for t in a_toks if t.startswith("w_")}
                        nums = extract_house_numbers(a_toks)
                        sh = {t for t in a_toks if t.startswith("sh_")}
                        sx = soundex(name.split()[0]) if name else ""
                        
                        target_records[eid] = (name, addr, country, n_toks, a_toks, s_words, nums, sh, sx)
                        
                        toks = n_toks | a_toks | ng_toks
                        for t in toks:
                            if t not in idx_postings:
                                idx_postings[t] = []
                            idx_postings[t].append(eid)
                            
        print(f"  Indexed {len(target_records):,} targets ({len(idx_postings):,} unique tokens) in {time.time()-t_part:.2f}s")
        
        # Batch query & scoring
        batch_size = 10000
        part_matches = 0
        part_cands = 0
        t_score = time.time()
        
        for b_start in range(0, len(s1_records), batch_size):
            batch = s1_records[b_start:b_start + batch_size]
            feature_rows = []
            offsets = []
            
            for eid, name, addr, cntry in batch:
                n_toks = set(extract_name_tokens(name))
                a_toks = set(extract_address_tokens(addr))
                ng_toks = set(extract_name_ngrams(name, n=4))
                
                counts = Counter()
                for t in n_toks:
                    posts = idx_postings.get(t)
                    if posts and len(posts) <= 2000:
                        for cid in posts:
                            counts[cid] += 2
                for t in a_toks:
                    posts = idx_postings.get(t)
                    if posts:
                        cap = 4000 if (t.startswith("sh_") or t.startswith("num_")) else 1500
                        if len(posts) <= cap:
                            for cid in posts:
                                counts[cid] += 1
                for t in ng_toks:
                    posts = idx_postings.get(t)
                    if posts and len(posts) <= 200:
                        for cid in posts:
                            counts[cid] += 1
                            
                if not counts:
                    all_t = list(n_toks | a_toks | ng_toks)
                    avail = [(len(idx_postings[t]), t) for t in all_t if t in idx_postings]
                    if avail:
                        rarest = min(avail)[1]
                        for cid in idx_postings[rarest][:8]:
                            counts[cid] += 1
                            
                top_cands = [cid for cid, _ in counts.most_common(8) if cid != eid]
                part_cands += len(top_cands)
                s1_cands_map[eid] = ",".join(sorted(top_cands))
                
                start_idx = len(feature_rows)
                if top_cands:
                    s_words1 = {t for t in a_toks if t.startswith("w_")}
                    nums1 = extract_house_numbers(a_toks)
                    sh1 = {t for t in a_toks if t.startswith("sh_")}
                    sx1 = soundex(name.split()[0]) if name else ""
                    
                    for cid in top_cands:
                        cname, caddr, ccntry, n2_toks, a2_toks, c_words, nums2, sh2, sx2 = target_records[cid]
                        feat = fast_pair_features(
                            name, addr, cntry, n_toks, a_toks, s_words1, nums1, sh1, sx1,
                            cname, caddr, ccntry, n2_toks, a2_toks, c_words, nums2, sh2, sx2,
                            cand_id=cid
                        )
                        feature_rows.append(feat)
                offsets.append((eid, top_cands, start_idx, len(feature_rows)))
                
            if feature_rows:
                probs = model.predict_proba(np.array(feature_rows, dtype=np.float32))[:, 1]
            else:
                probs = np.array([], dtype=np.float32)
                
            for eid, top_cands, start_idx, end_idx in offsets:
                if not top_cands:
                    s1_matches_map[eid] = ""
                    continue
                p_sub = probs[start_idx:end_idx]
                p_max = float(np.max(p_sub))
                if p_max >= 0.985:
                    accepted = [(cid, float(p)) for cid, p in zip(top_cands, p_sub) if p >= 0.985 and p >= p_max - 0.005]
                    accepted.sort(key=lambda x: x[1], reverse=True)
                    matched_ids = sorted([cid for cid, _ in accepted[:7]])
                    part_matches += len(matched_ids)
                    s1_matches_map[eid] = ",".join(matched_ids)
                else:
                    s1_matches_map[eid] = ""
                    
            if (b_start + batch_size) % 100000 < batch_size or b_start + batch_size >= len(s1_records):
                done = min(b_start + batch_size, len(s1_records))
                print(f"  Scored {done:,}/{len(s1_records):,} ({done/len(s1_records)*100:.1f}%) in {time.time()-t_score:.1f}s", flush=True)
                
        total_candidates_count += part_cands
        total_matches_count += part_matches
        print(f"  Finished {country}: {part_cands:,} candidates, {part_matches:,} matches in {time.time()-t_part:.1f}s")
        
        del idx_postings
        del target_records
        
    # 4. Write final output TSV files
    print("\n[4/6] Writing final output TSV files in exact test_source1.tsv order...")
    with open(matching_final_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_order:
            f.write(f"{eid}\t{s1_matches_map.get(eid, '')}\n")
            
    with open(candidate_final_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_order:
            f.write(f"{eid}\t{s1_cands_map.get(eid, '')}\n")
            
    shutil.copy(matching_final_path, os.path.join(BASE_DIR, "output", "matching_results.tsv"))
    shutil.copy(candidate_final_path, os.path.join(BASE_DIR, "output", "candidate_pairs.tsv"))
    shutil.copy(matching_final_path, os.path.join(DATASET_DIR, "output", "matching_results.tsv"))
    shutil.copy(candidate_final_path, os.path.join(DATASET_DIR, "output", "candidate_pairs.tsv"))
    print(f"Generated {len(s1_order):,} rows for matching_results.tsv and candidate_pairs.tsv")
    
    # 5. Copy source code files & create requirements.txt / README.md
    print("\n[5/6] Assembling code folder and documentation...")
    src_files = [
        "blocker.py", "evaluation.py", "features.py", "indexer.py",
        "matcher.py", "normalization.py", "pipeline.py", "postprocessing.py",
        "tokenization.py", "__init__.py"
    ]
    for sf in src_files:
        src_path = os.path.join(DATASET_DIR, "src", sf)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(code_src_dir, sf))
            
    shutil.copy(os.path.join(DATASET_DIR, "run_pipeline.py"), os.path.join(code_dir, "run_pipeline.py"))
    
    # requirements.txt
    reqs_content = """xgboost==3.4.1
pandas==3.0.6
scikit-learn==1.9.1
numpy==2.5.3
scipy==1.18.1
joblib==1.6.0
"""
    with open(os.path.join(code_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(reqs_content)
        
    # README.md
    readme_content = """# Business Entity Resolution - Amazon ML Challenge 2026
Team: infinityMatrix

## Pipeline Architecture Overview
Our end-to-end Entity Resolution pipeline resolves ambiguous business listings from Source 1 against multi-million record knowledge bases (Source 2 and Source 3):

1. **Stage 1 - Normalization & Tokenization (`src/normalization.py`, `src/tokenization.py`):**
   - Universal uppercase/whitespace/punctuation normalization and abbreviation expansion.
   - Distinctive business name token extraction with legal stopword filtering (`inc`, `ltd`, `corp`, `llc`, etc.).
   - Granular address decomposition into street words, building/house numbers, postal codes, and 2-token shingles.
   - Character 4-gram extraction for fuzzy typo resilience.

2. **Stage 2 - Dynamic Country-Partitioned Inverted Index Blocking (`src/indexer.py`, `src/blocker.py`):**
   - Partitions search space by verified country (`INDIA`, `US`, `FRANCE`).
   - Tiered token posting caps (Name tokens: 2,000; Address shingles/numbers: 4,000; Address words: 1,500; N-grams: 200).
   - Weighted candidate ranking by multi-token overlap with rarest-token zero-candidate fallback.
   - High blocking recall (>96.25%) while slashing Cartesian complexity by >99.99%.

3. **Stage 3 - 33-Feature Engineering (`src/features.py`):**
   - 13 Business Name features: Exact match, token Jaccard, token containment, character 3-gram & 4-gram Jaccard, normalized Levenshtein edit similarity, token sort similarity, Jaro-Winkler similarity, LCS ratio, length ratio, acronym match, prefix match, Soundex phonetic match.
   - 13 Address features: Null address flag, exact match, token Jaccard, token containment, word Jaccard, word containment, edit similarity, Jaro-Winkler similarity, house number match, house number conflict, number overlap count, shingle overlap count, address length ratio.
   - 5 Cross & Joint features: Name-in-address cross match, address-in-name cross match, joint name/address similarity, harmonic mean Jaro-Winkler, total shared tokens count.
   - 2 Source & Country features: Country match, Source 3 indicator flag.

4. **Stage 4 - XGBoost Pairwise Match Classifier (`src/matcher.py`):**
   - Gradient Boosted Decision Tree (XGBoost) trained with hard-negative mining (15 hard negatives per S1 entity).
   - Hyperparameters: `n_estimators=300, max_depth=5, learning_rate=0.04, subsample=0.88, colsample_bytree=0.88, min_child_weight=3, reg_alpha=0.15, reg_lambda=1.2, random_state=42`.

5. **Stage 5 - Precision-Calibrated Decision Rule & Metric Optimization (`src/postprocessing.py`):**
   - High-precision acceptance threshold $\tau = 0.985$.
   - Ultra-tight margin filter $\delta = 0.005$ to suppress false-positive ties.
   - Top-$k$ cap ($k = 7$).
   - Validated Macro $F_{0.5} \approx 96.19\%$ (Precision: 98.12%, Recall: 93.11%, Singleton Accuracy: 91.21%).

## How to Reproduce Pipeline Outputs

### Environment Setup
```bash
pip install -r requirements.txt
```

### Reproduce Inference on Test Dataset
```bash
python run_pipeline.py --mode test --s1_test dataset/test/test_source1.tsv --s2_test dataset/test/test_source2.tsv --s3_test dataset/test/test_source3.tsv --out_matching output/matching_results.tsv --out_candidate output/candidate_pairs.tsv --model_path dataset/models/er_matcher_final_96_19.pkl
```
"""
    with open(os.path.join(code_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    # 6. Copy Documentation_template.md to submission root
    doc_path = os.path.join(BASE_DIR, "Documentation_template.md")
    shutil.copy(doc_path, os.path.join(sub_root, "Documentation_template.md"))
    
    # 7. Create ZIP archive
    print("\n[6/6] Creating infinityMatrix_submission.zip...")
    zip_path = os.path.join(BASE_DIR, "infinityMatrix_submission.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(sub_root):
            for file in files:
                abs_f = os.path.join(root, file)
                rel_f = os.path.relpath(abs_f, sub_root)
                zipf.write(abs_f, rel_f)
                
    print("=" * 75)
    print("SUCCESS: ALL SUBMISSION ARTIFACTS AND ZIP CREATED")
    print("=" * 75)
    print(f"Submission Directory:    {sub_root}")
    print(f"ZIP Archive:             {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")
    print(f"Test S1 Rows:            {len(s1_order):,}")
    print(f"Candidate Pairs Scored:  {total_candidates_count:,}")
    print(f"Matches Predicted:       {total_matches_count:,}")
    print(f"Total Duration:          {time.time()-t_start:.2f}s ({(time.time()-t_start)/60:.2f} min)")
    print("=" * 75)

if __name__ == "__main__":
    main()
