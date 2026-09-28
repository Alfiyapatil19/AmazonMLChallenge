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
from src.features import FEATURE_NAMES, extract_pair_features

def main():
    print("=" * 75)
    print("INFINITYMATRIX SUBMISSION BUILDER & TEST INFERENCE ENGINE")
    print("=" * 75)
    t_start = time.time()
    
    # 1. Output paths
    sub_root = os.path.join(BASE_DIR, "infinityMatrix_submission")
    out_dir = os.path.join(sub_root, "output")
    code_src_dir = os.path.join(sub_root, "code", "business_entity_resolution", "src")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(code_src_dir, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    
    matching_final_path = os.path.join(out_dir, "matching_results.tsv")
    candidate_final_path = os.path.join(out_dir, "candidate_pairs.tsv")
    
    # 2. Load model
    model_path = os.path.join(DATASET_DIR, "models", "er_matcher_final_96_19.pkl")
    print(f"[1/5] Loading 33-feature model from {model_path}...")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    print(f"Model verified. Features: {len(FEATURE_NAMES)}, Validated Macro F0.5: {artifact['validation_metrics']['f05']*100:.2f}%")
    
    # 3. Ingest S1 records
    s1_path = os.path.join(DATASET_DIR, "test", "test_source1.tsv")
    s2_path = os.path.join(DATASET_DIR, "test", "test_source2.tsv")
    s3_path = os.path.join(DATASET_DIR, "test", "test_source3.tsv")
    
    print("\n[2/5] Ingesting test Source 1 records...")
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
    
    # 4. Process each country partition
    print("\n[3/5] Processing partitioned blocking & XGBoost matching...")
    for country in ["FRANCE", "US", "INDIA"]:
        s1_records = s1_by_country.get(country, [])
        if not s1_records:
            continue
            
        print(f"\n--- Partition: {country} ({len(s1_records):,} S1 queries) ---")
        t_part = time.time()
        
        # Build inverted index
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
                        target_records[eid] = (name, addr, country)
                        
                        toks = set(extract_name_tokens(name))
                        toks.update(extract_address_tokens(addr))
                        toks.update(extract_name_ngrams(name, n=4))
                        for t in toks:
                            if t not in idx_postings:
                                idx_postings[t] = []
                            idx_postings[t].append(eid)
                            
        print(f"  Indexed {len(target_records):,} targets ({len(idx_postings):,} unique tokens) in {time.time()-t_part:.2f}s")
        
        # Batch query & scoring
        batch_size = 5000
        part_matches = 0
        part_cands = 0
        t_score = time.time()
        
        for b_start in range(0, len(s1_records), batch_size):
            batch = s1_records[b_start:b_start + batch_size]
            feature_rows = []
            offsets = []
            
            for eid, name, addr, cntry in batch:
                n_toks = extract_name_tokens(name)
                a_toks = extract_address_tokens(addr)
                ng_toks = extract_name_ngrams(name, n=4)
                
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
                    all_t = n_toks + a_toks + ng_toks
                    avail = [(len(idx_postings[t]), t) for t in all_t if t in idx_postings]
                    if avail:
                        rarest = min(avail)[1]
                        for cid in idx_postings[rarest][:50]:
                            counts[cid] += 1
                            
                top_cands = [cid for cid, _ in counts.most_common(50) if cid != eid]
                part_cands += len(top_cands)
                s1_cands_map[eid] = ",".join(sorted(top_cands))
                
                start_idx = len(feature_rows)
                for cid in top_cands:
                    cname, caddr, ccntry = target_records[cid]
                    feature_rows.append(extract_pair_features(name, addr, cntry, cname, caddr, ccntry, cand_id=cid))
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
                print(f"  Scored {done:,}/{len(s1_records):,} S1 records in {time.time()-t_score:.1f}s", flush=True)
                
        total_candidates_count += part_cands
        total_matches_count += part_matches
        print(f"  Finished {country}: {part_cands:,} candidates, {part_matches:,} matches in {time.time()-t_part:.1f}s")
        
        del idx_postings
        del target_records
        
    # 5. Write final output TSV files
    print("\n[4/5] Writing final TSV files...")
    
    # Write matching_results.tsv
    with open(matching_final_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_order:
            f.write(f"{eid}\t{s1_matches_map.get(eid, '')}\n")
            
    # Write candidate_pairs.tsv
    with open(candidate_final_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_order:
            f.write(f"{eid}\t{s1_cands_map.get(eid, '')}\n")
            
    # Copy to root output/ and dataset/output/
    shutil.copy(matching_final_path, os.path.join(BASE_DIR, "output", "matching_results.tsv"))
    shutil.copy(candidate_final_path, os.path.join(BASE_DIR, "output", "candidate_pairs.tsv"))
    shutil.copy(matching_final_path, os.path.join(DATASET_DIR, "output", "matching_results.tsv"))
    shutil.copy(candidate_final_path, os.path.join(DATASET_DIR, "output", "candidate_pairs.tsv"))
    print(f"Written {len(s1_order):,} rows to matching_results.tsv and candidate_pairs.tsv")
    
    # 6. Copy source code files
    print("\n[5/5] Packaging source code and metadata...")
    src_files = [
        "blocker.py", "evaluation.py", "features.py", "indexer.py",
        "matcher.py", "normalization.py", "pipeline.py", "postprocessing.py",
        "tokenization.py", "__init__.py"
    ]
    for sf in src_files:
        src_path = os.path.join(DATASET_DIR, "src", sf)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(code_src_dir, sf))
            
    # Also copy run scripts
    shutil.copy(os.path.join(DATASET_DIR, "run_pipeline.py"), os.path.join(sub_root, "code", "business_entity_resolution", "run_pipeline.py"))
    
    print("=" * 75)
    print("INFERENCE & PACKAGING COMPLETED")
    print("=" * 75)
    print(f"Total Test S1 Entities:     {len(s1_order):,}")
    print(f"Total Candidate Pairs:      {total_candidates_count:,}")
    print(f"Total Matches Predicted:    {total_matches_count:,}")
    print(f"Total Duration:             {time.time()-t_start:.2f}s ({(time.time()-t_start)/60:.2f} min)")
    print("=" * 75)

if __name__ == "__main__":
    main()
