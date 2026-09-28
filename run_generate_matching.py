import os
import sys
import gc
import time
import pickle
import csv
import numpy as np

# Adjust paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.normalization import normalize_country, normalize_text, standardize_address_text
from src.indexer import CountryInvertedIndex
from src.features import FEATURE_NAMES, extract_pair_features

def main():
    print("=" * 70)
    print("GENERATING TEST MATCHING_RESULTS.TSV")
    print("=" * 70)
    t0 = time.time()
    
    # 1. Load validated 33-feature model
    model_path = os.path.join(BASE_DIR, "models", "er_matcher_final_96_19.pkl")
    print(f"[1/4] Loading validated model from {model_path}...")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    
    feature_count = model.get_booster().num_features() if hasattr(model, "get_booster") else len(FEATURE_NAMES)
    print(f"Model loaded successfully. Features: {feature_count}, Validated F0.5: {artifact.get('validation_metrics', {}).get('f05', 0)*100:.2f}%")
    assert feature_count == 33, f"Expected 33 features, got {feature_count}"
    
    # 2. Setup paths
    s1_path = os.path.join(BASE_DIR, "test", "test_source1.tsv")
    s2_path = os.path.join(BASE_DIR, "test", "test_source2.tsv")
    s3_path = os.path.join(BASE_DIR, "test", "test_source3.tsv")
    
    out_dir = os.path.join(ROOT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_matching_path = os.path.join(out_dir, "matching_results.tsv")
    
    # Also ensure dataset/output exists
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    
    # 3. Read S1 records grouped by country
    print("\n[2/4] Ingesting test Source 1 records...")
    t_s1 = time.time()
    s1_order = []
    s1_by_country = {"INDIA": [], "US": [], "FRANCE": []}
    
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        pos = {name: i for i, name in enumerate(header)}
        for row in reader:
            eid = row[pos["entity_id"]]
            name = row[pos["business_name"]]
            addr = row[pos["business_address"]]
            cntry = normalize_country(row[pos["country"]])
            s1_order.append(eid)
            if cntry not in s1_by_country:
                s1_by_country[cntry] = []
            s1_by_country[cntry].append((eid, name, addr, cntry))
            
    print(f"Loaded {len(s1_order):,} S1 entities in {time.time()-t_s1:.2f}s")
    for c, records in s1_by_country.items():
        print(f"  - {c}: {len(records):,} entities")
        
    # Dictionary to hold final matches for every S1 entity
    s1_predictions = {}
    total_matches_count = 0
    
    # 4. Process each country partition
    print("\n[3/4] Processing country partitions (blocking + feature extraction + inference)...")
    partitions = ["FRANCE", "US", "INDIA"]
    
    for country in partitions:
        s1_records = s1_by_country.get(country, [])
        if not s1_records:
            continue
            
        print(f"\n--- Partition: {country} ({len(s1_records):,} S1 records) ---")
        t_part = time.time()
        
        # Build inverted index for this country
        index = CountryInvertedIndex(
            country=country,
            name_token_cap=2000,
            addr_word_cap=1500,
            shingle_cap=4000,
            ngram_cap=200,
            use_ngrams=True
        )
        
        # Load targets for this country
        target_records = {}
        for target_path in [s2_path, s3_path]:
            with open(target_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                pos = {name: i for i, name in enumerate(header)}
                for row in reader:
                    cntry_raw = row[pos["country"]]
                    if normalize_country(cntry_raw) == country:
                        eid = row[pos["entity_id"]]
                        name = row[pos["business_name"]]
                        addr = row[pos["business_address"]]
                        index.add_record(eid, name, addr)
                        target_records[eid] = (name, addr, country)
                        
        print(f"  Indexed {len(target_records):,} target records in {time.time()-t_part:.2f}s")
        
        # Batch scoring
        batch_size = 5000
        part_matches = 0
        t_score = time.time()
        
        for b_start in range(0, len(s1_records), batch_size):
            batch = s1_records[b_start:b_start + batch_size]
            feature_rows = []
            offsets = []
            
            for eid, name, addr, cntry in batch:
                cands = list(index.query_candidates(name, addr))
                valid_cands = [c for c in cands if c != eid and c in target_records]
                start_idx = len(feature_rows)
                for cid in valid_cands:
                    cname, caddr, ccntry = target_records[cid]
                    feature_rows.append(extract_pair_features(name, addr, cntry, cname, caddr, ccntry, cand_id=cid))
                offsets.append((eid, valid_cands, start_idx, len(feature_rows)))
                
            if feature_rows:
                X_mat = np.array(feature_rows, dtype=np.float32)
                probs = model.predict_proba(X_mat)[:, 1]
            else:
                probs = np.array([], dtype=np.float32)
                
            for eid, valid_cands, start_idx, end_idx in offsets:
                if not valid_cands:
                    s1_predictions[eid] = ""
                    continue
                p_sub = probs[start_idx:end_idx]
                p_max = float(np.max(p_sub))
                if p_max < 0.985:
                    s1_predictions[eid] = ""
                else:
                    # Margin filter (delta = 0.005) and top-k (k = 7)
                    accepted = [(cid, float(p)) for cid, p in zip(valid_cands, p_sub) if p >= 0.985 and p >= p_max - 0.005]
                    accepted.sort(key=lambda x: x[1], reverse=True)
                    matched_ids = sorted([cid for cid, _ in accepted[:7]])
                    part_matches += len(matched_ids)
                    s1_predictions[eid] = ",".join(matched_ids)
                    
            if (b_start + batch_size) % 50000 < batch_size or b_start + batch_size >= len(s1_records):
                done = min(b_start + batch_size, len(s1_records))
                print(f"  Scored {done:,}/{len(s1_records):,} ({done/len(s1_records)*100:.1f}%) in {time.time()-t_score:.1f}s", flush=True)
                
        total_matches_count += part_matches
        print(f"  Completed {country}: {part_matches:,} matches predicted in {time.time()-t_part:.1f}s")
        
        # Free partition memory
        index.clear()
        del target_records
        del index
        gc.collect()
        
    # 5. Write matching_results.tsv in exact S1 order
    print(f"\n[4/4] Writing matching results to {out_matching_path}...")
    t_out = time.time()
    
    with open(out_matching_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_order:
            matched_str = s1_predictions.get(eid, "")
            f.write(f"{eid}\t{matched_str}\n")
            
    # Also write copy to dataset/output/matching_results.tsv
    ds_matching_path = os.path.join(BASE_DIR, "output", "matching_results.tsv")
    with open(ds_matching_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_order:
            matched_str = s1_predictions.get(eid, "")
            f.write(f"{eid}\t{matched_str}\n")
            
    total_duration = time.time() - t0
    print("=" * 70)
    print("SUCCESS: TEST INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Total Test S1 Records:    {len(s1_order):,}")
    print(f"Total Matches Predicted:  {total_matches_count:,}")
    print(f"Output File:              {out_matching_path}")
    print(f"File Size:                {os.path.getsize(out_matching_path):,} bytes")
    print(f"Total Time Elapsed:       {total_duration:.2f}s ({total_duration/60:.2f} min)")
    print("=" * 70)

if __name__ == "__main__":
    main()
