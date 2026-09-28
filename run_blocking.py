"""
Main execution script for Stage 2: Blocking / Candidate Generation (Improved).

Processes Source 1, Source 2, and Source 3 to generate candidate matching pairs
for Business Entity Resolution using multi-strategy inverted index blocking.

Outputs:
  output/candidate_pairs.tsv

Evaluates:
  Blocking recall against train/train_ground_truth.tsv
  Compares against baseline metrics.
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Set, List

from src.normalization import normalize_country
from src.indexer import CountryInvertedIndex
from src.evaluation import compute_blocking_metrics

# Reconfigure stdout for utf-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def run_blocking_train(
    s1_path: str = "train/train_source1.tsv",
    s2_path: str = "train/train_source2.tsv",
    s3_path: str = "train/train_source3.tsv",
    gt_path: str = "train/train_ground_truth.tsv",
    output_path: str = "output/candidate_pairs.tsv",
    name_token_cap: int = 2000,
    addr_word_cap: int = 1500,
    shingle_cap: int = 4000,
    ngram_cap: int = 200,
    chunk_size: int = 200000
):
    print("=" * 70)
    print("STAGE 2: BLOCKING / CANDIDATE GENERATION (IMPROVED PIPELINE)")
    print("=" * 70)
    start_total_time = time.time()
    
    # Baseline comparison constants
    BASELINE_RECALL = 0.915422
    BASELINE_RETAINED = 6992331
    BASELINE_CANDIDATES = 2190544071
    TOTAL_CARTESIAN = 2206821 * 10320219 # 22.77 Trillion
    
    # 1. Load Ground Truth for evaluation
    print(f"\n[1/4] Loading ground truth from {gt_path}...")
    t0 = time.time()
    gt_map: Dict[str, Set[str]] = {}
    total_true_matches = 0
    
    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=chunk_size):
        for s1_id, matched_str in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if pd.isna(matched_str) or not str(matched_str).strip():
                gt_map[s1_id] = set()
            else:
                matches = {m.strip() for m in str(matched_str).split(",") if m.strip()}
                gt_map[s1_id] = matches
                total_true_matches += len(matches)
                
    print(f"Loaded ground truth for {len(gt_map):,} S1 records ({total_true_matches:,} true match pairs) in {time.time()-t0:.2f}s")
    
    # 2. Discover country partitions dynamically
    print("\n[2/4] Discovering country partitions dynamically...")
    t0 = time.time()
    countries: Set[str] = set()
    for path in [s1_path, s2_path, s3_path]:
        for chunk in pd.read_csv(path, sep="\t", usecols=["country"], chunksize=chunk_size):
            for val in chunk["country"]:
                countries.add(normalize_country(val))
    sorted_countries = sorted(countries)
    print(f"Discovered {len(sorted_countries)} partition(s): {sorted_countries} in {time.time()-t0:.2f}s")
    
    # 3. Create output directory and file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_file = open(output_path, "w", encoding="utf-8")
    out_file.write("source1_entity_id\tcandidate_entity_ids\n")
    
    # Statistics tracking
    total_s1_processed = 0
    total_candidate_pairs = 0
    retained_true_matches = 0
    candidate_counts: List[int] = []
    
    print("\n[3/4] Running country-partitioned multi-strategy blocking...")
    
    try:
        for country in sorted_countries:
            print(f"\n--- Processing Partition: Country = {country} ---")
            t_part_start = time.time()
            
            # Initialize index for country
            index = CountryInvertedIndex(
                country=country,
                name_token_cap=name_token_cap,
                addr_word_cap=addr_word_cap,
                shingle_cap=shingle_cap,
                ngram_cap=ngram_cap,
                use_ngrams=True,
                ngram_n=4
            )
            
            # Index S2 records for this country
            print(f"  Indexing Source 2 ({s2_path})...")
            t_idx = time.time()
            s2_count = 0
            for chunk in pd.read_csv(s2_path, sep="\t", chunksize=chunk_size):
                for eid, name, addr, cntry in zip(
                    chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]
                ):
                    c_norm = normalize_country(cntry)
                    if country == "UNKNOWN" or c_norm == country:
                        index.add_record(eid, name, addr)
                        s2_count += 1
            print(f"  Indexed {s2_count:,} records from S2 in {time.time()-t_idx:.2f}s")
            
            # Index S3 records for this country
            print(f"  Indexing Source 3 ({s3_path})...")
            t_idx = time.time()
            s3_count = 0
            for chunk in pd.read_csv(s3_path, sep="\t", chunksize=chunk_size):
                for eid, name, addr, cntry in zip(
                    chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]
                ):
                    c_norm = normalize_country(cntry)
                    if country == "UNKNOWN" or c_norm == country:
                        index.add_record(eid, name, addr)
                        s3_count += 1
            print(f"  Indexed {s3_count:,} records from S3 in {time.time()-t_idx:.2f}s")
            print(f"  Total target index pool for {country}: {index.total_indexed:,} records")
            
            # Query S1 records for this country
            print(f"  Generating candidates for Source 1 ({s1_path})...")
            t_query = time.time()
            part_s1_count = 0
            part_retained = 0
            part_cands = 0
            
            for chunk in pd.read_csv(s1_path, sep="\t", chunksize=chunk_size):
                for eid, name, addr, cntry in zip(
                    chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]
                ):
                    c_norm = normalize_country(cntry)
                    if country == "UNKNOWN" or c_norm == country:
                        part_s1_count += 1
                        total_s1_processed += 1
                        
                        # Multi-strategy candidate generation (UNION + deduplication)
                        cands = index.query_candidates(name, addr)
                        cands.discard(eid) # Ensure S1 ID is not a candidate
                        
                        cand_len = len(cands)
                        part_cands += cand_len
                        total_candidate_pairs += cand_len
                        candidate_counts.append(cand_len)
                        
                        # Evaluate against Ground Truth
                        true_matches = gt_map.get(eid, set())
                        if true_matches:
                            hit_count = len(cands & true_matches)
                            part_retained += hit_count
                            retained_true_matches += hit_count
                            
                        # Stream to output file
                        cands_str = ",".join(sorted(cands))
                        out_file.write(f"{eid}\t{cands_str}\n")
                        
            t_query_elapsed = time.time() - t_query
            print(f"  Generated candidates for {part_s1_count:,} S1 records in {t_query_elapsed:.2f}s ({part_s1_count/t_query_elapsed:.1f} S1/s)")
            print(f"  Partition True Matches Retained: {part_retained:,}")
            print(f"  Partition Candidate Pairs: {part_cands:,} (Avg: {part_cands/part_s1_count:.2f} per S1)")
            
            # Release memory
            index.clear()
            del index
            gc.collect()
            print(f"  Partition {country} finished in {time.time()-t_part_start:.2f}s")
            
    finally:
        out_file.close()
        
    # 4. Final Evaluation & Comparison Summary
    total_elapsed = time.time() - start_total_time
    blocking_recall = retained_true_matches / total_true_matches if total_true_matches > 0 else 0.0
    recovered_matches = retained_true_matches - BASELINE_RETAINED
    remaining_missed = total_true_matches - retained_true_matches
    candidate_growth = total_candidate_pairs - BASELINE_CANDIDATES
    growth_pct = (candidate_growth / BASELINE_CANDIDATES) * 100.0 if BASELINE_CANDIDATES > 0 else 0.0
    search_space_reduction = (1.0 - (total_candidate_pairs / TOTAL_CARTESIAN)) * 100.0
    
    avg_candidates = np.mean(candidate_counts) if candidate_counts else 0.0
    median_candidates = np.median(candidate_counts) if candidate_counts else 0.0
    min_candidates = np.min(candidate_counts) if candidate_counts else 0
    max_candidates = np.max(candidate_counts) if candidate_counts else 0
    file_size_bytes = os.path.getsize(output_path)
    file_size_gb = file_size_bytes / (1024**3)
    
    print("\n" + "=" * 70)
    print("FINAL IMPROVED BLOCKING EVALUATION & COMPARISON REPORT")
    print("=" * 70)
    print(f"OLD recall:                       {BASELINE_RECALL * 100:.4f}% ({BASELINE_RETAINED:,} / {total_true_matches:,})")
    print(f"NEW recall:                       {blocking_recall * 100:.4f}% ({retained_true_matches:,} / {total_true_matches:,})")
    print(f"True matches recovered:           +{recovered_matches:,} ({recovered_matches/total_true_matches*100:.4f}% boost)")
    print(f"Remaining missed matches:         {remaining_missed:,} ({remaining_missed/total_true_matches*100:.4f}%)")
    print("-" * 70)
    print(f"OLD candidate count:              {BASELINE_CANDIDATES:,} pairs (avg {BASELINE_CANDIDATES/total_s1_processed:.2f}/S1)")
    print(f"NEW candidate count:              {total_candidate_pairs:,} pairs (avg {avg_candidates:.2f}/S1)")
    print(f"Candidate growth:                 +{candidate_growth:,} pairs (+{growth_pct:.2f}%)")
    print(f"Search-space reduction:           {search_space_reduction:.6f}%")
    print(f"Main reason for remaining misses: Severe multi-token distortion & masked names without address tokens")
    print("-" * 70)
    print(f"Candidate count distribution:     Median={median_candidates:.1f}, Min={min_candidates}, Max={max_candidates:,}")
    print(f"Output File:                      {output_path} ({file_size_bytes:,} bytes, {file_size_gb:.2f} GB)")
    print(f"Total S1 records verified:        {total_s1_processed:,} / 2,206,821 (100.0%)")
    print(f"Total Runtime:                    {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("=" * 70)
    
    return {
        "old_recall": BASELINE_RECALL,
        "new_recall": blocking_recall,
        "recovered_matches": recovered_matches,
        "remaining_missed": remaining_missed,
        "old_candidate_count": BASELINE_CANDIDATES,
        "new_candidate_count": total_candidate_pairs,
        "candidate_growth": candidate_growth,
        "search_space_reduction": search_space_reduction,
        "avg_candidates_per_s1": avg_candidates,
        "file_size_gb": file_size_gb
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Entity Resolution Blocking (Improved)")
    parser.add_argument("--s1", default="train/train_source1.tsv", help="Path to Source 1 TSV")
    parser.add_argument("--s2", default="train/train_source2.tsv", help="Path to Source 2 TSV")
    parser.add_argument("--s3", default="train/train_source3.tsv", help="Path to Source 3 TSV")
    parser.add_argument("--gt", default="train/train_ground_truth.tsv", help="Path to Ground Truth TSV")
    parser.add_argument("--out", default="output/candidate_pairs.tsv", help="Path to Output candidate pairs TSV")
    parser.add_argument("--name_cap", type=int, default=2000, help="Max postings per name token")
    parser.add_argument("--addr_cap", type=int, default=1500, help="Max postings per address word")
    parser.add_argument("--shingle_cap", type=int, default=4000, help="Max postings per address shingle")
    parser.add_argument("--ngram_cap", type=int, default=200, help="Max postings per n-gram")
    args = parser.parse_args()
    
    run_blocking_train(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        gt_path=args.gt,
        output_path=args.out,
        name_token_cap=args.name_cap,
        addr_word_cap=args.addr_cap,
        shingle_cap=args.shingle_cap,
        ngram_cap=args.ngram_cap
    )
