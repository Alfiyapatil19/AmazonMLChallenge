"""
End-to-End Pipeline Runner for Amazon Business Entity Resolution Challenge.

Supports:
  1. Training & Validation Mode:
     python run_pipeline.py --mode train
  2. Test Inference & Final Output Generation:
     python run_pipeline.py --mode test
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple

from src.normalization import normalize_country
from src.indexer import CountryInvertedIndex
from src.features import extract_pair_features
from src.matcher import EntityResolutionMatcher
from src.postprocessing import (
    compute_macro_f05,
    optimize_threshold,
    write_matching_results_tsv
)
from src.pipeline import EndToEndEntityResolutionPipeline

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def run_train_mode(args):
    """Executes Stage 1-5 training, validation, and threshold optimization."""
    pipeline = EndToEndEntityResolutionPipeline(
        name_token_cap=args.name_cap,
        addr_word_cap=args.addr_cap,
        shingle_cap=args.shingle_cap,
        ngram_cap=args.ngram_cap,
        chunk_size=args.chunk_size,
        model_path=args.model_path
    )
    
    results = pipeline.train_and_evaluate(
        s1_path=args.s1_train,
        s2_path=args.s2_train,
        s3_path=args.s3_train,
        gt_path=args.gt_train,
        n_sample_s1=args.n_sample_s1,
        val_split=args.val_split
    )
    return results


def run_test_inference(args):
    """Executes full inference on the Test Dataset and generates output TSVs."""
    print("=" * 70)
    print("STAGE 1 - 6: TEST INFERENCE & OUTPUT GENERATION")
    print("=" * 70)
    t_start = time.time()
    
    os.makedirs(os.path.dirname(args.out_matching), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_candidate), exist_ok=True)
    
    # 1. Check trained model
    matcher = EntityResolutionMatcher()
    if os.path.exists(args.model_path):
        print(f"Loading trained matcher from {args.model_path}...")
        matcher.load_model(args.model_path)
    else:
        print(f"Model not found at {args.model_path}. Training a quick model on training data...")
        pipeline = EndToEndEntityResolutionPipeline(model_path=args.model_path)
        pipeline.train_and_evaluate(
            s1_path=args.s1_train,
            s2_path=args.s2_train,
            s3_path=args.s3_train,
            gt_path=args.gt_train,
            n_sample_s1=10000
        )
        matcher.load_model(args.model_path)
        
    threshold = args.threshold
    print(f"Using Decision Threshold (tau): {threshold:.2f}")
    
    # 2. Discover test country partitions dynamically (e.g. US, India, France)
    print("\n[1/3] Discovering test country partitions dynamically...")
    countries: Set[str] = set()
    for path in [args.s1_test, args.s2_test, args.s3_test]:
        for chunk in pd.read_csv(path, sep="\t", usecols=["country"], chunksize=args.chunk_size):
            for val in chunk["country"]:
                countries.add(normalize_country(val))
    sorted_countries = sorted(countries)
    print(f"Discovered {len(sorted_countries)} test partition(s): {sorted_countries}")
    
    # 3. Open output file handles
    matching_f = open(args.out_matching, "w", encoding="utf-8")
    matching_f.write("source1_entity_id\tmatched_entity_ids\n")
    
    candidate_f = open(args.out_candidate, "w", encoding="utf-8")
    candidate_f.write("source1_entity_id\tcandidate_entity_ids\n")
    
    total_test_s1 = 0
    total_matches_predicted = 0
    total_cands_generated = 0
    
    print("\n[2/3] Processing test partitions & scoring candidate pairs...")
    
    try:
        for country in sorted_countries:
            print(f"\n--- Processing Partition: {country} ---")
            t_part = time.time()
            
            # Build inverted index for this country
            index = CountryInvertedIndex(
                country=country,
                name_token_cap=args.name_cap,
                addr_word_cap=args.addr_cap,
                shingle_cap=args.shingle_cap,
                ngram_cap=args.ngram_cap,
                use_ngrams=True
            )
            
            # Store target records in memory for feature extraction
            target_records = {}
            for path in [args.s2_test, args.s3_test]:
                for chunk in pd.read_csv(path, sep="\t", chunksize=args.chunk_size):
                    for eid, name, addr, cntry in zip(
                        chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]
                    ):
                        c_norm = normalize_country(cntry)
                        if country == "UNKNOWN" or c_norm == country:
                            index.add_record(eid, name, addr)
                            target_records[eid] = (name, addr, cntry)
                            
            print(f"  Indexed {index.total_indexed:,} test target records for {country}")
            
            # Query S1 records and predict matches
            part_s1_count = 0
            for chunk in pd.read_csv(args.s1_test, sep="\t", chunksize=args.chunk_size):
                for eid, name, addr, cntry in zip(
                    chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]
                ):
                    c_norm = normalize_country(cntry)
                    if country == "UNKNOWN" or c_norm == country:
                        part_s1_count += 1
                        total_test_s1 += 1
                        
                        cands = list(index.query_candidates(name, addr))
                        cands = [c for c in cands if c != eid]
                        total_cands_generated += len(cands)
                        
                        # Stream to candidate_pairs.tsv
                        cands_str = ",".join(sorted(cands))
                        candidate_f.write(f"{eid}\t{cands_str}\n")
                        
                        # Featurize & Predict
                        if not cands:
                            matching_f.write(f"{eid}\t\n")
                            continue
                            
                        X_pair = []
                        valid_cands = []
                        for c_id in cands:
                            if c_id in target_records:
                                c_name, c_addr, c_cntry = target_records[c_id]
                                feat = extract_pair_features(name, addr, cntry, c_name, c_addr, c_cntry)
                                X_pair.append(feat)
                                valid_cands.append(c_id)
                                
                        if X_pair:
                            probs = matcher.predict_proba(np.array(X_pair, dtype=np.float32))
                            matched = [cid for cid, p in zip(valid_cands, probs) if p >= threshold]
                            total_matches_predicted += len(matched)
                            matched_str = ",".join(sorted(matched))
                            matching_f.write(f"{eid}\t{matched_str}\n")
                        else:
                            matching_f.write(f"{eid}\t\n")
                            
            print(f"  Completed {part_s1_count:,} test S1 records for {country} in {time.time()-t_part:.2f}s")
            
            # Release partition memory
            index.clear()
            del target_records
            
    finally:
        matching_f.close()
        candidate_f.close()
        
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("FINAL TEST SUBMISSION FILES GENERATED")
    print("=" * 70)
    print(f"Total Test S1 Records Processed:     {total_test_s1:,}")
    print(f"Total Candidate Pairs Generated:     {total_cands_generated:,} (Avg: {total_cands_generated/total_test_s1:.1f}/S1)")
    print(f"Total Matches Predicted:             {total_matches_predicted:,} (Avg: {total_matches_predicted/total_test_s1:.2f}/S1)")
    print(f"Matching Results Saved to:           {args.out_matching}")
    print(f"Candidate Pairs Saved to:            {args.out_candidate}")
    print(f"Total Inference Time:                {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Business Entity Resolution End-to-End Pipeline")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Execution mode (train or test)")
    
    # Dataset paths
    parser.add_argument("--s1_train", default="train/train_source1.tsv")
    parser.add_argument("--s2_train", default="train/train_source2.tsv")
    parser.add_argument("--s3_train", default="train/train_source3.tsv")
    parser.add_argument("--gt_train", default="train/train_ground_truth.tsv")
    
    parser.add_argument("--s1_test", default="test/test_source1.tsv")
    parser.add_argument("--s2_test", default="test/test_source2.tsv")
    parser.add_argument("--s3_test", default="test/test_source3.tsv")
    
    # Output paths
    parser.add_argument("--out_matching", default="output/matching_results.tsv")
    parser.add_argument("--out_candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--model_path", default="models/er_matcher.pkl")
    
    # Blocking Hyperparameters
    parser.add_argument("--name_cap", type=int, default=2000)
    parser.add_argument("--addr_cap", type=int, default=1500)
    parser.add_argument("--shingle_cap", type=int, default=4000)
    parser.add_argument("--ngram_cap", type=int, default=200)
    parser.add_argument("--chunk_size", type=int, default=200000)
    
    # ML Hyperparameters
    parser.add_argument("--n_sample_s1", type=int, default=20000)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.70)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        run_train_mode(args)
    elif args.mode == "test":
        run_test_inference(args)
