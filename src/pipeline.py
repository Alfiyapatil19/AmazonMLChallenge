"""
End-to-End Pipeline Orchestrator for Amazon Business Entity Resolution.

Coordinates:
- Stage 1: Ingestion & Normalization
- Stage 2: Country-Partitioned Blocking
- Stage 3: Pairwise Feature Engineering (28 dense features)
- Stage 4: Machine Learning Matching (XGBoost)
- Stage 5: Post-Processing & F_0.5 Threshold Optimization (Margin + Top-K)
- Stage 6: Official Submission Output Generation
"""

import os
import gc
import time
from typing import Dict, Set, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from .normalization import normalize_text, normalize_country
from .tokenization import extract_name_tokens, extract_address_tokens, extract_name_ngrams
from .indexer import CountryInvertedIndex
from .blocker import EntityResolutionBlocker
from .features import extract_pair_features, FEATURE_NAMES
from .matcher import EntityResolutionMatcher
from .postprocessing import compute_macro_f05, optimize_threshold, apply_decision_rules, write_matching_results_tsv
from .evaluation import compute_blocking_metrics, load_ground_truth_tsv


class EndToEndEntityResolutionPipeline:
    """
    Complete end-to-end Entity Resolution pipeline.
    """

    def __init__(
        self,
        name_token_cap: int = 2000,
        addr_word_cap: int = 1500,
        shingle_cap: int = 4000,
        ngram_cap: int = 200,
        chunk_size: int = 200000,
        model_path: str = "models/er_matcher.pkl"
    ):
        self.blocker = EntityResolutionBlocker(
            name_token_cap=name_token_cap,
            addr_word_cap=addr_word_cap,
            shingle_cap=shingle_cap,
            ngram_cap=ngram_cap,
            chunk_size=chunk_size
        )
        self.matcher = EntityResolutionMatcher()
        self.model_path = model_path
        self.optimal_threshold = 0.985
        self.optimal_delta = 0.005
        self.optimal_k = 7

    def train_and_evaluate(
        self,
        s1_path: str = "train/train_source1.tsv",
        s2_path: str = "train/train_source2.tsv",
        s3_path: str = "train/train_source3.tsv",
        gt_path: str = "train/train_ground_truth.tsv",
        n_sample_s1: int = 10000,
        val_split: float = 0.3
    ) -> Dict[str, float]:
        """
        Trains matching model on training data using mined hard negatives and evaluates macro F_0.5.
        """
        print("=" * 70)
        print("STAGE 1 - 5: TRAINING & VALIDATION PIPELINE")
        print("=" * 70)
        
        # 1. Load Ground Truth sample
        print(f"\n[1/5] Loading Ground Truth ({n_sample_s1:,} entities)...")
        gt_df = pd.read_csv(gt_path, sep="\t", nrows=n_sample_s1)
        s1_ids = list(gt_df["source1_entity_id"])
        s1_id_set = set(s1_ids)
        
        gt_map: Dict[str, Set[str]] = {}
        all_true_targets = set()
        for _, r in gt_df.iterrows():
            eid = r["source1_entity_id"]
            if pd.isna(r["matched_entity_ids"]) or not str(r["matched_entity_ids"]).strip():
                gt_map[eid] = set()
            else:
                matches = {m.strip() for m in str(r["matched_entity_ids"]).split(",") if m.strip()}
                gt_map[eid] = matches
                all_true_targets.update(matches)
                
        # 2. Load Source 1 records
        print("\n[2/5] Ingesting Source 1 records...")
        s1_dict: Dict[str, Dict[str, Any]] = {}
        for chunk in pd.read_csv(s1_path, sep="\t", chunksize=100000):
            sub = chunk[chunk["entity_id"].isin(s1_id_set)]
            for _, r in sub.iterrows():
                s1_dict[r["entity_id"]] = r.to_dict()
            if len(s1_dict) == len(s1_id_set):
                break
                
        # 3. Load Target S2 and S3 records
        print("\n[3/5] Ingesting Target S2 and S3 records...")
        target_dict: Dict[str, Dict[str, Any]] = {}
        for path in [s2_path, s3_path]:
            for chunk in pd.read_csv(path, sep="\t", chunksize=200000):
                sub = chunk[chunk["entity_id"].isin(all_true_targets)]
                for _, r in sub.iterrows():
                    target_dict[r["entity_id"]] = r.to_dict()
                if len(target_dict) >= len(all_true_targets):
                    break
                    
        # Load background sample for hard negative mining
        bg_s2 = pd.read_csv(s2_path, sep="\t", nrows=60000)
        bg_s3 = pd.read_csv(s3_path, sep="\t", nrows=60000)
        for chunk in [bg_s2, bg_s3]:
            for _, r in chunk.iterrows():
                if r["entity_id"] not in target_dict:
                    target_dict[r["entity_id"]] = r.to_dict()
                    
        print(f"Ingested {len(s1_dict):,} S1 entities, {len(target_dict):,} target records.")
        
        # 4. Generate Candidates & Extract Pairwise Features
        print("\n[4/5] Building Inverted Index & Extracting Features...")
        idx = CountryInvertedIndex(country="ALL")
        for eid, rec in target_dict.items():
            idx.add_record(eid, rec.get("business_name"), rec.get("business_address"))
            
        X_list: List[List[float]] = []
        y_list: List[int] = []
        
        n_train = int(len(s1_ids) * (1.0 - val_split))
        train_s1_ids = set(s1_ids[:n_train])
        val_s1_ids = set(s1_ids[n_train:])
        val_gt_map = {eid: gt_map[eid] for eid in val_s1_ids}
        
        for eid in train_s1_ids:
            s1_r = s1_dict[eid]
            true_m = gt_map.get(eid, set())
            cands = idx.query_candidates(s1_r.get("business_name"), s1_r.get("business_address"))
            cands.discard(eid)
            
            # Positives
            for m_id in true_m:
                if m_id in target_dict:
                    m_r = target_dict[m_id]
                    feat = extract_pair_features(
                        s1_r.get("business_name"), s1_r.get("business_address"), s1_r.get("country"),
                        m_r.get("business_name"), m_r.get("business_address"), m_r.get("country"),
                        cand_id=m_id
                    )
                    X_list.append(feat)
                    y_list.append(1)
                    
            # Hard Negatives (sample up to 10 per entity)
            negs = list(cands - true_m)
            if negs:
                for n_id in negs[:10]:
                    if n_id in target_dict:
                        n_r = target_dict[n_id]
                        feat = extract_pair_features(
                            s1_r.get("business_name"), s1_r.get("business_address"), s1_r.get("country"),
                            n_r.get("business_name"), n_r.get("business_address"), n_r.get("country"),
                            cand_id=n_id
                        )
                        X_list.append(feat)
                        y_list.append(0)
                        
        X_train = np.array(X_list, dtype=np.float32)
        y_train = np.array(y_list, dtype=np.int32)
        print(f"Training dataset: {len(X_train):,} pairs (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,})")
        
        # Train Matcher
        print("Training Gradient Boosted Matcher (XGBoost)...")
        self.matcher.fit(X_train, y_train)
        self.matcher.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # 5. Validation & Threshold Optimization
        print("\n[5/5] Optimizing F_0.5 Threshold & Margin Filter on Validation Set...")
        val_scores: Dict[str, List[Tuple[str, float]]] = {}
        for eid in val_s1_ids:
            s1_r = s1_dict[eid]
            cands = list(idx.query_candidates(s1_r.get("business_name"), s1_r.get("business_address")))
            cands = [c for c in cands if c != eid]
            if not cands:
                val_scores[eid] = []
                continue
                
            X_val = []
            valid_cands = []
            for c_id in cands:
                if c_id in target_dict:
                    c_r = target_dict[c_id]
                    feat = extract_pair_features(
                        s1_r.get("business_name"), s1_r.get("business_address"), s1_r.get("country"),
                        c_r.get("business_name"), c_r.get("business_address"), c_r.get("country"),
                        cand_id=c_id
                    )
                    X_val.append(feat)
                    valid_cands.append(c_id)
                    
            if X_val:
                probs = self.matcher.predict_proba(np.array(X_val, dtype=np.float32))
                val_scores[eid] = list(zip(valid_cands, probs))
            else:
                val_scores[eid] = []
                
        best_tau, best_delta, best_k, best_f05 = optimize_threshold(val_scores, val_gt_map)
        self.optimal_threshold = best_tau
        self.optimal_delta = best_delta
        self.optimal_k = best_k
        
        print("\n" + "=" * 70)
        print("END-TO-END VALIDATION EVALUATION RESULTS")
        print("=" * 70)
        print(f"Optimal Decision Threshold (tau):  {best_tau:.2f}")
        print(f"Optimal Margin Delta (delta):      {best_delta:.2f}")
        print(f"Optimal Max-K Matches per Entity:  {best_k}")
        print(f"Validation Macro-Averaged F_0.5:   {best_f05 * 100:.2f}% ({best_f05:.4f})")
        print("=" * 70)
        
        return {
            "optimal_threshold": best_tau,
            "optimal_delta": best_delta,
            "optimal_k": best_k,
            "macro_f05_score": best_f05
        }
