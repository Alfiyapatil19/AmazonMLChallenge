"""
Evaluation module for Business Entity Resolution Blocking.

Calculates blocking recall and candidate statistics using ground truth:
- blocking_recall = (true matched IDs present in candidate sets) / (total true matched IDs)
- total S1 records
- total candidate pairs
- average, median, min, and max candidates per S1
"""

from typing import Dict, Set, List, Tuple
import numpy as np
import pandas as pd


def compute_blocking_metrics(
    candidate_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Computes blocking recall and candidate distribution metrics.
    
    Args:
        candidate_map: Dict mapping S1 entity_id -> Set of candidate S2/S3 entity_ids
        ground_truth_map: Dict mapping S1 entity_id -> Set of true matched S2/S3 entity_ids
        
    Returns:
        Dictionary of summary statistics.
    """
    total_s1 = len(ground_truth_map)
    total_true_matches = 0
    retained_true_matches = 0
    
    candidate_counts: List[int] = []
    
    for s1_id, true_matches in ground_truth_map.items():
        total_true_matches += len(true_matches)
        cands = candidate_map.get(s1_id, set())
        candidate_counts.append(len(cands))
        
        # Count intersection of generated candidates and ground truth matches
        retained = cands & true_matches
        retained_true_matches += len(retained)
        
    blocking_recall = retained_true_matches / total_true_matches if total_true_matches > 0 else 0.0
    total_candidate_pairs = sum(candidate_counts)
    
    metrics = {
        "num_s1_records": float(total_s1),
        "total_true_matches": float(total_true_matches),
        "retained_true_matches": float(retained_true_matches),
        "blocking_recall": float(blocking_recall),
        "percentage_retained": float(blocking_recall * 100.0),
        "total_candidate_pairs": float(total_candidate_pairs),
        "avg_candidates_per_s1": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "median_candidates_per_s1": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "min_candidates_per_s1": float(np.min(candidate_counts)) if candidate_counts else 0.0,
        "max_candidates_per_s1": float(np.max(candidate_counts)) if candidate_counts else 0.0,
    }
    return metrics


def load_ground_truth_tsv(gt_path: str) -> Dict[str, Set[str]]:
    """Loads ground truth TSV into a mapping of S1 ID -> Set of matched target IDs."""
    gt_map: Dict[str, Set[str]] = {}
    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=200000):
        for s1_id, matched_str in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if pd.isna(matched_str) or not str(matched_str).strip():
                gt_map[s1_id] = set()
            else:
                matches = {m.strip() for m in str(matched_str).split(",") if m.strip()}
                gt_map[s1_id] = matches
    return gt_map
