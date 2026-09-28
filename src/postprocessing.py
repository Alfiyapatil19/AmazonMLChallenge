"""
Post-Processing and F_0.5 Metric Optimization Module (Stage 5).

Implements:
1. Official macro-averaged F_0.5 metric computation (handling singletons)
2. Global decision threshold, ultra-tight margin filter (delta=0.008), and top-k optimization
3. Output TSV file formatting for matching_results.tsv and candidate_pairs.tsv
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd


def compute_f05_score_per_entity(
    predicted_matches: Set[str],
    true_matches: Set[str]
) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity according to official competition rules:
    - Singletons: If true matches is empty, score is 1.0 if prediction is empty, else 0.0.
    - Non-singletons: If prediction is empty, score is 0.0. Otherwise F_0.5 = (1.25 * P * R) / (0.25 * P + R).
    """
    if not true_matches:
        return 1.0 if not predicted_matches else 0.0
        
    if not predicted_matches:
        return 0.0
        
    intersection = len(predicted_matches & true_matches)
    if intersection == 0:
        return 0.0
        
    precision = intersection / len(predicted_matches)
    recall = intersection / len(true_matches)
    
    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0
        
    f05 = (1.25 * precision * recall) / denominator
    return f05


def compute_macro_f05(
    predictions_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]]
) -> float:
    """Computes macro-averaged F_0.5 across all Source 1 entities in ground truth."""
    scores = []
    for s1_id, true_matches in ground_truth_map.items():
        pred_matches = predictions_map.get(s1_id, set())
        score = compute_f05_score_per_entity(pred_matches, true_matches)
        scores.append(score)
        
    return float(np.mean(scores)) if scores else 0.0


def apply_decision_rules(
    pairs: List[Tuple[str, float]],
    tau: float = 0.985,
    delta: float = 0.005,
    max_k: Optional[int] = 7
) -> Set[str]:
    """
    Applies high-precision decision threshold, ultra-tight relative margin filter, and top-k cap.
    """
    if not pairs:
        return set()
    p_max = max(p for _, p in pairs)
    if p_max < tau:
        return set()
    accepted = [(cid, p) for cid, p in pairs if p >= tau and (p >= p_max - delta)]
    accepted.sort(key=lambda x: x[1], reverse=True)
    if max_k is not None and len(accepted) > max_k:
        accepted = accepted[:max_k]
    return {cid for cid, _ in accepted}


def optimize_threshold(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    ground_truth_map: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
    max_k_options: Optional[List[Optional[int]]] = None
) -> Tuple[float, Optional[int], float]:
    """
    Finds optimal decision threshold tau and max_k maximizing macro F_0.5 score.
    
    Returns:
        (best_threshold, best_max_k, best_f05_score)
    """
    if thresholds is None:
        thresholds = list(np.arange(0.70, 0.99, 0.01))
    if max_k_options is None:
        max_k_options = [4, 5, 6, None]
        
    best_tau = 0.985
    best_k = 6
    best_f05 = -1.0
    
    for tau in thresholds:
        for k in max_k_options:
            preds = {}
            for s1_id, pairs in candidate_scores.items():
                filtered = [(cid, p) for cid, p in pairs if p >= tau]
                filtered.sort(key=lambda x: x[1], reverse=True)
                if k is not None and len(filtered) > k:
                    filtered = filtered[:k]
                preds[s1_id] = {cid for cid, _ in filtered}
                
            score = compute_macro_f05(preds, ground_truth_map)
            if score > best_f05:
                best_f05 = score
                best_tau = float(tau)
                best_k = k
                
    return best_tau, best_k, best_f05


def optimize_decision_rules(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    ground_truth_map: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
    delta_options: Optional[List[float]] = None,
    max_k_options: Optional[List[Optional[int]]] = None
) -> Tuple[float, float, Optional[int], float]:
    """
    Finds optimal decision threshold tau, tight margin delta, and max_k maximizing macro F_0.5 score.
    
    Returns:
        (best_threshold, best_delta, best_max_k, best_f05_score)
    """
    if thresholds is None:
        thresholds = [0.950, 0.960, 0.970, 0.975, 0.980, 0.985]
    if delta_options is None:
        delta_options = [0.008, 0.010, 0.015, 0.020]
    if max_k_options is None:
        max_k_options = [4, 5, 6, None]
        
    best_tau = 0.985
    best_delta = 0.008
    best_k = 6
    best_f05 = -1.0
    
    for tau in thresholds:
        for delta in delta_options:
            for k in max_k_options:
                preds = {}
                for s1_id, pairs in candidate_scores.items():
                    preds[s1_id] = apply_decision_rules(pairs, tau=tau, delta=delta, max_k=k)
                    
                score = compute_macro_f05(preds, ground_truth_map)
                if score > best_f05:
                    best_f05 = score
                    best_tau = float(tau)
                    best_delta = float(delta)
                    best_k = k
                    
    return best_tau, best_delta, best_k, best_f05


def write_matching_results_tsv(
    predictions_map: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: str
) -> None:
    """
    Writes predictions to output/matching_results.tsv in the official format:
    source1_entity_id\tmatched_entity_ids
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = predictions_map.get(s1_id, set())
            matched_str = ",".join(sorted(matches)) if matches else ""
            f.write(f"{s1_id}\t{matched_str}\n")
