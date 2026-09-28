"""
Unit and Integration tests for End-to-End Entity Resolution stages.
"""

import unittest
import numpy as np
from src.features import extract_pair_features, FEATURE_NAMES
from src.matcher import EntityResolutionMatcher
from src.postprocessing import (
    compute_f05_score_per_entity,
    compute_macro_f05,
    optimize_threshold,
    apply_decision_rules
)


class TestEndToEndStages(unittest.TestCase):

    # 1. Feature Engineering Tests
    def test_feature_extraction_dimension(self):
        feat = extract_pair_features(
            "Acme Logistics Inc", "100 Main St, New York, NY", "US",
            "Acme Logistics LLC", "100 Main Street, New York, NY", "US",
            cand_id="S2-1"
        )
        self.assertEqual(len(feat), len(FEATURE_NAMES))
        self.assertEqual(len(feat), 33)
        self.assertTrue(all(isinstance(v, (int, float)) for v in feat))
        self.assertGreater(feat[1], 0.5)

    def test_feature_extraction_missing_values(self):
        feat = extract_pair_features(None, None, None, None, None, None)
        self.assertEqual(len(feat), len(FEATURE_NAMES))
        addr_null_idx = FEATURE_NAMES.index("addr_null")
        self.assertEqual(feat[addr_null_idx], 1.0) # addr_null

    # 2. Matcher Model Tests
    def test_matcher_fit_and_predict(self):
        pos = np.ones((20, len(FEATURE_NAMES)), dtype=np.float32)
        neg = np.zeros((20, len(FEATURE_NAMES)), dtype=np.float32)
        X = np.vstack([pos, neg])
        y = np.array([1]*20 + [0]*20, dtype=np.int32)

        matcher = EntityResolutionMatcher(n_estimators=20, max_depth=3, min_child_weight=1)
        matcher.fit(X, y)
        probs = matcher.predict_proba(X)
        self.assertEqual(len(probs), 40)
        self.assertGreater(probs[0], probs[20])

    # 3. Macro F_0.5 Metric Tests
    def test_f05_singletons(self):
        self.assertEqual(compute_f05_score_per_entity(set(), set()), 1.0)
        self.assertEqual(compute_f05_score_per_entity({"S2-1"}, set()), 0.0)

    def test_f05_precision_weighting(self):
        true_m = {"S2-1", "S3-1"}
        self.assertAlmostEqual(compute_f05_score_per_entity({"S2-1", "S3-1"}, true_m), 1.0)
        f_over = compute_f05_score_per_entity({"S2-1", "S3-1", "S2-999"}, true_m)
        self.assertAlmostEqual(f_over, 0.7142857, places=4)

    # 4. Threshold Optimization Tests
    def test_threshold_optimization(self):
        cand_scores = {
            "S1-1": [("S2-1", 0.95), ("S2-2", 0.20)],
            "S1-2": [("S3-1", 0.85), ("S3-2", 0.40)],
            "S1-3": [("S2-3", 0.15)]
        }
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": {"S3-1"},
            "S1-3": set()
        }
        best_tau, best_k, best_f05 = optimize_threshold(cand_scores, gt, thresholds=[0.3, 0.5, 0.7, 0.8])
        self.assertGreaterEqual(best_f05, 0.99)
        self.assertGreaterEqual(best_tau, 0.5)


if __name__ == "__main__":
    unittest.main()
