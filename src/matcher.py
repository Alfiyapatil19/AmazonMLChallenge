"""
Machine Learning Matching Model Module for Business Entity Resolution (Stage 4).

Trains a high-performance Gradient Boosted Decision Tree (XGBoost) classifier
to predict pairwise match probabilities between Source 1 and candidate records.
"""

import os
import pickle
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_NAMES, extract_pair_features


class EntityResolutionMatcher:
    """
    Pairwise ML Classifier for predicting match likelihood.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.04,
        subsample: float = 0.88,
        colsample_bytree: float = 0.88,
        min_child_weight: int = 3,
        reg_alpha: float = 0.15,
        reg_lambda: float = 1.2,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the gradient boosted classifier on feature matrix X and labels y."""
        if HAS_XGB:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1
            )
            self.model.fit(X, y)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
            self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of match (class 1)."""
        if self.model is None:
            raise ValueError("Matcher model has not been trained yet.")
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def save_model(self, path: str) -> None:
        """Saves model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": FEATURE_NAMES}, f)

    def load_model(self, path: str) -> None:
        """Loads model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
