"""
anomaly.py — Isolation Forest anomaly detector for zero-day traffic.

Trained exclusively on benign (Normal Traffic) flows from the training set.
At inference it scores every flow; flows that the supervised classifier labels
as BENIGN but that look statistically unusual get flagged as low-confidence
"Unknown / Anomaly" alerts.

Why Isolation Forest:
- Unsupervised — no attack labels needed.
- O(n log n) training, O(log n) inference — fast enough for real-time use.
- Returns a continuous anomaly score via decision_function, so we can tune
  the threshold without retraining.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

import config


class AnomalyDetector:
    """
    Wraps IsolationForest trained on benign-only flow features.

    Usage:
        detector = AnomalyDetector()
        detector.fit(X_benign_scaled)
        detector.save(config.ANOMALY_MODEL_FILE)

        # At inference:
        detector = AnomalyDetector.load(config.ANOMALY_MODEL_FILE)
        is_anom, score = detector.predict(x_scaled)
    """

    def __init__(self, contamination: float = None, random_state: int = None):
        self.contamination = contamination if contamination is not None else config.ANOMALY_CONTAMINATION
        self.random_state = random_state if random_state is not None else config.RANDOM_SEED
        self._model: IsolationForest | None = None

    def fit(self, X_benign: np.ndarray) -> "AnomalyDetector":
        """
        Fit on benign-only scaled feature matrix.

        Args:
            X_benign: (N, F) array of scaled features from benign flows only.

        Returns:
            self
        """
        self._model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_benign)
        print(
            f"[AnomalyDetector] Fitted on {len(X_benign):,} benign samples "
            f"(contamination={self.contamination})"
        )
        return self

    def predict(self, X: np.ndarray) -> Tuple[bool, float]:
        """
        Score a single flow (or batch).

        Args:
            X: (1, F) or (N, F) scaled feature array.

        Returns:
            (is_anomaly, score) where score < ANOMALY_SCORE_THRESHOLD is anomalous.
            For batches returns (array_of_bool, array_of_float).
        """
        if self._model is None:
            raise RuntimeError("AnomalyDetector not fitted. Call fit() or load() first.")

        scores = self._model.decision_function(X)
        is_anomaly = scores < config.ANOMALY_SCORE_THRESHOLD

        if X.shape[0] == 1:
            return bool(is_anomaly[0]), float(scores[0])
        return is_anomaly, scores

    def save(self, path: Path = None) -> None:
        """Persist model to disk."""
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted AnomalyDetector.")
        path = Path(path) if path is not None else config.ANOMALY_MODEL_FILE
        joblib.dump(self._model, path)
        print(f"[AnomalyDetector] Saved to {path}")

    @classmethod
    def load(cls, path: Path = None) -> "AnomalyDetector":
        """
        Load a previously saved detector from disk.

        Args:
            path: Path to the .pkl file (default: config.ANOMALY_MODEL_FILE).

        Returns:
            AnomalyDetector with _model populated.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        path = Path(path) if path is not None else config.ANOMALY_MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Anomaly model not found at {path}. "
                f"Run: python train_pipeline.py"
            )
        detector = cls.__new__(cls)
        detector.contamination = config.ANOMALY_CONTAMINATION
        detector.random_state = config.RANDOM_SEED
        detector._model = joblib.load(path)
        return detector
