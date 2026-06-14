"""
tests/test_anomaly.py — Unit tests for the Isolation Forest anomaly detector.

Run: pytest tests/test_anomaly.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import config
from src.detection.anomaly import AnomalyDetector


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def benign_data():
    """Small benign-like feature matrix for fast detector tests."""
    rng = np.random.default_rng(42)
    # 200 samples, all "normal" (tight Gaussian)
    return rng.normal(loc=0.0, scale=1.0, size=(200, len(config.SELECTED_FEATURES)))


@pytest.fixture
def fitted_detector(benign_data):
    """AnomalyDetector fitted on benign data."""
    det = AnomalyDetector(contamination=0.05, random_state=42)
    det.fit(benign_data)
    return det


# ─── TESTS ────────────────────────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_fit_returns_self(self, benign_data):
        """fit() returns the detector instance (for chaining)."""
        det = AnomalyDetector(contamination=0.05)
        result = det.fit(benign_data)
        assert result is det

    def test_predict_single_sample_returns_tuple(self, fitted_detector, benign_data):
        """Single-sample predict returns (bool, float)."""
        sample = benign_data[0:1]
        is_anom, score = fitted_detector.predict(sample)
        assert isinstance(is_anom, bool)
        assert isinstance(score, float)

    def test_normal_traffic_mostly_not_anomalous(self, fitted_detector, benign_data):
        """Traffic similar to training data should rarely be flagged as anomalous."""
        rng = np.random.default_rng(99)
        # Generate test data from same distribution as training
        normal_test = rng.normal(loc=0.0, scale=1.0, size=(50, len(config.SELECTED_FEATURES)))
        is_anom, _ = fitted_detector.predict(normal_test)
        flagged = np.sum(is_anom)
        # With contamination=0.05, expect ~5% false positives; allow up to 15%
        assert flagged / len(normal_test) < 0.15, \
            f"Too many normal samples flagged as anomalous: {flagged}/{len(normal_test)}"

    def test_extreme_values_flagged_as_anomalous(self, fitted_detector):
        """Flows with extreme feature values should be flagged as anomalous."""
        rng = np.random.default_rng(7)
        # Extreme outlier: 10σ above mean in every feature
        extreme = rng.normal(loc=20.0, scale=0.1, size=(10, len(config.SELECTED_FEATURES)))
        is_anom, _ = fitted_detector.predict(extreme)
        # All extreme samples should be caught
        assert np.all(is_anom), \
            f"Expected all extreme samples to be flagged, got {np.sum(is_anom)}/{len(extreme)}"

    def test_predict_unfitted_raises(self):
        """Calling predict before fit raises RuntimeError."""
        det = AnomalyDetector()
        sample = np.zeros((1, len(config.SELECTED_FEATURES)))
        with pytest.raises(RuntimeError):
            det.predict(sample)

    def test_save_and_load_roundtrip(self, fitted_detector, benign_data, tmp_path):
        """Saved detector produces identical predictions after loading."""
        save_path = tmp_path / "anomaly.pkl"
        fitted_detector.save(save_path)

        loaded = AnomalyDetector.load(save_path)
        sample = benign_data[0:1]

        orig_anom, orig_score = fitted_detector.predict(sample)
        load_anom, load_score = loaded.predict(sample)

        assert orig_anom == load_anom
        assert abs(orig_score - load_score) < 1e-6

    def test_save_unfitted_raises(self, tmp_path):
        """Saving an unfitted detector raises RuntimeError."""
        det = AnomalyDetector()
        with pytest.raises(RuntimeError):
            det.save(tmp_path / "should_not_exist.pkl")

    def test_load_missing_file_raises(self, tmp_path):
        """Loading from a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AnomalyDetector.load(tmp_path / "does_not_exist.pkl")

    def test_batch_predict_returns_arrays(self, fitted_detector, benign_data):
        """Batch predict (N > 1) returns boolean array and float array."""
        is_anom, scores = fitted_detector.predict(benign_data[:10])
        assert hasattr(is_anom, "__len__")
        assert len(is_anom) == 10
        assert len(scores) == 10
        assert scores.dtype == float or np.issubdtype(scores.dtype, np.floating)

    def test_score_threshold_respected(self, fitted_detector, benign_data):
        """is_anomaly == True iff score < ANOMALY_SCORE_THRESHOLD."""
        sample = benign_data[0:1]
        is_anom, score = fitted_detector.predict(sample)
        expected = score < config.ANOMALY_SCORE_THRESHOLD
        assert is_anom == expected
