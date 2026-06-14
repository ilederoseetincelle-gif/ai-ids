"""
tests/test_training.py — Unit tests for model training and evaluation.

Run: pytest tests/test_training.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import config
from src.training.train import ModelTrainer
from src.training.evaluate import ModelEvaluator


# ─── FIXTURES ─────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_xy():
    """Small, balanced synthetic dataset for fast model training tests."""
    X, y = make_classification(
        n_samples=500,
        n_features=len(config.SELECTED_FEATURES),
        n_informative=10,
        n_redundant=4,
        n_classes=3,
        random_state=42,
    )
    return X, y


@pytest.fixture
def trained_rf(synthetic_xy):
    """A small trained Random Forest for evaluation tests."""
    X, y = synthetic_xy
    rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    rf.fit(X, y)
    return rf


# ─── TRAINER TESTS ────────────────────────────────────────────────────────────
class TestModelTrainer:

    def test_train_random_forest_basic(self, synthetic_xy, tmp_path, monkeypatch):
        """train_random_forest produces a fitted, savable model."""
        X, y = synthetic_xy
        monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "rf.pkl")
        # Use smaller params for test speed
        monkeypatch.setattr(config, "RF_PARAMS", {
            "n_estimators": 20, "random_state": 42, "n_jobs": 1,
        })

        trainer = ModelTrainer()
        rf = trainer.train_random_forest(X, y)

        assert rf is not None
        assert hasattr(rf, "predict")
        assert hasattr(rf, "feature_importances_")
        assert config.MODEL_FILE.exists()

    def test_predictions_in_class_range(self, trained_rf, synthetic_xy):
        """Predictions must be valid class indices (0, 1, 2 for our 3-class fixture)."""
        X, y = synthetic_xy
        preds = trained_rf.predict(X[:50])
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba_sums_to_one(self, trained_rf, synthetic_xy):
        """predict_proba rows should each sum to ~1.0."""
        X, _ = synthetic_xy
        proba = trained_rf.predict_proba(X[:10])
        sums = proba.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_load_model_roundtrip(self, synthetic_xy, tmp_path, monkeypatch):
        """Train → save → load → same predictions."""
        X, y = synthetic_xy
        model_path = tmp_path / "rf.pkl"
        monkeypatch.setattr(config, "MODEL_FILE", model_path)
        monkeypatch.setattr(config, "RF_PARAMS",
                            {"n_estimators": 20, "random_state": 42, "n_jobs": 1})

        trainer = ModelTrainer()
        rf = trainer.train_random_forest(X, y)
        preds_before = rf.predict(X[:50])

        rf_loaded = trainer.load_model(model_path)
        preds_after = rf_loaded.predict(X[:50])

        assert np.array_equal(preds_before, preds_after)

    def test_load_model_missing_file_raises(self, tmp_path):
        """load_model raises FileNotFoundError for missing file."""
        trainer = ModelTrainer()
        with pytest.raises(FileNotFoundError):
            trainer.load_model(tmp_path / "does_not_exist.pkl")


# ─── EVALUATOR TESTS ──────────────────────────────────────────────────────────
class TestModelEvaluator:

    def test_full_report_structure(self, trained_rf, synthetic_xy):
        """full_report returns dict with expected keys and value types."""
        X, y = synthetic_xy
        ev = ModelEvaluator()
        result = ev.full_report(trained_rf, X, y, "TestRF")

        expected_keys = {"model", "accuracy", "macro_f1", "weighted_f1",
                         "ms_per_sample", "fpr_benign"}
        assert expected_keys.issubset(result.keys())
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_confusion_matrix_plot_saved(self, trained_rf, synthetic_xy, tmp_path):
        """plot_confusion_matrix saves a non-empty PNG."""
        X, y = synthetic_xy
        ev = ModelEvaluator()
        save_path = tmp_path / "cm.png"
        ev.plot_confusion_matrix(trained_rf, X, y, save_path, "TestRF")
        assert save_path.exists()
        assert save_path.stat().st_size > 1000  # non-trivial PNG

    def test_roc_plot_saved(self, trained_rf, synthetic_xy, tmp_path):
        """plot_roc_curves saves a non-empty PNG."""
        X, y = synthetic_xy
        ev = ModelEvaluator()
        save_path = tmp_path / "roc.png"
        ev.plot_roc_curves(trained_rf, X, y, save_path, "TestRF")
        assert save_path.exists()
        assert save_path.stat().st_size > 1000

    def test_compare_models_returns_dataframe(self, trained_rf, synthetic_xy):
        """compare_models returns a DataFrame with one row per model."""
        X, y = synthetic_xy
        ev = ModelEvaluator()
        df = ev.compare_models({"RF-A": trained_rf, "RF-B": trained_rf}, X, y)
        assert len(df) == 2
        assert "accuracy" in df.columns
