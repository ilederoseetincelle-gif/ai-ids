"""
tests/test_explainability.py — Unit tests for the SHAP explainer.

Run: pytest tests/test_explainability.py -v

These tests are skipped automatically if the 'shap' package is not installed,
so they never break a setup that hasn't run 'pip install shap'.
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

shap = pytest.importorskip("shap", reason="shap not installed — skipping SHAP tests")

from src.explainability.shap_explainer import SHAPExplainer, TOP_N_CONTRIBUTIONS


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_rf_and_data():
    """Small trained RF and matching test data (module-scoped for speed)."""
    X, y = make_classification(
        n_samples=300,
        n_features=len(config.SELECTED_FEATURES),
        n_informative=10,
        n_redundant=4,
        n_classes=3,
        random_state=42,
    )
    rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1)
    rf.fit(X, y)
    return rf, X, y


@pytest.fixture(scope="module")
def explainer(trained_rf_and_data):
    rf, _, _ = trained_rf_and_data
    return SHAPExplainer(rf, config.SELECTED_FEATURES)


# ─── TESTS ────────────────────────────────────────────────────────────────────

class TestSHAPExplainer:

    def test_explain_returns_dict(self, explainer, trained_rf_and_data):
        """explain() returns a dict (possibly empty on error, but not None)."""
        _, X, y = trained_rf_and_data
        result = explainer.explain(X[0:1], class_idx=0)
        assert isinstance(result, dict)

    def test_explain_top_n_limit(self, explainer, trained_rf_and_data):
        """explain() returns at most TOP_N_CONTRIBUTIONS features."""
        _, X, _ = trained_rf_and_data
        result = explainer.explain(X[0:1], class_idx=0, top_n=TOP_N_CONTRIBUTIONS)
        assert len(result) <= TOP_N_CONTRIBUTIONS

    def test_explain_feature_names_are_valid(self, explainer, trained_rf_and_data):
        """All returned feature names must be in SELECTED_FEATURES."""
        _, X, _ = trained_rf_and_data
        result = explainer.explain(X[0:1], class_idx=0)
        for feat in result:
            assert feat in config.SELECTED_FEATURES, \
                f"Unexpected feature name: {feat}"

    def test_explain_values_are_floats(self, explainer, trained_rf_and_data):
        """All returned SHAP values must be Python floats."""
        _, X, _ = trained_rf_and_data
        result = explainer.explain(X[0:1], class_idx=0)
        for val in result.values():
            assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_explain_ordered_by_magnitude(self, explainer, trained_rf_and_data):
        """Result should be sorted by |shap_value| descending."""
        _, X, _ = trained_rf_and_data
        result = explainer.explain(X[0:1], class_idx=0, top_n=5)
        if len(result) < 2:
            pytest.skip("Too few features to verify ordering")
        magnitudes = [abs(v) for v in result.values()]
        assert magnitudes == sorted(magnitudes, reverse=True), \
            "SHAP contributions should be sorted by |value| descending"

    def test_explain_different_classes_differ(self, explainer, trained_rf_and_data):
        """SHAP explanations for different predicted classes should not be identical."""
        _, X, _ = trained_rf_and_data
        result_0 = explainer.explain(X[0:1], class_idx=0)
        result_1 = explainer.explain(X[0:1], class_idx=1)
        # At least the values should differ (same features may appear but with different magnitudes)
        vals_0 = list(result_0.values())
        vals_1 = list(result_1.values())
        # If both are non-empty and not identical → OK
        if vals_0 and vals_1:
            assert vals_0 != vals_1, \
                "SHAP values for different classes should differ"

    def test_explain_custom_top_n(self, explainer, trained_rf_and_data):
        """Custom top_n parameter is respected."""
        _, X, _ = trained_rf_and_data
        result = explainer.explain(X[0:1], class_idx=0, top_n=3)
        assert len(result) <= 3

    def test_import_error_without_shap(self):
        """SHAPExplainer raises ImportError if shap is not installed (mocked)."""
        import unittest.mock as mock
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "shap":
                raise ImportError("mocked missing shap")
            return real_import(name, *args, **kwargs)

        rng = np.random.default_rng(0)
        X, y = make_classification(n_samples=50, n_features=len(config.SELECTED_FEATURES), random_state=0)
        rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="shap is not installed"):
                SHAPExplainer(rf, config.SELECTED_FEATURES)
