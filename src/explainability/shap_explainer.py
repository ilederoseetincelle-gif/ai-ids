"""
shap_explainer.py — Per-alert SHAP feature contributions.

Uses TreeExplainer (native tree-based model support, no sampling needed) to
compute exact Shapley values for each prediction. For a single sample, this
takes ~1-10ms on modern hardware — acceptable for real-time alerting.

SHAP values tell you HOW MUCH each feature shifted the prediction away from
the expected value. A large positive SHAP value for "SYN Flag Count" on a
PortScan alert means that feature is the primary driver of the detection.

This module is optional: if 'shap' is not installed, SHAPExplainer raises
ImportError on construction and the engine falls back gracefully.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

import config

# Top-N feature contributions to include per alert (keeps alert JSON small)
TOP_N_CONTRIBUTIONS = 5


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer for a trained tree-based classifier.

    Handles both:
    - RandomForestClassifier: shap_values is list[(n_samples, n_features)]
      one array per class.
    - XGBClassifier: shap_values may be (n_samples, n_features, n_classes)
      or a list, depending on shap version.
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: Fitted RandomForestClassifier or XGBClassifier.
            feature_names: Ordered list of feature names matching model input columns.

        Raises:
            ImportError: If the shap package is not installed.
        """
        try:
            import shap as _shap
        except ImportError as e:
            raise ImportError(
                "shap is not installed. Run: pip install shap>=0.47"
            ) from e

        self.feature_names = feature_names
        self._explainer = _shap.TreeExplainer(model)

    def explain(
        self,
        X_scaled: np.ndarray,
        class_idx: int,
        top_n: int = TOP_N_CONTRIBUTIONS,
    ) -> dict[str, float]:
        """
        Compute SHAP values for one sample and return the top-N contributions
        for the predicted class.

        Args:
            X_scaled: (1, F) scaled feature array (already clipped to ±5σ).
            class_idx: The predicted class index (SHAP values are class-specific).
            top_n: How many features to include in the returned dict.

        Returns:
            Dict mapping feature_name → shap_value, sorted by |shap_value| desc,
            limited to top_n entries. Empty dict on any error (non-fatal).
        """
        try:
            raw = self._explainer.shap_values(X_scaled, check_additivity=False)
            vals = self._extract_class_values(raw, class_idx)

            if vals is None or len(vals) == 0:
                return {}

            # Sort by absolute contribution, take top N
            indices = np.argsort(np.abs(vals))[::-1][:top_n]
            return {
                self.feature_names[i]: round(float(vals[i]), 4)
                for i in indices
            }
        except Exception:
            return {}

    def _extract_class_values(
        self,
        raw,
        class_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Normalize shap_values output across SHAP versions and model types.

        SHAP returns several shapes depending on model and version:
        - List of (n_samples, n_features): one entry per class (RF, older SHAP).
        - 3-D array (n_samples, n_features, n_classes): XGBoost multi-class.
        - 2-D array (n_samples, n_features): binary models.
        """
        if isinstance(raw, list):
            # list[(n_samples, n_features)] — index by class
            if class_idx < len(raw):
                return raw[class_idx][0]
            return raw[-1][0]  # fallback to last class

        arr = np.array(raw)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            idx = min(class_idx, arr.shape[2] - 1)
            return arr[0, :, idx]
        if arr.ndim == 2:
            # (n_samples, n_features) — binary or already class-specific
            return arr[0]
        return None
