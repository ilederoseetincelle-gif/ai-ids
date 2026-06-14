"""
features.py — Feature engineering for the AI-IDS.

Provides feature selection, scaling, and class-imbalance correction.
The scaler fitted here is saved and later loaded by the live detection
engine to ensure training-inference consistency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

import config


class FeatureEngineer:
    """Handles feature selection, scaling, and class imbalance correction."""

    def __init__(self):
        self.scaler: StandardScaler | None = None

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only the columns listed in config.SELECTED_FEATURES.
        Missing columns are filled with zeros and a warning is printed.

        Args:
            df: DataFrame with the full CICIDS2017 columns.

        Returns:
            DataFrame with only the 20 selected features.
        """
        available = [c for c in config.SELECTED_FEATURES if c in df.columns]
        missing = set(config.SELECTED_FEATURES) - set(available)

        if missing:
            print(f"WARNING: {len(missing)} features missing from input data:")
            for f in sorted(missing):
                print(f"  - {f}")
            print("Missing features will be filled with zeros.")

        X = df[available].copy()
        for feat in missing:
            X[feat] = 0.0

        # Enforce the exact column order from config
        X = X[config.SELECTED_FEATURES]
        return X

    def fit_scaler(self, X_train: np.ndarray) -> StandardScaler:
        """
        Fit a StandardScaler on the training data ONLY. Save it to disk.
        NEVER call fit on test data (data leakage).

        Args:
            X_train: Training feature matrix.

        Returns:
            Fitted StandardScaler instance.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        joblib.dump(self.scaler, config.SCALER_FILE)
        print(f"Scaler saved to {config.SCALER_FILE}")
        return self.scaler

    def transform(self, X: np.ndarray, scaler: StandardScaler | None = None) -> np.ndarray:
        """
        Apply the scaler transformation. Clips extreme outliers beyond
        ±5 standard deviations to improve model robustness.

        Args:
            X: Feature matrix to transform.
            scaler: Optional externally loaded scaler. Uses self.scaler if None.

        Returns:
            Scaled and clipped feature matrix.
        """
        scaler = scaler or self.scaler
        if scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler first.")
        X_scaled = scaler.transform(X)
        X_scaled = np.clip(X_scaled, -5.0, 5.0)
        return X_scaled

    def apply_smote(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_neighbors: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to oversample minority attack classes.
        Classes with fewer than k_neighbors+1 samples cannot be oversampled;
        they are skipped with a warning.

        Args:
            X_train: Training features.
            y_train: Training labels.
            k_neighbors: SMOTE neighbor parameter.

        Returns:
            (X_resampled, y_resampled)
        """
        if k_neighbors is None:
            k_neighbors = config.SMOTE_K_NEIGHBORS

        # Print class distribution before
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nClass distribution BEFORE SMOTE:")
        for cls, cnt in zip(unique, counts):
            name = config.LABEL_NAMES.get(int(cls), f"UNKNOWN({cls})")
            print(f"  {name:40s} {cnt:>10,}")

        # SMOTE requires at least k_neighbors+1 samples per minority class
        min_required = k_neighbors + 1
        eligible_classes = {cls: cnt for cls, cnt in zip(unique, counts) if cnt >= min_required}

        if len(eligible_classes) < len(unique):
            skipped = set(unique) - set(eligible_classes.keys())
            print(f"WARNING: Skipping SMOTE for {len(skipped)} classes with <{min_required} samples")

        try:
            smote = SMOTE(random_state=config.RANDOM_SEED, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        except ValueError as e:
            print(f"SMOTE failed: {e}")
            print("Falling back to original data without oversampling.")
            return X_train, y_train

        unique_r, counts_r = np.unique(y_res, return_counts=True)
        print("\nClass distribution AFTER SMOTE:")
        for cls, cnt in zip(unique_r, counts_r):
            name = config.LABEL_NAMES.get(int(cls), f"UNKNOWN({cls})")
            print(f"  {name:40s} {cnt:>10,}")

        return X_res, y_res

    def get_feature_importance(self, model, feature_names: list[str]) -> pd.DataFrame:
        """
        Extract feature importance from a fitted tree-based model.

        Args:
            model: Fitted RandomForestClassifier or similar with feature_importances_.
            feature_names: List of feature names matching the model's input columns.

        Returns:
            DataFrame with columns ['feature', 'importance'], sorted descending.
        """
        if not hasattr(model, "feature_importances_"):
            raise AttributeError("Model has no feature_importances_ attribute.")
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df.to_csv(config.FEATURE_IMPORTANCE_FILE, index=False)
        return importance_df

    @staticmethod
    def load_scaler(path: Path = None) -> StandardScaler:
        """Load a previously saved scaler from disk."""
        if path is None:
            path = config.SCALER_FILE
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {path}. Run training first."
            )
        return joblib.load(path)
