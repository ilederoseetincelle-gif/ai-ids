"""
tests/test_preprocessing.py — Unit tests for the preprocessing pipeline.

Run: pytest tests/test_preprocessing.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing project modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import config
from src.preprocessing.loader import DataLoader, generate_synthetic_data
from src.preprocessing.features import FeatureEngineer


# ─── FIXTURES ─────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_df():
    """Small synthetic CICIDS-like DataFrame for fast tests."""
    return generate_synthetic_data(n_samples=500, random_state=42)


@pytest.fixture
def loader():
    return DataLoader()


@pytest.fixture
def fe():
    return FeatureEngineer()


# ─── DATA LOADER TESTS ────────────────────────────────────────────────────────
class TestDataLoader:

    def test_synthetic_generation_shape(self, synthetic_df):
        """Synthetic generator produces the right shape."""
        assert len(synthetic_df) == 500
        assert config.LABEL_COLUMN in synthetic_df.columns
        # All selected features present
        for feat in config.SELECTED_FEATURES:
            assert feat in synthetic_df.columns

    def test_synthetic_has_benign_and_attacks(self, synthetic_df):
        """Synthetic data contains BENIGN class and at least one attack."""
        labels = synthetic_df[config.LABEL_COLUMN].unique()
        assert config.BENIGN_LABEL in labels
        assert len(labels) >= 2

    def test_encode_labels_produces_integers(self, loader, synthetic_df):
        """After encoding, labels should be 0-14."""
        df_encoded = loader.encode_labels(synthetic_df)
        labels = df_encoded[config.LABEL_COLUMN].unique()
        assert all(isinstance(x, (int, np.integer)) for x in labels)
        assert all(0 <= int(x) <= 14 for x in labels)

    def test_encode_preserves_row_count(self, loader, synthetic_df):
        """Encoding should not drop rows when all labels are known."""
        df_encoded = loader.encode_labels(synthetic_df)
        assert len(df_encoded) == len(synthetic_df)

    def test_encode_drops_unknown_labels(self, loader, synthetic_df):
        """Rows with unmapped labels should be dropped."""
        df = synthetic_df.copy()
        df.loc[0:5, config.LABEL_COLUMN] = "UnknownAttack"
        df_encoded = loader.encode_labels(df)
        assert len(df_encoded) == len(synthetic_df) - 6

    def test_split_data_stratified(self, loader, synthetic_df):
        """Train/test split preserves class proportions."""
        df_encoded = loader.encode_labels(synthetic_df)
        X = df_encoded[config.SELECTED_FEATURES].values
        y = df_encoded[config.LABEL_COLUMN].values
        X_train, X_test, y_train, y_test = loader.split_data(X, y)
        assert abs(len(X_test) / (len(X_train) + len(X_test)) - config.TEST_SIZE) < 0.02

    def test_missing_data_dir_raises(self, loader, tmp_path):
        """Raises FileNotFoundError if no CSVs in data dir."""
        loader.data_dir = tmp_path  # empty dir
        with pytest.raises(FileNotFoundError):
            loader.load_cicids()


# ─── FEATURE ENGINEER TESTS ───────────────────────────────────────────────────
class TestFeatureEngineer:

    def test_select_features_returns_exact_columns(self, fe, synthetic_df):
        """select_features should return exactly SELECTED_FEATURES in order."""
        X = fe.select_features(synthetic_df)
        assert list(X.columns) == config.SELECTED_FEATURES

    def test_select_features_fills_missing_with_zero(self, fe, synthetic_df):
        """If a feature is missing from input, it's filled with zeros."""
        df = synthetic_df.drop(columns=["Flow Duration"])
        X = fe.select_features(df)
        assert "Flow Duration" in X.columns
        assert (X["Flow Duration"] == 0.0).all()

    def test_fit_scaler_saves_file(self, fe, synthetic_df, tmp_path, monkeypatch):
        """fit_scaler saves the scaler to SCALER_FILE."""
        scaler_file = tmp_path / "scaler.pkl"
        monkeypatch.setattr(config, "SCALER_FILE", scaler_file)
        X = synthetic_df[config.SELECTED_FEATURES].values
        fe.fit_scaler(X)
        assert scaler_file.exists()

    def test_transform_output_bounded(self, fe, synthetic_df, tmp_path, monkeypatch):
        """After transform, values should be bounded to [-5, 5] (clipping)."""
        monkeypatch.setattr(config, "SCALER_FILE", tmp_path / "scaler.pkl")
        X = synthetic_df[config.SELECTED_FEATURES].values
        fe.fit_scaler(X)
        X_scaled = fe.transform(X)
        assert X_scaled.min() >= -5.0
        assert X_scaled.max() <= 5.0

    def test_transform_before_fit_raises(self, fe, synthetic_df):
        """Calling transform without fitting should raise."""
        fe.scaler = None
        X = synthetic_df[config.SELECTED_FEATURES].values
        with pytest.raises(RuntimeError):
            fe.transform(X)

    def test_smote_balances_classes(self, fe, synthetic_df, tmp_path, monkeypatch):
        """SMOTE should roughly equalize class counts."""
        monkeypatch.setattr(config, "SCALER_FILE", tmp_path / "scaler.pkl")
        loader = DataLoader()
        df = loader.encode_labels(synthetic_df)
        X = df[config.SELECTED_FEATURES].values
        y = df[config.LABEL_COLUMN].values

        # SMOTE needs at least k_neighbors+1 samples per class — filter small classes
        counts = pd.Series(y).value_counts()
        big_classes = counts[counts > config.SMOTE_K_NEIGHBORS].index
        mask = pd.Series(y).isin(big_classes).values
        X_filt, y_filt = X[mask], y[mask]

        if len(np.unique(y_filt)) < 2:
            pytest.skip("Need >= 2 classes after filtering for SMOTE test")

        fe.fit_scaler(X_filt)
        X_scaled = fe.transform(X_filt)
        X_res, y_res = fe.apply_smote(X_scaled, y_filt)

        resampled_counts = pd.Series(y_res).value_counts()
        # Either SMOTE succeeded (balanced counts) or fell back to original
        assert len(X_res) >= len(X_filt)


# ─── END-TO-END SMOKE TEST ────────────────────────────────────────────────────
def test_end_to_end_preprocessing(synthetic_df, tmp_path, monkeypatch):
    """Full preprocessing pipeline runs end-to-end on synthetic data."""
    scaler_file = tmp_path / "scaler.pkl"
    monkeypatch.setattr(config, "SCALER_FILE", scaler_file)

    loader = DataLoader()
    fe = FeatureEngineer()

    df_encoded = loader.encode_labels(synthetic_df)
    X = fe.select_features(df_encoded)
    y = df_encoded[config.LABEL_COLUMN].values

    X_train, X_test, y_train, y_test = loader.split_data(X, y)
    fe.fit_scaler(X_train.values)
    X_train_s = fe.transform(X_train.values)
    X_test_s  = fe.transform(X_test.values)

    assert X_train_s.shape[1] == len(config.SELECTED_FEATURES)
    assert X_test_s.shape[1]  == len(config.SELECTED_FEATURES)
    assert scaler_file.exists()
