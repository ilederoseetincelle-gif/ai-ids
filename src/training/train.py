"""
train.py — Model training for the AI-IDS.

Implements:
- Random Forest baseline training
- XGBoost and MLP comparison models
- GridSearchCV hyperparameter tuning

All models operate on the scaled feature matrix from FeatureEngineer.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import config


class ModelTrainer:
    """Train and compare ML models for intrusion detection."""

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        save: bool = True,
    ) -> RandomForestClassifier:
        """
        Train a Random Forest classifier with the parameters from config.RF_PARAMS.

        Args:
            X_train: Scaled training features.
            y_train: Integer-encoded labels.
            save: If True, save the fitted model to config.MODEL_FILE.

        Returns:
            Trained RandomForestClassifier.
        """
        print(f"Training Random Forest with params: {config.RF_PARAMS}")
        rf = RandomForestClassifier(**config.RF_PARAMS)

        t0 = time.time()
        rf.fit(X_train, y_train)
        duration = time.time() - t0
        print(f"Random Forest trained in {duration:.1f}s")

        if save:
            joblib.dump(rf, config.MODEL_FILE)
            print(f"Model saved to {config.MODEL_FILE}")

        return rf

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        save: bool = True,
    ):
        """
        Train an XGBoost classifier using config.XGB_PARAMS.

        Args:
            X_train: Scaled training features.
            y_train: Integer-encoded labels.
            save: If True, save the fitted model to config.XGB_MODEL_FILE.

        Returns:
            Trained XGBClassifier, or None if xgboost is not installed.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("xgboost not installed — skipping XGBoost training.")
            return None

        print(f"Training XGBoost classifier with params: {config.XGB_PARAMS}")
        xgb = XGBClassifier(**config.XGB_PARAMS)

        t0 = time.time()
        xgb.fit(X_train, y_train)
        duration = time.time() - t0
        print(f"XGBoost trained in {duration:.1f}s")

        if save:
            joblib.dump(xgb, config.XGB_MODEL_FILE)
            print(f"XGBoost model saved to {config.XGB_MODEL_FILE}")

        return xgb

    def train_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> MLPClassifier:
        """
        Train a small MLP neural network for comparison.

        Args:
            X_train: Scaled training features.
            y_train: Integer-encoded labels.

        Returns:
            Trained MLPClassifier.
        """
        print("Training MLP neural network...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=50,
            random_state=config.RANDOM_SEED,
            early_stopping=True,
            verbose=False,
        )

        t0 = time.time()
        mlp.fit(X_train, y_train)
        duration = time.time() - t0
        print(f"MLP trained in {duration:.1f}s")

        return mlp

    def hyperparameter_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 3,
    ) -> RandomForestClassifier:
        """
        Run GridSearchCV over config.GRID_SEARCH_PARAMS.
        WARNING: This can take 1–3 hours on the full CICIDS2017 dataset.

        Args:
            X_train: Scaled training features.
            y_train: Integer-encoded labels.
            cv: Number of cross-validation folds (default 3 for speed).

        Returns:
            Best RandomForestClassifier after grid search.
        """
        print(f"Running GridSearchCV (cv={cv}) — this may take a while...")
        base_rf = RandomForestClassifier(
            random_state=config.RANDOM_SEED,
            n_jobs=1,  # parallelism at GridSearchCV level instead
            class_weight="balanced",
        )
        gs = GridSearchCV(
            base_rf,
            config.GRID_SEARCH_PARAMS,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=2,
        )

        t0 = time.time()
        gs.fit(X_train, y_train)
        duration = time.time() - t0
        print(f"GridSearchCV completed in {duration:.1f}s ({duration/60:.1f} min)")

        print(f"\nBest params: {gs.best_params_}")
        print(f"Best macro-F1: {gs.best_score_:.4f}")

        joblib.dump(gs.best_estimator_, config.MODEL_TUNED_FILE)
        print(f"Tuned model saved to {config.MODEL_TUNED_FILE}")

        return gs.best_estimator_

    @staticmethod
    def load_model(path: Path = None):
        """Load a previously trained model from disk."""
        if path is None:
            path = config.MODEL_FILE
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run training first."
            )
        return joblib.load(path)

    @staticmethod
    def load_best_model():
        """
        Load the best available model: XGBoost if present, otherwise RF.
        Raises FileNotFoundError if neither is found.
        """
        if config.XGB_MODEL_FILE.exists():
            print(f"Loading XGBoost model from {config.XGB_MODEL_FILE}")
            return joblib.load(config.XGB_MODEL_FILE)
        if config.MODEL_FILE.exists():
            print(f"Loading Random Forest model from {config.MODEL_FILE}")
            return joblib.load(config.MODEL_FILE)
        raise FileNotFoundError(
            f"No trained model found. "
            f"Expected {config.XGB_MODEL_FILE} or {config.MODEL_FILE}.\n"
            f"Run: python train_pipeline.py"
        )
