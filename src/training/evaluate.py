"""
evaluate.py — Model evaluation and comparison.

Produces:
- Classification reports (precision/recall/F1 per class)
- Confusion matrix heatmap
- ROC-AUC curves (one-vs-rest per class)
- Side-by-side model comparison table
- False positive rate analysis on BENIGN class
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # headless backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import config


class ModelEvaluator:
    """Compute metrics and generate plots for trained models."""

    def full_report(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model",
    ) -> dict:
        """
        Print a full evaluation report and return metrics as a dict.

        Args:
            model: Trained classifier with predict and predict_proba methods.
            X_test: Test feature matrix.
            y_test: True labels.
            model_name: Display name for the report header.

        Returns:
            Dictionary with all computed metrics.
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluation Report — {model_name}")
        print('=' * 60)

        # Timed predictions
        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t0
        ms_per_sample = 1000 * pred_time / len(X_test)

        # Core metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"\nAccuracy:     {acc:.4f}")
        print(f"Macro F1:     {macro_f1:.4f}")
        print(f"Weighted F1:  {weighted_f1:.4f}")
        print(f"Inference:    {ms_per_sample:.3f} ms/sample")

        # Per-class breakdown — labels must include both true and predicted
        # classes (model may predict a class that doesn't appear in test)
        all_labels = sorted(set(np.unique(y_test)).union(set(np.unique(y_pred))))
        target_names = [config.LABEL_NAMES.get(int(i), f"CLS_{i}") for i in all_labels]
        print("\nPer-class classification report:")
        print(classification_report(
            y_test, y_pred,
            labels=all_labels,
            target_names=target_names,
            zero_division=0,
        ))

        # False Positive Rate on BENIGN class
        fpr_benign = self._benign_fpr(y_test, y_pred)
        print(f"False Positive Rate (BENIGN): {fpr_benign:.4f}")

        return {
            "model": model_name,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "ms_per_sample": ms_per_sample,
            "fpr_benign": fpr_benign,
        }

    def _benign_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the false positive rate for BENIGN traffic:
        fraction of actually-benign flows incorrectly flagged as an attack.

        FPR = FP / (FP + TN) where positive = attack, negative = benign.
        """
        benign_id = config.ATTACK_LABELS[config.BENIGN_LABEL]
        benign_mask = y_true == benign_id
        if benign_mask.sum() == 0:
            return 0.0
        benign_predictions = y_pred[benign_mask]
        false_positives = (benign_predictions != benign_id).sum()
        total_benign = benign_mask.sum()
        return float(false_positives / total_benign)

    def plot_confusion_matrix(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Path,
        model_name: str = "Model",
        normalize: bool = True,
    ) -> None:
        """
        Generate and save a confusion matrix heatmap.

        Args:
            model: Trained classifier.
            X_test: Test feature matrix.
            y_test: True labels.
            save_path: Where to save the PNG.
            model_name: Title for the plot.
            normalize: If True, normalize counts to proportions.
        """
        y_pred = model.predict(X_test)
        labels = sorted(np.unique(y_test))
        class_names = [config.LABEL_NAMES.get(i, f"CLS_{i}") for i in labels]

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt=".2f" if normalize else "d",
            cmap="Blues", xticklabels=class_names, yticklabels=class_names,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Confusion matrix saved to {save_path}")

    def plot_roc_curves(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Path,
        model_name: str = "Model",
    ) -> None:
        """
        Plot one-vs-rest ROC curves for each class.

        Args:
            model: Classifier with predict_proba method.
            X_test: Test features.
            y_test: True labels.
            save_path: Where to save the PNG.
            model_name: Title for the plot.
        """
        if not hasattr(model, "predict_proba"):
            print(f"Skipping ROC curves — {model_name} has no predict_proba.")
            return

        classes = sorted(np.unique(y_test))
        y_proba = model.predict_proba(X_test)

        # Align proba columns to classes present in y_test
        proba_map = dict(zip(model.classes_, range(len(model.classes_))))

        y_bin = label_binarize(y_test, classes=classes)
        if y_bin.shape[1] == 1:
            # Binary case — expand to 2 columns
            y_bin = np.hstack([1 - y_bin, y_bin])

        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cls in enumerate(classes):
            if cls not in proba_map:
                continue
            col = proba_map[cls]
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, col])
            try:
                auc = roc_auc_score(y_bin[:, i], y_proba[:, col])
            except ValueError:
                auc = float("nan")
            name = config.LABEL_NAMES.get(cls, f"CLS_{cls}")
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curves (One-vs-Rest) — {model_name}", fontsize=14)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"ROC curves saved to {save_path}")

    def compare_models(
        self,
        models: Dict[str, object],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build a comparison table for multiple trained models.

        Args:
            models: Dictionary {name: trained_model}.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            DataFrame with a row per model and columns for each metric.
        """
        rows = []
        for name, model in models.items():
            if model is None:
                continue
            rows.append(self.full_report(model, X_test, y_test, name))

        df = pd.DataFrame(rows)
        print("\n" + "=" * 60)
        print("Model Comparison Summary")
        print("=" * 60)
        print(df.to_string(index=False))
        return df
