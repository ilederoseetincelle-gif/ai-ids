"""
loader.py — Load and clean the CICIDS2017 dataset.

This module handles:
- Loading 8 CSV files and merging into a single DataFrame
- Removing NaN, Inf, and duplicate rows
- Encoding string labels to integers
- Stratified train/test split

The CICIDS2017 raw CSVs have several known quality issues handled here:
- Whitespace in column names (all leading/trailing spaces stripped)
- Inf values in a handful of flow-rate columns
- ~10k duplicate rows
- Latin-1 encoded special characters in Web Attack labels
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


class DataLoader:
    """Handles loading, cleaning, and splitting the CICIDS2017 dataset."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR

    def load_cicids(self) -> pd.DataFrame:
        """
        Load the CICIDS2017 dataset.

        Prefers a pre-cleaned single CSV (cicids2017_cleaned.csv) if present.
        Falls back to loading and merging all raw *.csv files in data_dir.

        Returns:
            Cleaned DataFrame ready for feature selection and encoding.

        Raises:
            FileNotFoundError: If no CSV files are found in data_dir.
        """
        cleaned_file = self.data_dir / "cicids2017_cleaned.csv"
        if cleaned_file.exists():
            print(f"Loading pre-cleaned dataset from {cleaned_file} ...")
            df = pd.read_csv(cleaned_file, low_memory=False)
            print(f"Shape: {df.shape}")
            print("\nLabel distribution:")
            print(df[config.LABEL_COLUMN].value_counts())
            return df.reset_index(drop=True)

        csv_files = sorted(glob.glob(str(self.data_dir / "*.csv")))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}. "
                f"Download CICIDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html "
                f"and place the 8 CSV files in {self.data_dir}/"
            )

        print(f"Found {len(csv_files)} CSV files. Loading...")
        frames = []
        for f in tqdm(csv_files, desc="Loading CSVs"):
            # Use latin1 because some label columns contain 0x96 byte (Web Attack labels)
            df = pd.read_csv(f, encoding="latin1", low_memory=False)
            frames.append(df)

        df = pd.concat(frames, ignore_index=True)
        print(f"Raw shape: {df.shape}")

        # CRITICAL: strip whitespace from column names — CICIDS2017 has
        # leading spaces in many column names (e.g. " Label" instead of "Label")
        df.columns = df.columns.str.strip()

        # The raw CSVs use a "Label" column with 15 fine-grained class names.
        # Remap to the 7-class "Attack Type" scheme used throughout this project.
        if "Label" in df.columns and config.LABEL_COLUMN not in df.columns:
            df = _remap_raw_labels(df)

        # Clean data quality issues
        print("Cleaning data quality issues...")
        initial_rows = len(df)

        # Replace Inf with NaN (Flow Bytes/s and Flow Packets/s have divisions
        # by zero when Flow Duration = 0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with any NaN values
        df.dropna(inplace=True)
        print(f"  Dropped {initial_rows - len(df)} rows with NaN/Inf values")

        # Drop duplicates (CICIDS2017 has ~10k exact duplicates)
        before_dedup = len(df)
        df.drop_duplicates(inplace=True)
        print(f"  Dropped {before_dedup - len(df)} duplicate rows")

        print(f"Clean shape: {df.shape}")
        print("\nLabel distribution:")
        print(df[config.LABEL_COLUMN].value_counts())

        return df.reset_index(drop=True)

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map string labels to integers using config.ATTACK_LABELS.
        Rows with unmapped labels are removed with a warning.

        Args:
            df: DataFrame with a string Label column.

        Returns:
            DataFrame with Label column converted to integers.
        """
        df = df.copy()
        label_col = config.LABEL_COLUMN

        # Map using our consistent integer scheme
        df["_label_int"] = df[label_col].map(config.ATTACK_LABELS)

        unmapped = df["_label_int"].isna().sum()
        if unmapped > 0:
            unknown_labels = df[df["_label_int"].isna()][label_col].unique()
            print(f"WARNING: {unmapped} rows have unknown labels: {unknown_labels}")
            print("  Dropping these rows.")
            df = df.dropna(subset=["_label_int"])

        df[label_col] = df["_label_int"].astype(int)
        df = df.drop(columns=["_label_int"])

        return df

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = None,
        random_state: int = None,
        chronological: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split features and labels into train and test sets.

        Args:
            X: Feature matrix.
            y: Labels (integer-encoded).
            test_size: Fraction for test set (default 0.2).
            random_state: Random seed (only used when chronological=False).
            chronological: If True, take the first (1-test_size) rows as train
                and the remainder as test, preserving capture-time ordering.
                Use this for real CICIDS2017 data (Mon→Fri ordering) to avoid
                future-flow leakage into training. Use False for synthetic data.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = config.TEST_SIZE
        if random_state is None:
            random_state = config.RANDOM_SEED

        if chronological:
            # Preserve temporal order — no shuffle. CICIDS2017 CSVs are
            # ordered Mon→Fri so row order reflects capture time; a random
            # split would leak Friday attacks into Monday training.
            n = len(X)
            split_idx = int(n * (1.0 - test_size))
            X_train = X.iloc[:split_idx]
            X_test  = X.iloc[split_idx:]
            y_train = np.asarray(y)[:split_idx]
            y_test  = np.asarray(y)[split_idx:]
            print(f"\nChronological split: rows 0–{split_idx - 1} → train, "
                  f"{split_idx}–{n - 1} → test")
            print(f"Train set: {len(X_train)} samples")
            print(f"Test set:  {len(X_test)} samples")
            return X_train, X_test, y_train, y_test

        # Stratification requires at least 2 samples per class. Fall back to
        # a non-stratified split when any class has only 1 sample.
        unique, counts = np.unique(np.asarray(y), return_counts=True)
        stratify = y if counts.min() >= 2 else None
        if stratify is None:
            print(f"[warn] Some classes have <2 samples ({counts.min()}) — using non-stratified split.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set:  {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test


# ─── SYNTHETIC DATA FOR TESTING ──────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset mimicking CICIDS2017 structure.
    Used for end-to-end testing when the real dataset isn't available.

    Args:
        n_samples: Total number of rows to generate.
        random_state: Random seed.

    Returns:
        DataFrame with the same columns and label distribution as CICIDS2017.
    """
    rng = np.random.default_rng(random_state)

    # Class weights aligned with config.ATTACK_LABELS simplified labels.
    # Approximate CICIDS2017 proportions: BENIGN ~80%, DoS ~9%, PortScan ~6%, etc.
    class_weights = {
        "Normal Traffic": 0.80,
        "DoS":            0.094,
        "Port Scanning":  0.060,
        "DDoS":           0.030,
        "Brute Force":    0.005,
        "Bots":           0.001,
        "Web Attacks":    0.010,
    }
    # Normalize to exactly 1.0
    total = sum(class_weights.values())
    labels = list(class_weights.keys())
    probs = [w / total for w in class_weights.values()]
    label_samples = rng.choice(labels, size=n_samples, p=probs)

    data = {}
    for feat in config.SELECTED_FEATURES:
        # Each feature gets a different distribution depending on the name
        if "Duration" in feat:
            data[feat] = rng.exponential(scale=1e6, size=n_samples)
        elif "Packets" in feat:
            data[feat] = rng.poisson(lam=10, size=n_samples).astype(float)
        elif "Length" in feat or "Bytes" in feat:
            data[feat] = rng.exponential(scale=500, size=n_samples)
        elif "IAT" in feat:
            data[feat] = rng.exponential(scale=1e5, size=n_samples)
        elif "Flag" in feat:
            data[feat] = rng.poisson(lam=2, size=n_samples).astype(float)
        else:
            data[feat] = rng.normal(loc=0, scale=1, size=n_samples)

    # Add signal: shift feature distributions per class so ML can learn
    for i, label in enumerate(label_samples):
        if label != config.BENIGN_LABEL:
            class_idx = config.ATTACK_LABELS[label]
            for feat in config.SELECTED_FEATURES[:10]:
                data[feat][i] *= (1 + 0.3 * class_idx)

    df = pd.DataFrame(data)
    df[config.LABEL_COLUMN] = label_samples
    return df


def _remap_raw_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the 15 raw CICIDS2017 Label strings to the 7-class Attack Type scheme.
    Rows with unrecognised labels are kept as-is; encode_labels will warn and
    drop them later.
    """
    raw_to_group = {
        "BENIGN": "Normal Traffic",
        # DoS family
        "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
        "DoS Slowhttptest": "DoS", "Heartbleed": "DoS",
        # DDoS
        "DDoS": "DDoS",
        # Port scan
        "PortScan": "Port Scanning",
        # Brute force
        "FTP-Patator": "Brute Force", "SSH-Patator": "Brute Force",
        # Web attacks
        "Web Attack \x96 Brute Force": "Web Attacks",
        "Web Attack \x96 XSS": "Web Attacks",
        "Web Attack \x96 Sql Injection": "Web Attacks",
        # Infiltration / Bot
        "Infiltration": "Bots", "Bot": "Bots",
    }
    df = df.copy()
    df[config.LABEL_COLUMN] = df["Label"].map(raw_to_group).fillna(df["Label"])
    df = df.drop(columns=["Label"])
    return df


if __name__ == "__main__":
    # Quick smoke test
    loader = DataLoader()
    try:
        df = loader.load_cicids()
        print(f"\nSuccess — shape: {df.shape}")
    except FileNotFoundError as e:
        print(f"No real data available: {e}")
        print("\nGenerating synthetic data for testing...")
        df = generate_synthetic_data(10000)
        print(f"Synthetic shape: {df.shape}")
        df.columns = df.columns.str.strip()
        print(df[config.LABEL_COLUMN].value_counts())
