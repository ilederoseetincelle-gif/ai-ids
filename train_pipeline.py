"""
train_pipeline.py — End-to-end training entry point for the AI-IDS.

Usage:
    python train_pipeline.py                       # XGBoost (default)
    python train_pipeline.py --model rf            # Random Forest
    python train_pipeline.py --synthetic           # Synthetic data (no dataset needed)
    python train_pipeline.py --tune                # GridSearchCV RF (slow: 1-3 hrs)
    python train_pipeline.py --compare             # Train XGB + RF + MLP, compare all
    python train_pipeline.py --sample 0.1          # 10% of data (fast iteration)
    python train_pipeline.py --no-anomaly          # Skip anomaly detector training

Pipeline steps:
    1. Load CICIDS2017 CSVs (or generate synthetic data)
    2. Encode string labels to integers
    3. Select the 20 features defined in config
    4. Chronological 80/20 train/test split (random for --synthetic)
    5. Fit StandardScaler on training data, save to disk
    6. Apply SMOTE to balance minority attack classes
    7. Train primary model (XGBoost by default, or RF with --model rf)
    8. Optionally compare XGBoost + RF + MLP (--compare)
    9. Optionally run GridSearchCV (--tune)
   10. Full evaluation: classification report, confusion matrix, ROC curves
   11. Train Isolation Forest anomaly detector on benign-only data
   12. Save all artifacts to models/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import config
from src.preprocessing.loader import DataLoader, generate_synthetic_data
from src.preprocessing.features import FeatureEngineer
from src.training.train import ModelTrainer
from src.training.evaluate import ModelEvaluator
from src.detection.anomaly import AnomalyDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the AI-IDS primary model and anomaly detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["xgb", "rf"], default="xgb",
        help="Primary model to train (default: xgb). "
             "XGBoost is faster and typically more accurate on CICIDS2017.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of CICIDS2017 (for testing).",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run GridSearchCV hyperparameter tuning on RF (slow: 1-3 hours).",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Train XGBoost + RF + MLP and print a side-by-side comparison table.",
    )
    parser.add_argument(
        "--sample", type=float, default=None,
        help="Fraction of data to use for fast iteration (e.g. 0.1 = 10%%).",
    )
    parser.add_argument(
        "--synthetic-rows", type=int, default=30000,
        help="Number of synthetic rows to generate (default: 30000).",
    )
    parser.add_argument(
        "--no-anomaly", action="store_true",
        help="Skip Isolation Forest anomaly detector training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.time()

    print("=" * 72)
    print(f" AI-IDS — Training Pipeline  [primary model: {args.model.upper()}]")
    print("=" * 72)

    # ─── Step 1: Load data ────────────────────────────────────────────────────
    loader = DataLoader()
    if args.synthetic:
        print(f"\n[Step 1/7] Generating {args.synthetic_rows:,} synthetic rows...")
        df = generate_synthetic_data(args.synthetic_rows)
    else:
        print("\n[Step 1/7] Loading CICIDS2017 dataset from data/ ...")
        try:
            df = loader.load_cicids()
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print("\nTIP: Run with --synthetic to test the pipeline without real data.")
            return 1

    if args.sample is not None and 0 < args.sample < 1:
        print(f"\nSampling {args.sample:.0%} of data for faster iteration...")
        df = df.sample(frac=args.sample, random_state=config.RANDOM_SEED).reset_index(drop=True)
        print(f"Sampled shape: {df.shape}")

    # ─── Step 2: Encode labels ────────────────────────────────────────────────
    print("\n[Step 2/7] Encoding labels...")
    df = loader.encode_labels(df)

    # ─── Step 3: Feature engineering ──────────────────────────────────────────
    print("\n[Step 3/7] Selecting features...")
    fe = FeatureEngineer()
    X = fe.select_features(df)
    y = df[config.LABEL_COLUMN].values
    print(f"Feature matrix: {X.shape}")

    # ─── Step 4: Train/test split ─────────────────────────────────────────────
    # Use chronological split for real CICIDS2017 data (row order reflects
    # Mon→Fri capture time). Synthetic data has no temporal ordering so
    # stratified random split is appropriate there.
    use_chronological = not args.synthetic
    split_method = "chronological (80/20 by row order)" if use_chronological else "stratified random"
    print(f"\n[Step 4/7] Train/test split — {split_method}...")
    X_train, X_test, y_train, y_test = loader.split_data(
        X, y, chronological=use_chronological
    )

    # ─── Step 5: Fit scaler, apply SMOTE ──────────────────────────────────────
    print("\n[Step 5/7] Fitting scaler and applying SMOTE...")
    scaler = fe.fit_scaler(X_train.values)
    X_train_s = fe.transform(X_train.values)
    X_test_s  = fe.transform(X_test.values)
    # Balance using undersampling of majority + SMOTE on minorities
    from imblearn.combine import SMOTETomek
    from imblearn.under_sampling import RandomUnderSampler
    import numpy as np

    # Step 1: Undersample Normal Traffic to 300,000
    rus = RandomUnderSampler(
        sampling_strategy={"Normal Traffic": 300000} if False else
        {0: 300000},
        random_state=42
    )
    X_train_bal, y_train_bal = rus.fit_resample(X_train_s, y_train)

    # Step 2: Oversample minority attack classes to 50,000 each
    from collections import Counter
    counts = Counter(y_train_bal)
    target = {cls: max(cnt, 50000) for cls, cnt in counts.items()}
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_bal, y_train_bal)

    # ─── Step 6: Train primary model ──────────────────────────────────────────
    print(f"\n[Step 6/7] Training {args.model.upper()} as primary model...")
    trainer = ModelTrainer()
    models = {}

    if args.tune:
        print("  (--tune: running GridSearchCV on RF, this will take a while)")
        rf = trainer.hyperparameter_search(X_train_bal, y_train_bal, cv=3)
        models["Random Forest (tuned)"] = rf
        primary_model = rf
        primary_name = "Random Forest (tuned)"
    elif args.model == "xgb":
        xgb = trainer.train_xgboost(X_train_bal, y_train_bal, save=True)
        if xgb is None:
            print("XGBoost unavailable — falling back to Random Forest.")
            rf = trainer.train_random_forest(X_train_bal, y_train_bal, save=True)
            primary_model = rf
            primary_name = "Random Forest"
        else:
            primary_model = xgb
            primary_name = "XGBoost"
            models["XGBoost"] = xgb
    else:
        rf = trainer.train_random_forest(X_train_bal, y_train_bal, save=True)
        primary_model = rf
        primary_name = "Random Forest"
        models["Random Forest"] = rf

    if args.compare and not args.tune:
        print("\n  (--compare: training all models for comparison)")
        if "XGBoost" not in models:
            xgb = trainer.train_xgboost(X_train_bal, y_train_bal, save=False)
            if xgb is not None:
                models["XGBoost"] = xgb
        if "Random Forest" not in models:
            rf = trainer.train_random_forest(X_train_bal, y_train_bal, save=False)
            models["Random Forest"] = rf
        mlp = trainer.train_mlp(X_train_bal, y_train_bal)
        models["MLP"] = mlp

    # ─── Step 7: Evaluate and plot ────────────────────────────────────────────
    print("\n[Step 7/7] Evaluating models and generating plots...")
    evaluator = ModelEvaluator()

    if hasattr(primary_model, "feature_importances_"):
        importance_df = fe.get_feature_importance(primary_model, config.SELECTED_FEATURES)
        print("\nTop 10 features by importance:")
        print(importance_df.head(10).to_string(index=False))

    if len(models) > 1:
        _ = evaluator.compare_models(models, X_test_s, y_test)
    else:
        _ = evaluator.full_report(primary_model, X_test_s, y_test, primary_name)

    evaluator.plot_confusion_matrix(
        primary_model, X_test_s, y_test,
        save_path=config.PLOTS_DIR / f"confusion_matrix_{args.model}.png",
        model_name=primary_name,
    )
    evaluator.plot_roc_curves(
        primary_model, X_test_s, y_test,
        save_path=config.PLOTS_DIR / f"roc_curves_{args.model}.png",
        model_name=primary_name,
    )

    # ─── Anomaly detector ─────────────────────────────────────────────────────
    if not args.no_anomaly:
        print("\n[Bonus] Training Isolation Forest anomaly detector on benign data...")
        benign_id = config.ATTACK_LABELS[config.BENIGN_LABEL]
        # Use the pre-SMOTE scaled training data so the IF sees natural class ratios
        benign_mask = y_train == benign_id
        X_benign = X_train_s[benign_mask]
        print(f"  Benign training samples: {len(X_benign):,}")

        anomaly_detector = AnomalyDetector()
        anomaly_detector.fit(X_benign)
        anomaly_detector.save(config.ANOMALY_MODEL_FILE)
    else:
        print("\n[Bonus] Skipping anomaly detector training (--no-anomaly).")

    # ─── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print(" TRAINING COMPLETE")
    print("=" * 72)
    print(f"Total time:          {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Primary model:       {primary_name}")
    if args.model == "xgb" and config.XGB_MODEL_FILE.exists():
        print(f"Model saved to:      {config.XGB_MODEL_FILE}")
    else:
        print(f"Model saved to:      {config.MODEL_FILE}")
    print(f"Scaler saved to:     {config.SCALER_FILE}")
    if not args.no_anomaly and config.ANOMALY_MODEL_FILE.exists():
        print(f"Anomaly model:       {config.ANOMALY_MODEL_FILE}")
    print(f"Plots saved to:      {config.PLOTS_DIR}")
    print("\nNext step: run live detection")
    print("  python main.py --mode live --interface <your_iface>")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
