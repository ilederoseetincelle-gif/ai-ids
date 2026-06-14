# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv ids-env && source ids-env/bin/activate
pip install -r requirements.txt

# Verify environment without real data (all 33+ tests use synthetic data)
python train_pipeline.py --synthetic
pytest tests/ -v          # or: python run_tests.py

# Train (XGBoost default — faster and more accurate than RF)
python train_pipeline.py

# Train RF instead
python train_pipeline.py --model rf

# Fast iteration (10% sample)
python train_pipeline.py --sample 0.1

# Hyperparameter tuning + model comparison (1-3 hours)
python train_pipeline.py --tune --compare

# Demo: replay attack sequence + Streamlit dashboard (http://localhost:8501)
python main.py --mode demo --clear-log

# Live detection (root required)
sudo python main.py --mode live --interface eth0

# Dashboard only (reads existing logs/alerts.jsonl)
python main.py --mode dashboard

# Common make aliases
make train            # full XGBoost training
make train-rf         # RF training
make train-fast       # 10% sample
make train-synthetic  # synthetic data only
make demo             # demo mode
make test             # pytest
make clean            # remove models/logs/plots
```

## Architecture

End-to-end ML intrusion detection system trained on CICIDS2017, deployed as a real-time network monitor.

**Configuration** ([config.py](config.py)): Single source of truth for all paths, hyperparameters, class mappings, and thresholds. Import `config` everywhere — no magic numbers.

**Data pipeline** ([src/preprocessing/](src/preprocessing/)):
- `loader.py` — `DataLoader` loads 8 raw CICIDS2017 CSVs or the pre-cleaned `cicids2017_cleaned.csv`. Handles known data quality issues: leading spaces in column names, `Inf` in flow-rate columns, Latin-1 encoded Web Attack labels. `generate_synthetic_data()` mimics CICIDS2017 structure for testing without the real dataset. `split_data(..., chronological=True)` preserves Mon→Fri row order to avoid future-flow leakage (default for real data; synthetic uses stratified random).
- `features.py` — `FeatureEngineer` selects 20 flow features (listed in `config.SELECTED_FEATURES`), fits a `StandardScaler` on training data only (saved to `models/scaler.pkl`), clips outliers to ±5σ, and applies SMOTE with `k_neighbors=3` (small because Heartbleed class has ~11 samples).

**Training** ([src/training/](src/training/)):
- `train.py` — `ModelTrainer` trains XGBoost (default, fast with `hist` method), Random Forest, MLP, and GridSearchCV. XGBoost is saved to `models/xgb_ids.pkl`; RF to `models/rf_ids.pkl`.
- `evaluate.py` — `ModelEvaluator` computes accuracy, macro F1, weighted F1, false positive rate on BENIGN, and per-class classification report. Saves confusion matrix and ROC curve PNGs to `models/plots/`.

**Anomaly detection** ([src/detection/anomaly.py](src/detection/anomaly.py)): `AnomalyDetector` wraps `IsolationForest`, fit on benign-only scaled training data. Used alongside the supervised classifier to flag novel zero-day-style traffic the RF/XGB was not trained on. Saved to `models/anomaly.pkl`.

**Live detection** ([src/detection/](src/detection/)):
- `extractor.py` — `FlowExtractor` groups Scapy packets into bidirectional flows by 5-tuple, computes 20 CICIDS2017-compatible features per flow after a timeout, calls `on_flow_complete` callback. Runs flush loop in a background thread.
- `engine.py` — `DetectionEngine` loads the primary model (prefers XGBoost if `xgb_ids.pkl` exists, falls back to `rf_ids.pkl`), scaler, and optional anomaly detector. Per-flow `predict()` applies three filters before raising an alert: (1) BENIGN prediction is dropped, (2) confidence must be ≥ `CONFIDENCE_THRESHOLD` (0.70), (3) anomaly check runs in parallel for BENIGN predictions with a score below `ANOMALY_SCORE_THRESHOLD`. Alerts include `shap_contributions` (top 5 SHAP feature values) when SHAP is available.

**SHAP explainability** ([src/explainability/shap_explainer.py](src/explainability/shap_explainer.py)): `SHAPExplainer` wraps `shap.TreeExplainer` and returns the top-5 SHAP contributions per alert. Loaded lazily — silently skipped if `shap` is not installed.

**Dashboard** ([src/dashboard/app.py](src/dashboard/app.py)): Streamlit app with 2-second auto-refresh. Reads `logs/alerts.jsonl` (JSONL format, one alert per line). KPI row → alert timeline (last 30 min) → attack breakdown + severity pie → scrolling alert table. Anomaly alerts are styled distinctly. Dashboard is launched as a subprocess by `main.py --mode demo`.

**Alert format** (`logs/alerts.jsonl`): Each line is a JSON object with: `timestamp`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `attack_type`, `confidence`, `severity` (`HIGH`/`MEDIUM`/`LOW`), `shap_contributions` (dict), and for anomalies: `"attack_type": "Unknown / Anomaly"`.

## Key design constraints

- **No horizontal flip in augmentation** — not applicable here, but the SMOTE `k_neighbors=3` is load-bearing: Heartbleed has ~11 samples, so `k_neighbors` must be `< 11/2`.
- **Scaler fit on train only** — `FeatureEngineer.fit_scaler()` is called before SMOTE; the scaler is never fit on test or SMOTE-synthetic data.
- **Alert filtering order** matters in `engine.py`: benign filter → confidence threshold → severity grading. Changing this order changes recall/precision.
- **SHAP is optional** — if `shap` is not installed, the engine skips SHAP and falls back to the z-score `top_features` field.
- **Anomaly detector is optional** — engine checks for `models/anomaly.pkl` at startup; missing file is not an error.
- **All tests run on synthetic data** — `pytest tests/ -v` requires no real dataset. The `mock_model_and_scaler` fixture in `test_detection.py` is the pattern for engine tests.
