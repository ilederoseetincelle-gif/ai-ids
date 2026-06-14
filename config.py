"""
config.py — Central configuration for the AI-IDS project.

All paths, constants, and hyperparameters live here.
No magic numbers anywhere else in the codebase.

Usage:
    import config
    print(config.MODEL_FILE)
"""
from pathlib import Path

# ─── PROJECT PATHS ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
PLOTS_DIR    = MODELS_DIR / "plots"
LOGS_DIR     = PROJECT_ROOT / "logs"
TESTS_DIR    = PROJECT_ROOT / "tests"
SAMPLES_DIR  = TESTS_DIR / "samples"

for _d in (DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR, SAMPLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── DATASET ──────────────────────────────────────────────────────────────────
# The pre-cleaned CSV uses "Attack Type" (7 grouped classes).
# When loading raw CICIDS2017 CSVs, loader.py renames " Label" → "Attack Type"
# after grouping the 15 original classes into these 7 categories.
LABEL_COLUMN = "Attack Type"
BENIGN_LABEL = "Normal Traffic"

# Top 20 most-important features for CICIDS2017.
# Chosen because (a) they consistently rank high in RF feature_importances_
# and (b) they are computable in real time from packet captures.
SELECTED_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Bwd Packets/s",
    "Total Length of Fwd Packets",
    "Min Packet Length",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "PSH Flag Count",
    "FIN Flag Count",
    "ACK Flag Count",
]

# All 15 CICIDS2017 classes with integer encoding.
# Note: "Web Attack" labels in the raw CSVs contain byte 0x96 — kept here.
ATTACK_LABELS = {
    "Normal Traffic": 0,
    "Port Scanning":  1,
    "Web Attacks":    2,
    "Brute Force":    3,
    "DDoS":           4,
    "Bots":           5,
    "DoS":            6,
}
# Reverse mapping: integer -> label string (for display)
LABEL_NAMES = {v: k for k, v in ATTACK_LABELS.items()}

# ─── MODEL FILES ──────────────────────────────────────────────────────────────
MODEL_FILE              = MODELS_DIR / "rf_ids.pkl"          # RF (backward compat)
XGB_MODEL_FILE          = MODELS_DIR / "xgb_ids.pkl"         # XGBoost (preferred)
MODEL_TUNED_FILE        = MODELS_DIR / "rf_ids_tuned.pkl"
SCALER_FILE             = MODELS_DIR / "scaler.pkl"
LABEL_MAP_FILE          = MODELS_DIR / "label_map.pkl"
FEATURES_FILE           = MODELS_DIR / "feature_names.pkl"
FEATURE_IMPORTANCE_FILE = MODELS_DIR / "feature_importance.csv"
ANOMALY_MODEL_FILE      = MODELS_DIR / "anomaly.pkl"

# ─── MODEL HYPERPARAMETERS ────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":      200,
    "max_depth":         30,
    "min_samples_split": 2,
    "random_state":      42,
    "n_jobs":            -1,
    "class_weight":      "balanced",
}

XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         6,
    "learning_rate":     0.1,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "n_jobs":            -1,
    "random_state":      42,
    "tree_method":       "hist",     # fastest CPU training
    "eval_metric":       "mlogloss",
}

# Grid for hyperparameter search
GRID_SEARCH_PARAMS = {
    "n_estimators":      [100, 200],
    "max_depth":         [20, 30, None],
    "min_samples_split": [2, 5],
}

# ─── TRAINING ─────────────────────────────────────────────────────────────────
TEST_SIZE         = 0.20
RANDOM_SEED       = 42
SMOTE_K_NEIGHBORS = 3   # small value — Heartbleed class has ~11 samples

# ─── ANOMALY DETECTION ────────────────────────────────────────────────────────
# IsolationForest is fit on benign-only training flows to detect novel traffic
# the supervised model was never trained on.
ANOMALY_CONTAMINATION    = 0.05   # expected fraction of outliers in benign data
ANOMALY_SCORE_THRESHOLD  = 0.0    # decision_function < threshold → anomaly
                                   # (IF returns negative scores for outliers)

# ─── REAL-TIME DETECTION ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
SEVERITY_HIGH        = 0.90
SEVERITY_MEDIUM      = 0.70

FLOW_TIMEOUT_SEC     = 5     # idle time before a flow is finalized
MAX_FLOW_DURATION    = 120   # max seconds before force-finalizing a long flow

# BPF filter for packet capture — restrict to IPv4 to simplify feature extraction
CAPTURE_FILTER = "ip"

# ─── DASHBOARD ────────────────────────────────────────────────────────────────
DASHBOARD_PORT   = 8501
ALERT_LOG_FILE   = LOGS_DIR / "alerts.jsonl"
REFRESH_INTERVAL = 2   # seconds between dashboard refreshes

# ─── CLI ──────────────────────────────────────────────────────────────────────
SEPARATOR = "=" * 72

if __name__ == "__main__":
    print("AI-IDS Configuration")
    print(SEPARATOR)
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Data dir:      {DATA_DIR}")
    print(f"Models dir:    {MODELS_DIR}")
    print(f"Alert log:     {ALERT_LOG_FILE}")
    print(f"Features:      {len(SELECTED_FEATURES)} selected")
    print(f"Classes:       {len(ATTACK_LABELS)} labels")
    print(SEPARATOR)
