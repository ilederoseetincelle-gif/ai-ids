# AI-IDS — AI-Powered Intrusion Detection System

A machine learning–based Network Intrusion Detection System (IDS) built as a
Bachelor's final-year project in Network & Security. The system captures
live network traffic, extracts behavioral features from network flows, and
uses a trained XGBoost classifier to distinguish between normal traffic
and 6 categories of cyberattacks — all in real time.

---

## What This Project Does

Unlike traditional signature-based IDS tools (Snort, Suricata) that match
predefined attack rules, this system uses **anomaly detection via machine
learning**. It learns the statistical patterns of normal and malicious traffic
and generalizes to detect variants of known attacks even without an exact
signature match.

```
┌──────────────────────────────────────────────────────────────┐
│                        AI-IDS Pipeline                       │
└──────────────────────────────────────────────────────────────┘

  Network          Capture           Feature           ML Inference
  Traffic    ───►  Layer      ───►   Extraction   ───► (XGBoost +
  (Kali           (Scapy /            (20 flow           Isolation
   attacks)        Zeek)              features)          Forest + SHAP)
                                                             │
                                                             ▼
   ┌────────────────────────────────────────────────────────────┐
   │   Alert Log + Streamlit Dashboard  +  optional IPS layer    │
   │  (real-time visualization · auto-block malicious IPs via    │
   │   iptables when --ips is enabled)                           │
   └────────────────────────────────────────────────────────────┘
```

**Detection capabilities:** 6 attack categories derived from CICIDS2017 —
Port Scanning, Brute Force (SSH/FTP), Web Attacks (XSS, SQLi, BruteForce),
DoS, DDoS, and Bots. Raw CICIDS2017 fine-grained labels are grouped into
these 7 classes (including Normal Traffic) for more robust generalisation.

---

## Quick Start

### 1. Install dependencies

```bash
git clone <your-repo-url> ai-ids
cd ai-ids
python -m venv ids-env
source ids-env/bin/activate    # Windows: ids-env\Scripts\activate
pip install -r requirements.txt
```

### 2. Test the pipeline immediately (no dataset needed)

```bash
python train_pipeline.py --synthetic
python run_tests.py   # or: pytest tests/ -v
```

This runs the entire training pipeline on synthetic data and verifies all
56 unit tests pass. Useful to confirm your environment is correct **before**
downloading the 1.2 GB dataset.

### 3. Download the CICIDS2017 dataset

Download the 8 CSV files from the
[Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
and place them all into `data/`:

```
data/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-WorkingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

### 4. Train the model

```bash
python train_pipeline.py
```

This loads the dataset (~2.7M flows after cleaning), applies preprocessing,
trains **XGBoost** (the default — faster and more accurate than Random Forest),
fits the Isolation Forest anomaly detector on benign traffic, and generates
evaluation plots. Expected runtime: **5–15 minutes** depending on your CPU.

To train Random Forest instead:

```bash
python train_pipeline.py --model rf
```

For faster iteration during development:

```bash
python train_pipeline.py --sample 0.1   # use 10% of the data
```

For final tuning before the defense:

```bash
python train_pipeline.py --tune --compare   # GridSearchCV + model comparison (1–3 hrs)
```

### 5. Run the demo

```bash
python main.py --mode demo --clear-log
```

This runs the replay demonstration with the dashboard at
[http://localhost:8501](http://localhost:8501). It generates a realistic
attack sequence (recon → brute force → web attacks → DoS → DDoS → botnet)
without requiring live VMs.

### 6. Run live detection (requires root)

```bash
sudo python main.py --mode live --interface eth0
```

Replace `eth0` with your network interface name (find it with `ip a` on Linux).

### 7. Run with active prevention (IPS mode)

```bash
sudo python main.py --mode live --interface eth0 --ips           # enforce blocks
sudo python main.py --mode live --interface eth0 --ips-dry-run   # log intended blocks only
```

With `--ips`, the system goes beyond detection: a conservative policy engine
decides which alerts warrant a response, and confirmed malicious IPs are
auto-blocked via `iptables` (in a dedicated `AI_IDS_BLOCK` chain) for a
configurable duration. Whitelisted IPs in `config/whitelist.txt` are **never**
blocked. Use `--ips-dry-run` first to preview actions without touching the
firewall.

---

## Project Structure

```
ai-ids/
├── config.py                  # All paths, hyperparameters, label mappings
├── main.py                    # Detection entry point (live | replay | dashboard | demo)
├── train_pipeline.py          # Training entry point
├── replay_demo.py             # Standalone demo replay (defense-day backup)
├── run_tests.py               # Test runner (works without pytest)
├── requirements.txt           # Pinned dependencies
├── Makefile                   # Convenience commands
│
├── src/
│   ├── preprocessing/
│   │   ├── loader.py          # CICIDS2017 loader, label encoder, splitter
│   │   └── features.py        # Feature selector, scaler, SMOTE handler
│   ├── training/
│   │   ├── train.py           # XGBoost (default), Random Forest, MLP, GridSearchCV
│   │   └── evaluate.py        # Metrics, confusion matrix, ROC curves
│   ├── detection/
│   │   ├── extractor.py       # Live flow extractor (CICIDS-compatible)
│   │   ├── engine.py          # Inference + alert generation
│   │   └── anomaly.py         # Isolation Forest (zero-day / novel traffic)
│   ├── explainability/
│   │   └── shap_explainer.py  # Top-5 SHAP contributions per alert
│   ├── prevention/            # IPS layer (optional, --ips)
│   │   ├── firewall.py        # iptables wrapper (AI_IDS_BLOCK chain)
│   │   ├── policy.py          # Response policy engine (block/log/escalate)
│   │   └── responder.py       # Orchestrates detection → response + auto-unblock
│   └── dashboard/
│       └── app.py             # Streamlit real-time dashboard
│
├── tests/
│   ├── test_preprocessing.py  # 14 tests
│   ├── test_training.py       # 9 tests
│   └── test_detection.py      # 10 tests
│
├── data/                      # CICIDS2017 CSVs (gitignored, you provide)
├── models/                    # Saved model + scaler + plots (auto-generated)
├── logs/                      # Live alert log (JSONL format)
├── notebooks/                 # Jupyter EDA notebooks
└── docs/                      # Architecture diagrams
```

---

## Key Design Choices

### Why Gradient-Boosted Trees and not Deep Learning?

The default model is **XGBoost** (Random Forest is available via `--model rf`).
Three concrete reasons for tree ensembles over deep nets here:

1. **Better on tabular data.** Network flow features are tabular, not images
   or text where deep learning excels. Tree ensembles consistently outperform
   deep nets on this kind of data — and XGBoost edges out Random Forest on
   both accuracy and training speed (via the `hist` method).
2. **Interpretable.** Feature importances plus per-alert **SHAP contributions**
   (`src/explainability/shap_explainer.py`) show exactly which network
   characteristics triggered a detection — critical for explaining decisions
   to a SOC analyst (or a jury).
3. **Fast inference.** Sub-millisecond predictions per flow, no GPU required.
   This is essential for a real-time system processing thousands of flows
   per second.

### Catching Novel Attacks — Anomaly Detection

A supervised classifier can only recognize attack patterns it was trained on.
To flag traffic that looks like *nothing* in the training data, an
**Isolation Forest** (`src/detection/anomaly.py`) is fit on benign-only
traffic and runs alongside the classifier. When the supervised model predicts
BENIGN but the anomaly score is suspicious, the flow is raised as
`Unknown / Anomaly` — a lightweight zero-day safety net.

### How Class Imbalance Is Handled

CICIDS2017 is severely imbalanced — BENIGN traffic dominates (~80%) while
Heartbleed has only ~11 samples total. Two mechanisms:

1. **Class weighting** — Random Forest uses `class_weight="balanced"`; XGBoost
   weights samples inversely proportional to class frequency.
2. **SMOTE** (Synthetic Minority Over-sampling Technique) on the training
   set only, generating synthetic minority samples in feature space
   (`k_neighbors=3`, since Heartbleed has only ~11 samples).

### How False Positives Are Controlled

Three filters on every prediction before an alert is raised:

1. **Class filter:** BENIGN predictions never produce alerts (regardless of
   confidence).
2. **Confidence threshold:** Only predictions with `confidence ≥ 0.70`
   (configurable in `config.py`) become alerts.
3. **Severity grading:** Confidence is split into HIGH (≥0.90),
   MEDIUM (≥0.70), LOW — so SOC analysts can prioritize.

---

## Test Suite

Run all tests:

```bash
pytest tests/ -v          # if pytest installed
python run_tests.py       # standalone runner (no pytest needed)
```

Coverage:

| Module | Tests | Covers |
|---|---|---|
| `test_preprocessing.py` | 14 | Loading, cleaning, label encoding, SMOTE, scaling |
| `test_detection.py` | 15 | Flow extraction, inference, alerting, severity |
| `test_anomaly.py` | 10 | Isolation Forest fit, scoring, novel-traffic flagging |
| `test_training.py` | 9 | XGBoost/RF training, save/load, evaluation, plotting |
| `test_explainability.py` | 8 | SHAP explainer, top-5 contributions, graceful skip |

**All 56 tests pass on synthetic data with no real dataset required.**

---

## Lab Environment for Live Testing

To produce real attack traffic for testing live detection, set up an isolated
3-VM lab in VirtualBox:

```
                  Host-Only Network (192.168.56.0/24)
   ┌──────────────────────────────────────────────────────────┐
   │                                                          │
   │  ┌────────────────┐    ┌────────────────┐    ┌──────────┴─┐
   │  │ Kali Attacker  │    │ Ubuntu Victim  │    │ Monitor VM │
   │  │ 192.168.56.10  │    │ 192.168.56.20  │    │192.168.56.30│
   │  │ nmap, hydra,   │    │ Apache, SSH,   │    │ AI-IDS     │
   │  │ hping3         │    │ FTP            │    │ + Wireshark │
   │  └────────────────┘    └────────────────┘    └────────────┘
   │                                                          │
   └──────────────────────────────────────────────────────────┘
```

**Important:** All 3 VMs must use **Host-Only Networking** — never Bridged
or NAT. This keeps attack traffic isolated from your real network and ISP.
Enable **promiscuous mode** on the Monitor VM's network adapter so it can
see all traffic between Kali and Victim.

### Test attack commands (from Kali)

```bash
# Port scan
nmap -sV -p 1-65535 192.168.56.20

# SSH brute force
hydra -l root -P /usr/share/wordlists/rockyou.txt ssh://192.168.56.20

# DoS
sudo hping3 -S --flood -p 80 192.168.56.20
```

While these run, start the IDS on the Monitor VM:

```bash
sudo python main.py --mode live --interface enp0s8 --clear-log
```

You should see real-time alerts in the terminal and on the dashboard.

---

## Defense Day Checklist

The day before:

- [ ] Run `python train_pipeline.py --tune --compare` overnight for the best results
- [ ] Run `python run_tests.py` — confirm all 56 tests pass
- [ ] Run `python main.py --mode demo --clear-log` and watch the full demo
- [ ] Record a screen capture of the demo as a video backup (`OBS Studio` is free)
- [ ] Print 2 copies of the report
- [ ] Copy everything to a USB stick

The morning of:

- [ ] Arrive 30 minutes early
- [ ] Plug into projector — verify display works
- [ ] Open terminal — run `make demo` to verify the dashboard loads
- [ ] Have your Q&A answer sheet on the table
- [ ] If the live demo fails: switch to the recorded video without hesitation

---

## Limitations

This is a Bachelor's project — be honest about what it is and isn't:

- **CICIDS2017 is from 2017.** Modern attacks (ransomware-as-a-service,
  AI-generated phishing, supply chain attacks) are not in the dataset. The
  model would need retraining on more recent traffic before real production
  use.
- **Lab dataset, not production.** CICIDS2017 was generated in a controlled
  environment. Real enterprise networks have far more variation.
- **No encrypted traffic analysis.** The model uses flow-level features which
  work on encrypted traffic, but content-based features would require
  deep packet inspection.
- **Supervised model + anomaly fallback.** The XGBoost classifier only
  recognizes attack patterns it was trained on. An Isolation Forest anomaly
  detector partially mitigates this by flagging novel benign-looking traffic
  as `Unknown / Anomaly`, but a deep autoencoder trained on richer features
  would push zero-day coverage further (future work).

---

## References

1. Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018). *Toward
   Generating a New Intrusion Detection Dataset and Intrusion Traffic
   Characterization.* ICISSP 2018, pp. 108-116.
2. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.
3. Chawla, N.V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling
   Technique.* JAIR, 16, 321-357.
4. NIST SP 800-94: *Guide to Intrusion Detection and Prevention Systems
   (IDPS).* NIST, 2007.
5. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.*
   JMLR, 12, 2825-2830.

---

## License

Academic use only. This project is submitted as a Bachelor's degree
deliverable in Network & Security.
