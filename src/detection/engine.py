"""
engine.py — Real-time detection and alerting engine.

The DetectionEngine loads the primary trained model (XGBoost preferred,
RF fallback), the StandardScaler, and optionally:
  - AnomalyDetector (Isolation Forest) for zero-day traffic detection
  - SHAPExplainer for per-alert feature attribution

Alert pipeline per flow:
  1. Build scaled feature vector
  2. Supervised model: predict class + confidence
  3. If BENIGN AND anomaly detector available → anomaly check
  4. Apply filters: benign-class filter → confidence threshold
  5. If alert raised: compute SHAP contributions (if SHAP available)
  6. Assign severity, log to JSONL, print to console
  7. If IPS enabled: forward alert to Responder (block / log / escalate)

Alerts are persisted as JSON lines in config.ALERT_LOG_FILE so the
dashboard can read them asynchronously.
"""
from __future__ import annotations

import json
import threading
import time
import datetime as _dt
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

import config
from src.detection.extractor import FlowExtractor
from src.prevention import Firewall, Responder, ResponsePolicy

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _COLOR_MAP = {
        "HIGH":    Fore.RED,
        "MEDIUM":  Fore.YELLOW,
        "LOW":     Fore.CYAN,
        "ANOMALY": Fore.MAGENTA,
    }
    _RESET = Style.RESET_ALL
except ImportError:
    _COLOR_MAP = {"HIGH": "", "MEDIUM": "", "LOW": "", "ANOMALY": ""}
    _RESET = ""

ANOMALY_ATTACK_TYPE = "Unknown / Anomaly"
ANOMALY_CONFIDENCE  = 0.0   # no probability from unsupervised model


class DetectionEngine:
    """Runs real-time ML-based intrusion detection on flow features."""

    def __init__(
        self,
        model_path: Path = None,
        scaler_path: Path = None,
        alert_log: Path = None,
        anomaly_path: Path = None,
        enable_shap: bool = True,
        enable_ips: bool = False,
        ips_dry_run: bool = False,
    ):
        """
        Args:
            model_path:   Path to primary model. If None, prefers XGB then RF.
            scaler_path:  Path to scaler. Defaults to config.SCALER_FILE.
            alert_log:    Path to JSONL alert log. Defaults to config.ALERT_LOG_FILE.
            anomaly_path: Path to anomaly model. If None uses config.ANOMALY_MODEL_FILE.
                          Missing file is silently ignored (anomaly check disabled).
            enable_shap:  If True (default), load SHAP explainer when shap is installed.
            enable_ips:   If True, enable the IPS responder (block malicious IPs).
            ips_dry_run:  If True, log intended blocks but do not modify iptables.
        """
        if scaler_path is None:
            scaler_path = config.SCALER_FILE
        if alert_log is None:
            alert_log = config.ALERT_LOG_FILE
        if anomaly_path is None:
            anomaly_path = config.ANOMALY_MODEL_FILE

        scaler_path = Path(scaler_path)
        alert_log   = Path(alert_log)

        # ── Primary supervised model ──────────────────────────────────────────
        if model_path is not None:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}.\n"
                    f"Run: python train_pipeline.py"
                )
            print(f"Loading model from {model_path}...")
            self.model = joblib.load(model_path)
        else:
            # Auto-select: prefer XGBoost if available
            if config.XGB_MODEL_FILE.exists():
                print(f"Loading XGBoost model from {config.XGB_MODEL_FILE}...")
                self.model = joblib.load(config.XGB_MODEL_FILE)
            elif config.MODEL_FILE.exists():
                print(f"Loading Random Forest model from {config.MODEL_FILE}...")
                self.model = joblib.load(config.MODEL_FILE)
            else:
                raise FileNotFoundError(
                    f"No trained model found. "
                    f"Expected {config.XGB_MODEL_FILE} or {config.MODEL_FILE}.\n"
                    f"Run: python train_pipeline.py"
                )

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}.\n"
                f"Run: python train_pipeline.py"
            )
        print(f"Loading scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)

        self.alert_log = alert_log
        self.alert_log.parent.mkdir(parents=True, exist_ok=True)

        self.extractor  = FlowExtractor(on_flow_complete=self._on_flow)
        self.benign_id  = config.ATTACK_LABELS[config.BENIGN_LABEL]

        # ── Anomaly detector (optional) ───────────────────────────────────────
        self._anomaly: Optional[object] = None
        if Path(anomaly_path).exists():
            try:
                from src.detection.anomaly import AnomalyDetector
                self._anomaly = AnomalyDetector.load(anomaly_path)
                print(f"Anomaly detector loaded from {anomaly_path}")
            except Exception as e:
                print(f"[warn] Could not load anomaly detector: {e}")
        else:
            print(f"[info] No anomaly model at {anomaly_path} — anomaly detection disabled.")

        # ── SHAP explainer (optional) ─────────────────────────────────────────
        self._shap: Optional[object] = None
        if enable_shap:
            try:
                from src.explainability.shap_explainer import SHAPExplainer
                self._shap = SHAPExplainer(self.model, config.SELECTED_FEATURES)
                print("SHAP explainer loaded.")
            except ImportError:
                print("[info] shap not installed — SHAP explanations disabled. "
                      "Run: pip install shap>=0.47")
            except Exception as e:
                print(f"[warn] Could not load SHAP explainer: {e}")

        # ── IPS responder (optional) ──────────────────────────────────────────
        self._responder: Optional[object] = None
        if enable_ips:
            fw = Firewall(dry_run=ips_dry_run, audit_log=config.IPS_AUDIT_LOG)
            policy = ResponsePolicy()
            n_wl = policy.load_whitelist(str(config.IPS_WHITELIST))
            self._responder = Responder(firewall=fw, policy=policy)
            mode = "DRY-RUN" if ips_dry_run else "ACTIVE"
            print(f"[IPS] Enabled — mode={mode}, whitelist={n_wl} IPs")

        # Statistics
        self.flows_seen    = 0
        self.alerts_raised = 0
        self._lock         = threading.Lock()
        self._running      = False

    # ── Internal callback ─────────────────────────────────────────────────────

    def _on_flow(self, features: dict) -> None:
        """Callback invoked by FlowExtractor for each completed flow."""
        with self._lock:
            self.flows_seen += 1
        try:
            alert = self.predict(features)
        except Exception as exc:
            src = features.get("_src_ip", "?")
            dst = features.get("_dst_ip", "?")
            print(f"[warn] predict() failed for flow {src} → {dst}: {exc}")
            return
        if alert is not None:
            self.log_alert(alert)

    # ── Core inference ────────────────────────────────────────────────────────

    def predict(self, features: dict) -> Optional[dict]:
        """
        Run the full detection pipeline on a single flow's features.

        Returns an alert dict if an attack is detected, otherwise None.
        """
        vec = np.array([[features[f] for f in config.SELECTED_FEATURES]])
        vec_scaled = np.clip(self.scaler.transform(vec), -5.0, 5.0)

        proba        = self.model.predict_proba(vec_scaled)[0]
        pred_idx     = int(np.argmax(proba))
        confidence   = float(proba[pred_idx])
        pred_class   = int(self.model.classes_[pred_idx])

        # ── Anomaly check for flows the supervised model calls BENIGN ─────────
        if pred_class == self.benign_id and self._anomaly is not None:
            is_anom, anom_score = self._anomaly.predict(vec_scaled)
            if is_anom:
                return self._build_alert(
                    features      = features,
                    attack_name   = ANOMALY_ATTACK_TYPE,
                    confidence    = ANOMALY_CONFIDENCE,
                    severity      = "LOW",
                    vec_scaled    = vec_scaled,
                    pred_idx      = pred_idx,
                    extra         = {"anomaly_score": round(float(anom_score), 4)},
                )

        # ── Supervised attack filters ─────────────────────────────────────────
        if pred_class == self.benign_id:
            return None
        if confidence < config.CONFIDENCE_THRESHOLD:
            return None

        severity = (
            "HIGH"   if confidence >= config.SEVERITY_HIGH   else
            "MEDIUM" if confidence >= config.SEVERITY_MEDIUM else
            "LOW"
        )
        attack_name = config.LABEL_NAMES.get(pred_class, f"CLS_{pred_class}")

        return self._build_alert(
            features   = features,
            attack_name= attack_name,
            confidence = confidence,
            severity   = severity,
            vec_scaled = vec_scaled,
            pred_idx   = pred_idx,
        )

    def _build_alert(
        self,
        features:    dict,
        attack_name: str,
        confidence:  float,
        severity:    str,
        vec_scaled:  np.ndarray,
        pred_idx:    int,
        extra:       dict | None = None,
    ) -> dict:
        """Construct the alert dict, adding SHAP contributions if available."""
        alert: dict = {
            "timestamp":   _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "src_ip":      features.get("_src_ip",   "unknown"),
            "dst_ip":      features.get("_dst_ip",   "unknown"),
            "src_port":    features.get("_src_port",  0),
            "dst_port":    features.get("_dst_port",  0),
            "protocol":    features.get("_protocol",  0),
            "attack_type": attack_name,
            "confidence":  round(confidence, 4),
            "severity":    severity,
        }

        if extra:
            alert.update(extra)

        # SHAP contributions (replaces z-score top_features when available)
        if self._shap is not None:
            shap_contribs = self._shap.explain(vec_scaled, pred_idx)
            alert["shap_contributions"] = shap_contribs
        else:
            alert["top_features"] = self._top_feature_values(features, vec_scaled, 3)

        return alert

    def _top_feature_values(self, features: dict, vec_scaled: np.ndarray, n: int = 3) -> dict:
        """Fallback: return the n most extreme feature values by z-score.

        Uses the already-scaled vector from predict() to avoid a second
        scaler.transform() call.
        """
        abs_z = np.abs(vec_scaled[0])
        top   = np.argsort(abs_z)[-n:][::-1]
        vec_raw = np.array([features[f] for f in config.SELECTED_FEATURES])
        return {config.SELECTED_FEATURES[i]: round(float(vec_raw[i]), 2) for i in top}

    # ── Alert logging ─────────────────────────────────────────────────────────

    def log_alert(self, alert: dict) -> None:
        """Append an alert to the JSONL log, print a summary, and trigger IPS."""
        if self._responder is not None:
            decision = self._responder.handle_alert(alert)
            alert["ips_action"] = decision.action
            alert["ips_reason"] = decision.reason

        with self._lock:
            with self.alert_log.open("a") as f:
                f.write(json.dumps(alert) + "\n")
            self.alerts_raised += 1

        color = _COLOR_MAP.get(
            "ANOMALY" if alert["attack_type"] == ANOMALY_ATTACK_TYPE
            else alert["severity"], ""
        )
        conf_str = (
            f"score={alert.get('anomaly_score', '')}"
            if alert["attack_type"] == ANOMALY_ATTACK_TYPE
            else f"confidence={alert['confidence']:.2%}"
        )
        print(
            f"{color}[ALERT #{self.alerts_raised}] "
            f"{alert['timestamp']}  "
            f"{alert['severity']:<6}  "
            f"{alert['attack_type']:<32}  "
            f"{alert['src_ip']} -> {alert['dst_ip']}:{alert['dst_port']}  "
            f"{conf_str}{_RESET}"
        )

    # ── Live / replay ─────────────────────────────────────────────────────────

    def start_live(self, interface: str, bpf_filter: str = None) -> None:
        """Start live packet capture on the given interface (requires root)."""
        if bpf_filter is None:
            bpf_filter = config.CAPTURE_FILTER

        try:
            from scapy.all import sniff
        except ImportError:
            raise RuntimeError("scapy is not installed. Run: pip install scapy")

        model_type = type(self.model).__name__
        shap_status = "enabled" if self._shap else "disabled"
        anom_status = "enabled" if self._anomaly else "disabled"

        print(f"\n{'='*60}")
        print(f"AI-IDS Detection Engine — LIVE MODE")
        print(f"{'='*60}")
        print(f"Interface:      {interface}")
        print(f"Filter:         {bpf_filter}")
        print(f"Model:          {model_type}")
        print(f"Threshold:      {config.CONFIDENCE_THRESHOLD}")
        print(f"SHAP:           {shap_status}")
        print(f"Anomaly det.:   {anom_status}")
        print(f"Alert log:      {self.alert_log}")
        print(f"{'='*60}\n")
        print("Capture started. Press Ctrl+C to stop.\n")

        self._running = True
        flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        flush_thread.start()

        try:
            sniff(
                iface=interface,
                filter=bpf_filter,
                prn=self.extractor.process_packet,
                store=False,
                stop_filter=lambda _: not self._running,
            )
        except PermissionError:
            print("ERROR: Live capture requires root. Try: sudo python main.py ...")
            self._running = False
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            print("\nFlushing remaining flows...")
            self.extractor.flush_all()
            self._print_stats()

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(1)
            self.extractor.flush_idle_flows()

    def replay_pcap(self, pcap_file: Path) -> None:
        """Process a saved .pcap file as if it were live traffic."""
        try:
            from scapy.all import rdpcap
        except ImportError:
            raise RuntimeError("scapy is not installed. Run: pip install scapy")

        pcap_file = Path(pcap_file)
        if not pcap_file.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_file}")

        print(f"\nReplaying {pcap_file}...")
        packets = rdpcap(str(pcap_file))
        print(f"Loaded {len(packets)} packets. Processing...")

        for pkt in packets:
            self.extractor.process_packet(pkt)

        count = self.extractor.flush_all()
        print(f"Finalized {count} flows from replay.")
        self._print_stats()

    def shutdown(self) -> None:
        """Stop the IPS unblock daemon cleanly. Call before program exit."""
        self._running = False
        if self._responder is not None:
            self._responder.shutdown()

    def _print_stats(self) -> None:
        print(f"\n{'='*60}")
        print("Detection Summary")
        print(f"{'='*60}")
        print(f"Flows processed:  {self.flows_seen}")
        print(f"Alerts raised:    {self.alerts_raised}")
        if self.flows_seen > 0:
            rate = 100 * self.alerts_raised / self.flows_seen
            print(f"Alert rate:       {rate:.2f}%")
        print(f"{'='*60}\n")
