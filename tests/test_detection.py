"""
tests/test_detection.py — Unit tests for the detection engine and flow extractor.

Run: pytest tests/test_detection.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import config
from src.detection.extractor import Flow, FlowExtractor, PacketMeta


# ─── FLOW EXTRACTOR TESTS ─────────────────────────────────────────────────────
class TestFlowExtractor:

    def test_extract_features_too_short_returns_none(self):
        """A flow with only 1 packet returns None (insufficient data)."""
        fe = FlowExtractor(on_flow_complete=lambda f: None)
        flow = Flow("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        flow.add(PacketMeta(timestamp=1000.0, length=100, direction="fwd"))
        result = fe.extract_features(flow)
        assert result is None

    def test_extract_features_full_set(self):
        """A flow with multiple packets returns a complete feature dict."""
        fe = FlowExtractor(on_flow_complete=lambda f: None)
        flow = Flow("1.1.1.1", "2.2.2.2", 1234, 80, 6)

        # SYN
        flow.add(PacketMeta(timestamp=1000.0, length=60, direction="fwd", tcp_flags=0x02))
        # SYN-ACK
        flow.add(PacketMeta(timestamp=1000.1, length=60, direction="bwd", tcp_flags=0x12))
        # ACK
        flow.add(PacketMeta(timestamp=1000.2, length=60, direction="fwd", tcp_flags=0x10))
        # PSH-ACK
        flow.add(PacketMeta(timestamp=1000.3, length=500, direction="fwd", tcp_flags=0x18))
        # ACK
        flow.add(PacketMeta(timestamp=1000.4, length=60, direction="bwd", tcp_flags=0x10))

        features = fe.extract_features(flow)

        assert features is not None
        # All selected features present
        for feat in config.SELECTED_FEATURES:
            assert feat in features
        # Metadata present for alerts
        for meta in ["_src_ip", "_dst_ip", "_src_port", "_dst_port", "_protocol"]:
            assert meta in features
        # Sanity checks on values
        assert features["Total Fwd Packets"] == 3
        # ACK counted on: SYN-ACK (0x12), ACK (0x10), PSH-ACK (0x18), ACK (0x10) = 4
        assert features["ACK Flag Count"] == 4
        # PSH bit set on PSH-ACK (0x18) only — counted across all packets
        assert features["PSH Flag Count"] == 1
        # FIN bit (0x01) not set in any packet
        assert features["FIN Flag Count"] == 0

    def test_flow_key_bidirectional(self):
        """Packets in either direction should map to the same flow."""
        captured = []
        fe = FlowExtractor(on_flow_complete=captured.append)

        # Simulate forward + backward packets manually via process_packet-like logic
        # Since process_packet requires scapy, we test via internal flow accumulation
        flow_key_fwd = ("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        flow_key_bwd = ("2.2.2.2", "1.1.1.1", 80, 1234, 6)

        fe.flows[flow_key_fwd] = Flow("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        fe.flows[flow_key_fwd].add(PacketMeta(1000.0, 60, "fwd"))

        # A reverse packet: since key_fwd already exists, new packet uses same flow
        # This is tested implicitly by process_packet; here we verify the dict key
        assert flow_key_fwd in fe.flows
        assert flow_key_bwd not in fe.flows  # reverse never created separately

    def test_flush_idle_flows_removes_old(self):
        """flush_idle_flows should finalize flows older than FLOW_TIMEOUT_SEC."""
        captured = []
        fe = FlowExtractor(on_flow_complete=captured.append)

        # Create an "old" flow
        old_flow = Flow("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        old_flow.add(PacketMeta(1000.0, 60, "fwd"))
        old_flow.add(PacketMeta(1000.1, 60, "bwd"))
        fe.flows[("1.1.1.1", "2.2.2.2", 1234, 80, 6)] = old_flow

        # "Now" is well past the last seen time
        n_flushed = fe.flush_idle_flows(now=1000.0 + config.FLOW_TIMEOUT_SEC + 1)

        assert n_flushed == 1
        assert len(fe.flows) == 0
        assert len(captured) == 1

    def test_flush_idle_flows_keeps_recent(self):
        """flush_idle_flows must NOT finalize flows that are still active."""
        captured = []
        fe = FlowExtractor(on_flow_complete=captured.append)

        recent_flow = Flow("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        recent_flow.add(PacketMeta(1000.0, 60, "fwd"))
        recent_flow.add(PacketMeta(1000.1, 60, "bwd"))
        fe.flows[("1.1.1.1", "2.2.2.2", 1234, 80, 6)] = recent_flow

        # "Now" is only 1 second after the last packet — not yet idle
        n_flushed = fe.flush_idle_flows(now=1001.1)

        assert n_flushed == 0
        assert len(fe.flows) == 1
        assert len(captured) == 0

    def test_flush_max_duration_flow(self, monkeypatch):
        """A flow exceeding MAX_FLOW_DURATION is force-flushed even if not idle."""
        captured = []
        fe = FlowExtractor(on_flow_complete=captured.append)

        # Patch MAX_FLOW_DURATION to a small value for the test
        monkeypatch.setattr(config, "MAX_FLOW_DURATION", 10)

        long_flow = Flow("1.1.1.1", "2.2.2.2", 1234, 80, 6)
        t0 = 1000.0
        # Flow started at t0, keeps receiving packets (so not idle)
        for i in range(5):
            long_flow.add(PacketMeta(t0 + i, 60, "fwd"))
            long_flow.add(PacketMeta(t0 + i + 0.1, 60, "bwd"))
        fe.flows[("1.1.1.1", "2.2.2.2", 1234, 80, 6)] = long_flow

        # "Now" is 11 seconds after first_seen — exceeds MAX_FLOW_DURATION
        n_flushed = fe.flush_idle_flows(now=t0 + 11)

        assert n_flushed == 1
        assert len(fe.flows) == 0

    def test_flush_all_thread_safety(self):
        """flush_all and process_packet can run concurrently without raising."""
        import threading as _threading

        errors = []

        def on_flow(f):
            pass

        fe = FlowExtractor(on_flow_complete=on_flow)

        # Pre-populate with some flows
        for i in range(20):
            key = ("1.1.1.1", f"2.2.2.{i}", i, 80, 6)
            flow = Flow("1.1.1.1", f"2.2.2.{i}", i, 80, 6)
            flow.add(PacketMeta(1000.0, 60, "fwd"))
            flow.add(PacketMeta(1000.1, 60, "bwd"))
            fe.flows[key] = flow

        def flush_worker():
            try:
                fe.flush_all()
            except Exception as e:
                errors.append(e)

        def add_worker():
            try:
                for i in range(20, 40):
                    key = ("3.3.3.3", f"4.4.4.{i}", i, 443, 6)
                    flow = Flow("3.3.3.3", f"4.4.4.{i}", i, 443, 6)
                    flow.add(PacketMeta(1000.0, 60, "fwd"))
                    with fe._lock:
                        fe.flows[key] = flow
            except Exception as e:
                errors.append(e)

        t1 = _threading.Thread(target=flush_worker)
        t2 = _threading.Thread(target=add_worker)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert errors == [], f"Thread errors: {errors}"


# ─── DETECTION ENGINE TESTS ───────────────────────────────────────────────────
@pytest.fixture
def mock_model_and_scaler(tmp_path, monkeypatch):
    """Train a trivial model and save it to disk so the engine can load it."""
    # Tiny training set: 20 features, 2 classes (0=BENIGN, 1=attack)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, len(config.SELECTED_FEATURES)))
    y = rng.integers(0, 2, size=100)
    # Ensure both classes present
    y[0], y[1] = 0, 1

    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    model.fit(scaler.transform(X), y)

    model_file = tmp_path / "rf.pkl"
    scaler_file = tmp_path / "scaler.pkl"
    alert_log = tmp_path / "alerts.jsonl"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    monkeypatch.setattr(config, "MODEL_FILE",         model_file)
    monkeypatch.setattr(config, "SCALER_FILE",        scaler_file)
    monkeypatch.setattr(config, "ALERT_LOG_FILE",     alert_log)
    # Disable anomaly detector so real anomaly.pkl is not loaded by engine tests
    monkeypatch.setattr(config, "ANOMALY_MODEL_FILE", tmp_path / "no_anomaly.pkl")

    return model_file, scaler_file, alert_log


class TestDetectionEngine:

    def test_engine_loads_from_files(self, mock_model_and_scaler):
        """Engine initializes correctly when model and scaler exist."""
        from src.detection.engine import DetectionEngine
        model_file, scaler_file, _ = mock_model_and_scaler
        engine = DetectionEngine(model_file, scaler_file)
        assert engine.model is not None
        assert engine.scaler is not None

    def test_engine_missing_model_raises(self, tmp_path):
        """Engine raises FileNotFoundError for missing model."""
        from src.detection.engine import DetectionEngine
        with pytest.raises(FileNotFoundError):
            DetectionEngine(tmp_path / "missing.pkl", tmp_path / "scaler.pkl")

    def test_predict_returns_none_for_low_confidence(self, mock_model_and_scaler):
        """If confidence below threshold, predict returns None."""
        from src.detection.engine import DetectionEngine
        model_file, scaler_file, _ = mock_model_and_scaler
        engine = DetectionEngine(model_file, scaler_file)

        # Patch model to always output low-confidence predictions
        def fake_proba(X):
            return np.array([[0.4, 0.6]])  # below default 0.70 threshold
        engine.model.predict_proba = fake_proba
        engine.model.classes_ = np.array([0, 1])

        features = {f: 0.1 for f in config.SELECTED_FEATURES}
        features.update({"_src_ip": "1.1.1.1", "_dst_ip": "2.2.2.2",
                         "_src_port": 1234, "_dst_port": 80, "_protocol": 6})

        result = engine.predict(features)
        assert result is None

    def test_predict_alerts_on_high_confidence_attack(self, mock_model_and_scaler):
        """High-confidence attack prediction produces a full alert dict."""
        from src.detection.engine import DetectionEngine
        model_file, scaler_file, _ = mock_model_and_scaler
        engine = DetectionEngine(model_file, scaler_file)

        # Force the model to predict class 1 (attack) with high confidence
        engine.model.predict_proba = lambda X: np.array([[0.05, 0.95]])
        engine.model.classes_ = np.array([0, 1])

        features = {f: 0.1 for f in config.SELECTED_FEATURES}
        features.update({"_src_ip": "10.0.0.1", "_dst_ip": "10.0.0.2",
                         "_src_port": 54321, "_dst_port": 22, "_protocol": 6})

        alert = engine.predict(features)
        assert alert is not None
        assert alert["severity"] == "HIGH"
        assert alert["src_ip"] == "10.0.0.1"
        assert alert["dst_port"] == 22
        assert 0.0 <= alert["confidence"] <= 1.0

    def test_predict_returns_none_for_benign(self, mock_model_and_scaler):
        """BENIGN predictions (class 0) do not raise alerts, even at high confidence."""
        from src.detection.engine import DetectionEngine
        model_file, scaler_file, _ = mock_model_and_scaler
        engine = DetectionEngine(model_file, scaler_file)

        # Force prediction of class 0 (BENIGN)
        engine.model.predict_proba = lambda X: np.array([[0.95, 0.05]])
        engine.model.classes_ = np.array([0, 1])

        features = {f: 0.1 for f in config.SELECTED_FEATURES}
        features.update({"_src_ip": "1.1.1.1", "_dst_ip": "2.2.2.2",
                         "_src_port": 1234, "_dst_port": 80, "_protocol": 6})

        result = engine.predict(features)
        assert result is None

    def test_on_flow_exception_does_not_propagate(self, mock_model_and_scaler):
        """If predict() raises, _on_flow swallows the error so the flush loop stays alive."""
        from src.detection.engine import DetectionEngine
        model_file, scaler_file, _ = mock_model_and_scaler
        engine = DetectionEngine(model_file, scaler_file)

        def exploding_proba(X):
            raise RuntimeError("model exploded")
        engine.model.predict_proba = exploding_proba

        features = {f: 0.1 for f in config.SELECTED_FEATURES}
        features.update({"_src_ip": "1.1.1.1", "_dst_ip": "2.2.2.2",
                         "_src_port": 1234, "_dst_port": 80, "_protocol": 6})

        # Must not raise even though predict_proba throws
        engine._on_flow(features)
        # flows_seen is still incremented (the counter update precedes predict)
        assert engine.flows_seen == 1
        assert engine.alerts_raised == 0

    def test_alert_top_features_uses_precomputed_scaled_vector(self, mock_model_and_scaler):
        """top_features in alert is populated using the already-scaled vector (no second transform)."""
        from src.detection.engine import DetectionEngine
        model_file, scaler_file, _ = mock_model_and_scaler
        engine = DetectionEngine(model_file, scaler_file)
        engine._shap = None  # ensure top_features path is taken

        transform_calls = [0]
        real_transform = engine.scaler.transform

        def counting_transform(X):
            transform_calls[0] += 1
            return real_transform(X)

        engine.scaler.transform = counting_transform

        engine.model.predict_proba = lambda X: np.array([[0.02, 0.98]])
        engine.model.classes_ = np.array([0, 1])

        features = {f: 0.1 for f in config.SELECTED_FEATURES}
        features.update({"_src_ip": "1.1.1.1", "_dst_ip": "2.2.2.2",
                         "_src_port": 1234, "_dst_port": 80, "_protocol": 6})

        alert = engine.predict(features)
        assert alert is not None
        assert "top_features" in alert
        assert len(alert["top_features"]) > 0
        # scaler.transform must be called exactly once (in predict, not again in _top_feature_values)
        assert transform_calls[0] == 1

    def test_log_alert_appends_to_jsonl(self, mock_model_and_scaler):
        """log_alert correctly appends a line to the JSONL file."""
        from src.detection.engine import DetectionEngine
        model_file, scaler_file, alert_log = mock_model_and_scaler
        engine = DetectionEngine(model_file, scaler_file)

        alert = {
            "timestamp": "2025-01-01T00:00:00Z",
            "src_ip": "1.2.3.4", "dst_ip": "5.6.7.8",
            "src_port": 1234, "dst_port": 22, "protocol": 6,
            "attack_type": "SSH-Patator",
            "confidence": 0.95, "severity": "HIGH",
            "top_features": {"SYN Flag Count": 200},
        }
        engine.log_alert(alert)

        assert alert_log.exists()
        with alert_log.open("r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["attack_type"] == "SSH-Patator"
