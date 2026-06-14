"""
replay_demo.py — Standalone demo replay for the defense presentation.

Generates a realistic stream of attack alerts WITHOUT requiring live VMs
or network capture. Reads from either:
  (a) a pre-saved JSONL file (tests/samples/demo_alerts.jsonl), or
  (b) synthesizes a realistic attack sequence on the fly.

Alerts are written to config.ALERT_LOG_FILE with configurable delays,
so the Streamlit dashboard shows them appearing in real time.

Usage:
    python replay_demo.py                      # Synthesize attack sequence
    python replay_demo.py --source saved       # Use tests/samples/demo_alerts.jsonl
    python replay_demo.py --speed 0.1          # Very fast (10% delays)
    python replay_demo.py --speed 1.0          # Normal speed

Purpose:
    This is the INSURANCE POLICY for defense day. If the live demo fails,
    run this to produce a smooth, reliable visual demonstration that tells
    the same technical story.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config


# ─── ATTACK SEQUENCE (synthetic) ──────────────────────────────────────────────
# A realistic narrative: scan → enumerate → brute force → escalate → DoS
ATTACK_SCENARIO = [
    # Phase 1 — Reconnaissance (port scanning)
    *[{"attack_type": "Port Scanning", "severity": "MEDIUM", "confidence": 0.78,
       "src_ip": "192.168.56.10", "dst_ip": "192.168.56.20",
       "src_port": None, "dst_port": p, "protocol": 6}
      for p in [21, 22, 23, 25, 53, 80, 110, 443, 445, 3306, 3389, 8080]],

    # Phase 2 — SSH brute force (grouped as "Brute Force" by the model)
    *[{"attack_type": "Brute Force", "severity": sev, "confidence": conf,
       "src_ip": "192.168.56.10", "dst_ip": "192.168.56.20",
       "src_port": None, "dst_port": 22, "protocol": 6}
      for sev, conf in [("MEDIUM", 0.82), ("HIGH", 0.91), ("HIGH", 0.94),
                        ("HIGH", 0.95), ("HIGH", 0.96)]],

    # Phase 3 — FTP brute force (in parallel from "another" IP)
    *[{"attack_type": "Brute Force", "severity": sev, "confidence": conf,
       "src_ip": "192.168.56.11", "dst_ip": "192.168.56.20",
       "src_port": None, "dst_port": 21, "protocol": 6}
      for sev, conf in [("MEDIUM", 0.77), ("HIGH", 0.89), ("HIGH", 0.92)]],

    # Phase 4 — Web attacks
    {"attack_type": "Web Attacks", "severity": "HIGH", "confidence": 0.93,
     "src_ip": "192.168.56.10", "dst_ip": "192.168.56.20",
     "src_port": None, "dst_port": 80, "protocol": 6},
    {"attack_type": "Web Attacks", "severity": "MEDIUM", "confidence": 0.81,
     "src_ip": "192.168.56.10", "dst_ip": "192.168.56.20",
     "src_port": None, "dst_port": 80, "protocol": 6},
    {"attack_type": "Web Attacks", "severity": "HIGH", "confidence": 0.95,
     "src_ip": "192.168.56.10", "dst_ip": "192.168.56.20",
     "src_port": None, "dst_port": 80, "protocol": 6},

    # Phase 5 — DoS escalation
    *[{"attack_type": "DoS", "severity": "HIGH", "confidence": c,
       "src_ip": "192.168.56.10", "dst_ip": "192.168.56.20",
       "src_port": None, "dst_port": 80, "protocol": 6}
      for c in [0.92, 0.94, 0.96, 0.97, 0.98, 0.98, 0.99]],

    # Phase 6 — DDoS (multiple source IPs)
    *[{"attack_type": "DDoS", "severity": "HIGH", "confidence": 0.99,
       "src_ip": f"192.168.56.{50 + i}", "dst_ip": "192.168.56.20",
       "src_port": None, "dst_port": 80, "protocol": 6}
      for i in range(8)],

    # Phase 7 — botnet activity
    {"attack_type": "Bots", "severity": "HIGH", "confidence": 0.88,
     "src_ip": "192.168.56.30", "dst_ip": "45.33.12.100",
     "src_port": None, "dst_port": 6667, "protocol": 6},
    {"attack_type": "Bots", "severity": "MEDIUM", "confidence": 0.79,
     "src_ip": "192.168.56.30", "dst_ip": "45.33.12.100",
     "src_port": None, "dst_port": 6667, "protocol": 6},
]


def make_alert(base: dict) -> dict:
    """Build a full alert dict from a scenario entry + current timestamp."""
    src_port = base["src_port"] if base["src_port"] is not None else random.randint(40000, 60000)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "src_ip": base["src_ip"],
        "dst_ip": base["dst_ip"],
        "src_port": src_port,
        "dst_port": base["dst_port"],
        "protocol": base["protocol"],
        "attack_type": base["attack_type"],
        "confidence": round(base["confidence"], 4),
        "severity": base["severity"],
        "top_features": {
            "Flow Packets/s": round(random.uniform(100, 5000), 2),
            "FIN Flag Count": random.randint(0, 20),
            "Flow Duration": round(random.uniform(1e3, 1e7), 0),
        },
    }


def load_saved_alerts(path: Path) -> list:
    """Load pre-saved alerts from a JSONL file."""
    if not path.exists():
        return []
    alerts = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                alerts.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return alerts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay attack alerts to the dashboard for demo purposes.",
    )
    parser.add_argument(
        "--source", choices=["synthetic", "saved"], default="synthetic",
        help="Where alerts come from (synthetic = generate; saved = file).",
    )
    parser.add_argument(
        "--file", type=Path,
        default=config.SAMPLES_DIR / "demo_alerts.jsonl",
        help="Saved alert file to replay (used with --source saved).",
    )
    parser.add_argument(
        "--speed", type=float, default=0.5,
        help="Speed multiplier — smaller = faster. 0.5 = half speed, 2.0 = 2x slower.",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop forever (replay continuously).",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear the alert log before starting.",
    )
    return parser.parse_args()


def replay_once(alerts: list, speed: float) -> None:
    """Write alerts to the log file one by one with delays between them."""
    log_file = config.ALERT_LOG_FILE
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(alerts)} alerts to {log_file}")
    print(f"Dashboard: http://localhost:{config.DASHBOARD_PORT}")
    print(f"Speed:     {speed}x  (lower = faster)")
    print("-" * 60)

    for i, alert in enumerate(alerts, 1):
        # For synthetic alerts, re-stamp with current time
        if "timestamp" not in alert:
            alert = make_alert(alert) if all(
                k in alert for k in ("attack_type", "src_ip", "dst_ip")
            ) else alert
        else:
            alert = dict(alert)
            alert["timestamp"] = datetime.now(timezone.utc).isoformat()

        with log_file.open("a") as f:
            f.write(json.dumps(alert) + "\n")

        sev = alert.get("severity", "—")
        atk = alert.get("attack_type", "?")
        print(f"[{i:03d}/{len(alerts)}]  {sev:<6}  {atk:<30}  {alert.get('src_ip')} → {alert.get('dst_ip')}:{alert.get('dst_port')}")

        # Variable delay: HIGH severity alerts come faster, LOW slower
        base_delay = 0.8 if sev == "HIGH" else 1.2 if sev == "MEDIUM" else 1.6
        time.sleep(base_delay * speed)


def main() -> int:
    args = parse_args()

    if args.clear:
        if config.ALERT_LOG_FILE.exists():
            config.ALERT_LOG_FILE.unlink()
        config.ALERT_LOG_FILE.touch()
        print(f"Cleared {config.ALERT_LOG_FILE}")

    # Gather alerts
    if args.source == "saved":
        alerts = load_saved_alerts(args.file)
        if not alerts:
            print(f"No alerts in {args.file}, falling back to synthetic.")
            alerts = [make_alert(a) for a in ATTACK_SCENARIO]
    else:
        alerts = [make_alert(a) for a in ATTACK_SCENARIO]

    print(f"\n{'='*60}")
    print(" AI-IDS DEMO REPLAY")
    print(f"{'='*60}")

    try:
        if args.loop:
            iteration = 1
            while True:
                print(f"\n>>> Iteration {iteration}")
                replay_once(alerts, args.speed)
                iteration += 1
                print("\nLoop: restarting in 5 seconds...")
                time.sleep(5)
        else:
            replay_once(alerts, args.speed)
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")

    print(f"\n{'='*60}")
    print(f" Total alerts written: see {config.ALERT_LOG_FILE}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
