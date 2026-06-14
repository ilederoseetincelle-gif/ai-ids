"""
main.py — Main entry point for the AI-IDS detection system.

Modes:
    live      Capture live traffic on a network interface and detect attacks
    replay    Replay a .pcap file through the detection engine
    dashboard Launch the Streamlit dashboard (alerts from logs/alerts.jsonl)
    demo      Run replay + dashboard together for demonstration

Examples:
    # Live detection (requires root)
    sudo python main.py --mode live --interface eth0

    # Replay a saved capture
    python main.py --mode replay --pcap tests/samples/portscan.pcap

    # Just the dashboard (for inspecting past alerts)
    python main.py --mode dashboard

    # Full demo: replay attack + dashboard in parallel
    python main.py --mode demo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered IDS — live detection and demonstration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["live", "replay", "dashboard", "demo"],
        help="Operation mode.",
    )
    parser.add_argument(
        "--interface", "-i", default="eth0",
        help="Network interface for live mode (e.g. eth0, enp0s8).",
    )
    parser.add_argument(
        "--pcap", type=Path,
        help="Path to .pcap file (required for --mode replay).",
    )
    parser.add_argument(
        "--port", type=int, default=config.DASHBOARD_PORT,
        help=f"Dashboard port (default: {config.DASHBOARD_PORT}).",
    )
    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Demo replay speed multiplier (default: 0.3 = 3x slower).",
    )
    parser.add_argument(
        "--clear-log", action="store_true",
        help="Clear the alert log before starting.",
    )
    return parser.parse_args()


def clear_alert_log() -> None:
    """Empty the alert log file to start fresh."""
    if config.ALERT_LOG_FILE.exists():
        config.ALERT_LOG_FILE.unlink()
    config.ALERT_LOG_FILE.touch()
    print(f"Cleared alert log: {config.ALERT_LOG_FILE}")


def run_live(interface: str) -> int:
    """Run the detection engine on a live interface."""
    from src.detection.engine import DetectionEngine

    try:
        engine = DetectionEngine()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    engine.start_live(interface)
    return 0


def run_replay(pcap_file: Path) -> int:
    """Replay a .pcap file through the detection engine."""
    from src.detection.engine import DetectionEngine

    if not pcap_file:
        print("ERROR: --pcap <file> is required for replay mode.")
        return 1
    if not pcap_file.exists():
        print(f"ERROR: pcap file not found: {pcap_file}")
        return 1

    try:
        engine = DetectionEngine()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    engine.replay_pcap(pcap_file)
    return 0


def run_dashboard(port: int) -> int:
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).resolve().parent / "src" / "dashboard" / "app.py"
    if not dashboard_path.exists():
        print(f"ERROR: Dashboard not found at {dashboard_path}")
        return 1

    print(f"Starting dashboard on http://localhost:{port} ...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(port),
            "--server.headless", "true",
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    return 0


def run_demo(speed: float, port: int) -> int:
    """
    Demonstration mode: launches the replay_demo + dashboard in parallel.
    Uses the pre-saved alert sequence if available, otherwise synthetic.
    """
    demo_script = Path(__file__).resolve().parent / "replay_demo.py"

    print("=" * 60)
    print(" AI-IDS DEMO MODE")
    print("=" * 60)
    print(f"Dashboard:  http://localhost:{port}")
    print(f"Replay:     running in background at {speed}x speed")
    print("=" * 60)

    # Start dashboard in background
    dashboard_path = Path(__file__).resolve().parent / "src" / "dashboard" / "app.py"
    dashboard_proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "true",
    ])

    # Give dashboard a moment to start
    time.sleep(3)

    # Run replay_demo in the foreground
    try:
        subprocess.run([
            sys.executable, str(demo_script),
            "--speed", str(speed),
        ])
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        dashboard_proc.terminate()
        dashboard_proc.wait(timeout=5)
    return 0


def main() -> int:
    args = parse_args()

    if args.clear_log:
        clear_alert_log()

    if args.mode == "live":
        return run_live(args.interface)
    elif args.mode == "replay":
        return run_replay(args.pcap)
    elif args.mode == "dashboard":
        return run_dashboard(args.port)
    elif args.mode == "demo":
        return run_demo(args.speed, args.port)

    return 1


if __name__ == "__main__":
    sys.exit(main())
