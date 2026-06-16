"""
Policy.py — Response policy engine for the AI-IDS IPS layer.

This module decides WHAT to do when an alert is generated. The decision is
based on three factors:
  1. The detected attack type
  2. The alert severity (HIGH / MEDIUM / LOW)
  3. The frequency of recent alerts from the same source

The policy is deliberately conservative for attack classes with low precision
(Web Attacks, Bots). Auto-blocking based on a 3% precision prediction would
produce 97 false-positive blocks for every 3 true-positive blocks — an
unacceptable trade-off.

Configuration is loaded from config.py to keep all tunable values in one place.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class ResponseAction:
    """A decision returned by the policy engine."""
    action: str           # "BLOCK", "LOG_ONLY", "ESCALATE", "IGNORE"
    duration_sec: int     # Block duration in seconds (0 if no block)
    reason: str           # Human-readable explanation


@dataclass
class ResponsePolicy:
    """
    Encapsulates the AI-IDS response policy.

    Default policy rationale:
      - DDoS (F1: 1.00):       BLOCK 1h — perfect precision, safe to auto-block
      - Port Scanning (0.99):  BLOCK 30m — reconnaissance is rarely legitimate
      - DoS (0.98):            BLOCK 1h — high precision, high impact attack
      - Brute Force (0.76):    BLOCK 1h — moderate precision, but high impact
      - Web Attacks (0.12):    LOG_ONLY — precision too low for auto-block
      - Bots (0.06):           LOG_ONLY — precision too low for auto-block
      - MEDIUM severity:       ESCALATE — block only on 3rd alert in 60s
      - LOW severity:          LOG_ONLY — borderline detections, log only
    """

    # Attack-type to action mapping for HIGH-severity alerts
    high_severity_actions: Dict[str, str] = field(default_factory=lambda: {
        "DDoS":          "BLOCK",
        "DoS":           "BLOCK",
        "Port Scanning": "BLOCK",
        "Brute Force":   "BLOCK",
        "Web Attacks":   "LOG_ONLY",
        "Bots":          "LOG_ONLY",
    })

    # Block duration per attack type (seconds)
    block_durations: Dict[str, int] = field(default_factory=lambda: {
        "DDoS":          3600,   # 1 hour
        "DoS":           3600,   # 1 hour
        "Port Scanning":  1800,  # 30 minutes
        "Brute Force":   3600,   # 1 hour
    })

    # Escalation: block on Nth MEDIUM alert within window_sec
    escalation_threshold: int = 3
    escalation_window_sec: int = 60
    escalation_block_duration: int = 900  # 15 minutes

    # Whitelist: these IPs are NEVER blocked
    whitelist: List[str] = field(default_factory=list)

    # Internal: track recent alerts per source IP for escalation
    _recent_alerts: Dict[str, List[datetime]] = field(
        default_factory=lambda: defaultdict(list))

    def decide(self, alert: dict) -> ResponseAction:
        """
        Decide the appropriate response for an alert.

        Parameters
        ----------
        alert : dict
            Alert dictionary with keys: 'src_ip', 'attack_type', 'severity',
            'confidence', 'timestamp'.

        Returns
        -------
        ResponseAction
            The decision: BLOCK, LOG_ONLY, ESCALATE, or IGNORE.
        """
        src_ip = alert.get("src_ip")
        attack = alert.get("attack_type")
        severity = alert.get("severity", "LOW").upper()

        # ── Safety: never block whitelisted IPs ───────────────────────────
        if src_ip in self.whitelist:
            return ResponseAction(
                action="IGNORE",
                duration_sec=0,
                reason=f"{src_ip} is whitelisted"
            )

        # ── HIGH severity: per-attack-type policy ─────────────────────────
        if severity == "HIGH":
            action = self.high_severity_actions.get(attack, "LOG_ONLY")
            if action == "BLOCK":
                duration = self.block_durations.get(attack, 1800)
                return ResponseAction(
                    action="BLOCK",
                    duration_sec=duration,
                    reason=f"HIGH severity {attack} from {src_ip}"
                )
            return ResponseAction(
                action="LOG_ONLY",
                duration_sec=0,
                reason=f"{attack} precision too low for auto-block"
            )

        # ── MEDIUM severity: escalation logic ─────────────────────────────
        if severity == "MEDIUM":
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=self.escalation_window_sec)

            # Prune old alerts outside the window
            self._recent_alerts[src_ip] = [
                t for t in self._recent_alerts[src_ip] if t >= window_start
            ]
            self._recent_alerts[src_ip].append(now)

            count = len(self._recent_alerts[src_ip])
            if count >= self.escalation_threshold:
                return ResponseAction(
                    action="BLOCK",
                    duration_sec=self.escalation_block_duration,
                    reason=(f"Escalated: {count} MEDIUM alerts from {src_ip} "
                            f"within {self.escalation_window_sec}s")
                )
            return ResponseAction(
                action="ESCALATE",
                duration_sec=0,
                reason=f"MEDIUM alert {count}/{self.escalation_threshold}"
            )

        # ── LOW severity: log only ────────────────────────────────────────
        return ResponseAction(
            action="LOG_ONLY",
            duration_sec=0,
            reason="LOW severity — monitoring only"
        )

    def load_whitelist(self, path: str) -> int:
        """
        Load a whitelist file (one IP per line, # for comments).
        Returns the number of IPs loaded.
        """
        try:
            with open(path) as f:
                self.whitelist = [
                    line.strip() for line in f
                    if line.strip() and not line.startswith("#")
                ]
            return len(self.whitelist)
        except FileNotFoundError:
            return 0
