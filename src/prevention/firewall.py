"""
Firewall.py — Low-level iptables wrapper for the AI-IDS IPS layer.

This module provides a clean Python interface around the Linux iptables
command. It is responsible for the actual kernel-level blocking of IP
addresses identified as malicious by the detection engine.

Why iptables (and not nftables)?
  - Universally available on every Linux distribution
  - Well-documented and stable interface
  - Compatible with Ubuntu 22.04, Kali, and Debian
  - Trivial to verify manually: `sudo iptables -L INPUT -n`

Why a custom chain (AI_IDS_BLOCK)?
  - Isolates IPS rules from existing firewall configuration
  - Easy to flush without affecting other rules: `iptables -F AI_IDS_BLOCK`
  - Easy to monitor: `iptables -L AI_IDS_BLOCK -n -v`
"""

import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Set

logger = logging.getLogger("ips.firewall")


class FirewallError(Exception):
    """Raised when an iptables command fails."""
    pass


class Firewall:
    """
    Linux iptables wrapper for IP-level blocking.

    Maintains a dedicated AI_IDS_BLOCK chain in the INPUT table. All blocks
    inserted by the AI-IDS are isolated in this chain, making them easy to
    audit, monitor, and revoke.
    """

    CHAIN_NAME = "AI_IDS_BLOCK"

    def __init__(self, dry_run: bool = False, audit_log: Optional[Path] = None):
        """
        Parameters
        ----------
        dry_run : bool
            If True, log intended actions but do not execute iptables commands.
            Useful for testing and demonstration without modifying the firewall.
        audit_log : Path
            Path to the audit log file (one JSON-line per block/unblock action).
        """
        self.dry_run = dry_run
        self.audit_log = audit_log
        self._blocked_ips: Set[str] = set()

        if not dry_run:
            self._ensure_chain()

    # ── Public API ────────────────────────────────────────────────────────

    def block(self, ip: str, reason: str = "AI-IDS detection") -> bool:
        """
        Block all incoming traffic from the given IP address.

        Returns True if the block was applied, False if the IP was already
        blocked or could not be blocked.
        """
        if ip in self._blocked_ips:
            logger.info(f"IP {ip} already blocked; skipping duplicate.")
            return False

        if self.dry_run:
            logger.warning(f"[DRY RUN] Would block {ip} — reason: {reason}")
            self._blocked_ips.add(ip)
            self._audit("BLOCK_DRYRUN", ip, reason)
            return True

        cmd = ["sudo", "iptables", "-I", self.CHAIN_NAME, "1",
               "-s", ip, "-j", "DROP"]
        try:
            self._run(cmd)
            self._blocked_ips.add(ip)
            self._audit("BLOCK", ip, reason)
            logger.warning(f"BLOCKED {ip} — {reason}")
            return True
        except FirewallError as e:
            logger.error(f"Failed to block {ip}: {e}")
            return False

    def unblock(self, ip: str, reason: str = "block expired") -> bool:
        """
        Remove the block on a specific IP address.

        Returns True if the unblock was applied, False otherwise.
        """
        if ip not in self._blocked_ips:
            logger.info(f"IP {ip} is not in blocked set; nothing to unblock.")
            return False

        if self.dry_run:
            logger.info(f"[DRY RUN] Would unblock {ip} — reason: {reason}")
            self._blocked_ips.discard(ip)
            self._audit("UNBLOCK_DRYRUN", ip, reason)
            return True

        cmd = ["sudo", "iptables", "-D", self.CHAIN_NAME,
               "-s", ip, "-j", "DROP"]
        try:
            self._run(cmd)
            self._blocked_ips.discard(ip)
            self._audit("UNBLOCK", ip, reason)
            logger.info(f"UNBLOCKED {ip} — {reason}")
            return True
        except FirewallError as e:
            logger.error(f"Failed to unblock {ip}: {e}")
            return False

    def flush_all(self) -> None:
        """
        Emergency kill switch: remove ALL blocks created by the AI-IDS.

        This restores connectivity instantly if the IPS misbehaves. Equivalent
        to running `sudo iptables -F AI_IDS_BLOCK` manually.
        """
        if self.dry_run:
            logger.warning("[DRY RUN] Would flush all AI-IDS blocks")
            self._blocked_ips.clear()
            return

        try:
            self._run(["sudo", "iptables", "-F", self.CHAIN_NAME])
            count = len(self._blocked_ips)
            self._audit("FLUSH_ALL", "*", f"emergency flush — {count} IPs released")
            logger.warning(f"FLUSHED {count} blocked IPs")
            self._blocked_ips.clear()
        except FirewallError as e:
            logger.error(f"Failed to flush AI-IDS chain: {e}")

    def blocked_ips(self) -> List[str]:
        """Return a list of currently blocked IP addresses."""
        return sorted(self._blocked_ips)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _ensure_chain(self) -> None:
        """
        Create the AI_IDS_BLOCK chain if it does not exist and ensure it is
        referenced from the INPUT chain.
        """
        # Create chain (ignore error if it already exists)
        try:
            self._run(["sudo", "iptables", "-N", self.CHAIN_NAME])
            logger.info(f"Created iptables chain {self.CHAIN_NAME}")
        except FirewallError:
            # Chain already exists — fine
            pass

        # Ensure INPUT chain jumps to our chain (idempotent check)
        result = subprocess.run(
            ["sudo", "iptables", "-C", "INPUT", "-j", self.CHAIN_NAME],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            self._run(["sudo", "iptables", "-I", "INPUT", "1",
                       "-j", self.CHAIN_NAME])
            logger.info(f"Linked INPUT chain to {self.CHAIN_NAME}")

    def _run(self, cmd: List[str]) -> str:
        """Run an iptables command and return stdout. Raises on non-zero exit."""
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise FirewallError(
                f"Command failed: {' '.join(cmd)}\n"
                f"stderr: {result.stderr.strip()}"
            )
        return result.stdout

    def _audit(self, action: str, ip: str, reason: str) -> None:
        """Append an audit record to the audit log file."""
        if self.audit_log is None:
            return

        import json
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "ip": ip,
            "reason": reason,
        }
        try:
            with open(self.audit_log, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.error(f"Failed to write audit log: {e}")
