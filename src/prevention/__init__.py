"""
AI-IDS Prevention (IPS) module.

Provides reactive intrusion prevention via iptables firewall manipulation.
The IPS layer reads alerts from the detection engine, applies a configurable
response policy, and blocks malicious source IPs at the kernel level.

Safety features:
  - Whitelist support (critical IPs never blocked)
  - Configurable block durations
  - Automatic unblock after expiry
  - Dry-run mode for testing
  - Audit logging
"""

from .firewall import Firewall
from .responder import Responder
from .policy import ResponsePolicy

__all__ = ["Firewall", "Responder", "ResponsePolicy"]
