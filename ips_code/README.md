# AI-IDS — Intrusion Prevention (IPS) Module

This folder contains the new IPS layer for the AI-IDS project.

## What it does

When the detection engine raises an alert, the IPS:

1. Consults the response policy (per attack type, per severity)
2. Blocks the malicious source IP via `iptables` (if policy says BLOCK)
3. Schedules automatic unblock after the configured duration
4. Logs every action to `logs/ips_audit.jsonl`

The IPS never blocks whitelisted IPs (your laptop, gateway, etc.).

## Installation

```bash
# From the project root:
cd ~/Desktop/projects/bachelor_project/ai-ids
source ids-env/bin/activate

# 1. Copy the new files
cp -r /path/to/ips_code/src/prevention src/
cp /path/to/ips_code/config/whitelist.txt config/

# 2. Apply the integration patches (manual edits — see INTEGRATION_PATCH_*.py)
#    - Edit src/detection/engine.py
#    - Edit main.py
#    - Edit src/dashboard/app.py
#    - Append to config.py

# 3. Edit config/whitelist.txt to include YOUR laptop IP and gateway
```

## Usage

### Run with IPS in DRY-RUN mode (recommended first)

```bash
sudo "/path/to/ids-env/bin/python3" main.py --mode live \
    --interface wlp2s0 --ips --ips-dry-run
```

In dry-run, the IPS logs intended blocks but does NOT modify iptables.
Use this to verify the policy works correctly before enabling real blocking.

### Run with IPS in REAL mode

```bash
sudo "/path/to/ids-env/bin/python3" main.py --mode live \
    --interface wlp2s0 --ips
```

### Verify iptables rules

```bash
sudo iptables -L AI_IDS_BLOCK -n -v
```

### Emergency kill switch

If anything goes wrong (e.g., you accidentally blocked yourself):

```bash
sudo iptables -F AI_IDS_BLOCK
```

This instantly removes ALL blocks added by the AI-IDS, restoring connectivity.

## Files

| File | Purpose |
|------|---------|
| `src/prevention/__init__.py` | Package exports |
| `src/prevention/firewall.py` | iptables wrapper |
| `src/prevention/policy.py`   | Response decision engine |
| `src/prevention/responder.py`| Orchestrator + unblock daemon |
| `config/whitelist.txt`       | IPs never to block |
| `logs/ips_audit.jsonl`       | Audit log (auto-generated) |

## Safety features

- Whitelist (whitelist.txt) — never block listed IPs
- Auto-unblock after duration expiry
- Audit log (every action recorded with timestamp)
- Dry-run mode for safe testing
- Emergency kill switch (`iptables -F AI_IDS_BLOCK`)
- Web Attacks and Bots are LOG_ONLY (precision too low to auto-block)

## Defense talking points

1. The IPS turns the system from detection-only to active defense
2. The policy is conservative — only classes with >0.60 precision can block
3. Web Attacks (precision 0.06) and Bots (precision 0.03) are LOG_ONLY —
   demonstrating awareness that auto-blocking on low-precision predictions
   would create unacceptable false-positive harm
4. The whitelist and kill switch are explicit safety engineering decisions
