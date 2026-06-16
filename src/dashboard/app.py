"""
src/dashboard/app.py — Real-time Streamlit dashboard for the AI-IDS.

Reads alerts from config.ALERT_LOG_FILE (JSONL format) and auto-refreshes.

Layout:
    ┌─────────────────────────────────────────────────────┐
    │  🛡️  AI-IDS  |  Real-Time Intrusion Detection       │
    ├──────────┬──────────┬──────────┬────────────────────┤
    │  Total   │  Active  │  High    │  Avg Conf          │
    │  Alerts  │  Threats │  Severity│  (last 5 min)      │
    ├──────────┴──────────┴──────────┴────────────────────┤
    │  Alert Timeline (line chart — alerts per minute)    │
    ├───────────────────────┬─────────────────────────────┤
    │  Attack Type          │  Severity Distribution      │
    │  Breakdown (bar)      │  (pie chart)                │
    ├───────────────────────┴─────────────────────────────┤
    │  Live Alert Feed (scrolling table, last 50 alerts)  │
    └─────────────────────────────────────────────────────┘

Launch:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Make project root importable when Streamlit runs this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

ANOMALY_ATTACK_TYPE = "Unknown / Anomaly"

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-IDS — Real-Time Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }

    /* KPI cards */
    div[data-testid="stMetric"] {
        background: #141a24;
        border: 1px solid #243141;
        border-radius: 6px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] {
        color: #7a8fa3 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }
    div[data-testid="stMetricValue"] {
        color: #fff !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* Plotly charts background */
    .stPlotlyChart {
        background: #141a24;
        border: 1px solid #243141;
        border-radius: 6px;
        padding: 10px;
    }

    /* DataFrame */
    div[data-testid="stDataFrame"] {
        border: 1px solid #243141;
        border-radius: 6px;
    }

    /* Header */
    .header-title {
        font-size: 28px;
        font-weight: 800;
        color: #fff;
        letter-spacing: -0.02em;
    }
    .header-title span { color: #00d4ff; }
    .header-sub {
        font-size: 12px;
        color: #7a8fa3;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=2)  # cache for 2 seconds to reduce file reads
def load_alerts(path: Path) -> pd.DataFrame:
    """Read the JSONL alert log into a DataFrame."""
    if not path.exists():
        return pd.DataFrame()

    rows = []
    try:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute dashboard KPIs."""
    if df.empty:
        return {
            "total": 0, "high_severity": 0, "avg_confidence": 0.0,
            "attack_types": 0, "most_recent": "—", "anomaly_count": 0,
        }

    now = pd.Timestamp.utcnow()
    last_5min = df[df["timestamp"] >= now - pd.Timedelta(minutes=5)] if "timestamp" in df.columns else df

    # Exclude anomaly alerts from confidence average (they have confidence=0)
    conf_df = last_5min
    if "attack_type" in conf_df.columns:
        conf_df = conf_df[conf_df["attack_type"] != ANOMALY_ATTACK_TYPE]

    return {
        "total":          len(df),
        "high_severity":  int((df["severity"] == "HIGH").sum()) if "severity" in df.columns else 0,
        "avg_confidence": float(conf_df["confidence"].mean()) if not conf_df.empty and "confidence" in conf_df.columns else 0.0,
        "attack_types":   int(df["attack_type"].nunique()) if "attack_type" in df.columns else 0,
        "most_recent":    (df["timestamp"].iloc[0].strftime("%H:%M:%S") if not df.empty and "timestamp" in df.columns else "—"),
        "anomaly_count":  int((df.get("attack_type", pd.Series()) == ANOMALY_ATTACK_TYPE).sum()),
    }


def style_severity(val: str) -> str:
    """CSS style for severity cell coloring."""
    color_map = {
        "HIGH":   "background-color: #3a0d0d; color: #ff6b6b; font-weight: 700;",
        "MEDIUM": "background-color: #3a2a0d; color: #ffa94d; font-weight: 600;",
        "LOW":    "background-color: #0d2a3a; color: #4dd0e1;",
    }
    return color_map.get(val, "")


def style_attack_type(val: str) -> str:
    """Highlight anomaly alerts with a distinct magenta style."""
    if val == ANOMALY_ATTACK_TYPE:
        return "background-color: #2a0d3a; color: #d4a0ff; font-style: italic;"
    return ""


def _format_shap(alert_row) -> str:
    """
    Format the top SHAP contribution (or z-score top feature) for display
    in the alert table. Returns a short string like 'Flow Duration (+0.42)'.
    """
    contribs = alert_row.get("shap_contributions") if isinstance(alert_row, dict) else None
    if contribs and isinstance(contribs, dict):
        top_feat, top_val = next(iter(contribs.items()))
        sign = "+" if top_val >= 0 else ""
        return f"{top_feat} ({sign}{top_val:.2f})"

    top_feats = alert_row.get("top_features") if isinstance(alert_row, dict) else None
    if top_feats and isinstance(top_feats, dict):
        top_feat = next(iter(top_feats))
        return top_feat

    return "—"


# ─── HEADER ───────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(
        '<div class="header-title">🛡️ AI-<span>IDS</span>  Real-Time Intrusion Detection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="header-sub">// Network traffic ML classifier · CICIDS2017 trained</div>',
        unsafe_allow_html=True,
    )
with col_h2:
    now_display = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"<div style='text-align:right;color:#7a8fa3;font-size:12px;margin-top:8px;'>🕒 {now_display}</div>", unsafe_allow_html=True)

st.markdown("<hr style='margin:12px 0;border-color:#243141;'>", unsafe_allow_html=True)


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = load_alerts(config.ALERT_LOG_FILE)
kpis = compute_kpis(df)


# ─── KPI ROW ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Alerts",         f"{kpis['total']:,}")
k2.metric("High Severity",        f"{kpis['high_severity']:,}")
k3.metric("Avg Confidence (5m)",  f"{kpis['avg_confidence']:.0%}" if kpis['avg_confidence'] else "—")
k4.metric("Attack Types Seen",    f"{kpis['attack_types']}")
k5.metric("Anomaly Alerts",       f"{kpis['anomaly_count']:,}")
k6.metric("Latest Alert",         kpis["most_recent"])


st.markdown("<br>", unsafe_allow_html=True)

# ─── IF NO DATA, SHOW PLACEHOLDER ────────────────────────────────────────────
if df.empty:
    st.info(
        "🔍 **No alerts yet.** "
        "Start the detection engine with:\n\n"
        "```\npython main.py --mode live --interface eth0\n```\n\n"
        "or replay a demo:\n\n"
        "```\npython replay_demo.py\n```"
    )
    # Auto-refresh
    time.sleep(config.REFRESH_INTERVAL)
    st.rerun()


# ─── TIMELINE CHART ───────────────────────────────────────────────────────────
st.markdown("#### Alert Timeline — last 30 minutes")

if "timestamp" in df.columns:
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=30)
    recent = df[df["timestamp"] >= cutoff].copy()

    if not recent.empty:
        recent["minute"] = recent["timestamp"].dt.floor("min")
        timeline = recent.groupby(["minute", "severity"]).size().reset_index(name="count")

        severity_colors = {"HIGH": "#ff5252", "MEDIUM": "#ffb74d", "LOW": "#4dd0e1"}
        fig_timeline = px.bar(
            timeline, x="minute", y="count", color="severity",
            color_discrete_map=severity_colors,
            category_orders={"severity": ["HIGH", "MEDIUM", "LOW"]},
            height=260,
        )
        fig_timeline.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d8e8",
            xaxis_title=None, yaxis_title="Alerts per minute",
            xaxis=dict(gridcolor="#243141", showgrid=True),
            yaxis=dict(gridcolor="#243141", showgrid=True),
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No alerts in the last 30 minutes.")


# ─── CHARTS ROW (attack breakdown + severity pie) ───────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Attack Type Breakdown")
    if "attack_type" in df.columns and not df.empty:
        attack_counts = df["attack_type"].value_counts().reset_index()
        attack_counts.columns = ["attack_type", "count"]

        fig_attacks = px.bar(
            attack_counts.head(10),
            y="attack_type", x="count",
            orientation="h",
            color="count",
            color_continuous_scale=["#1a56a0", "#00d4ff"],
            height=320,
        )
        fig_attacks.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d8e8",
            xaxis_title="Count", yaxis_title=None,
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_attacks, use_container_width=True)

with col_b:
    st.markdown("#### Severity Distribution")
    if "severity" in df.columns and not df.empty:
        sev_counts = df["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        color_seq = [
            {"HIGH": "#ff5252", "MEDIUM": "#ffb74d", "LOW": "#4dd0e1"}.get(s, "#888")
            for s in sev_counts["severity"]
        ]
        fig_sev = go.Figure(data=[go.Pie(
            labels=sev_counts["severity"],
            values=sev_counts["count"],
            marker=dict(colors=color_seq),
            hole=0.55,
            textfont=dict(color="#fff", size=14),
        )])
        fig_sev.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d8e8",
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_sev, use_container_width=True)


# ─── TOP IPs AND PORTS ROW ────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### Top Attacker IPs")
    if "src_ip" in df.columns and not df.empty:
        ip_counts = df["src_ip"].value_counts().head(5).reset_index()
        ip_counts.columns = ["src_ip", "count"]
        fig_ips = px.bar(
            ip_counts, y="src_ip", x="count", orientation="h",
            color="count", color_continuous_scale=["#1a56a0", "#ff5252"],
            height=240,
        )
        fig_ips.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d8e8", xaxis_title="Alert Count", yaxis_title=None,
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False,
        )
        st.plotly_chart(fig_ips, use_container_width=True)

with col_d:
    st.markdown("#### Top Targeted Ports")
    if "dst_port" in df.columns and not df.empty:
        port_counts = df["dst_port"].value_counts().head(5).reset_index()
        port_counts.columns = ["dst_port", "count"]
        port_counts["dst_port"] = port_counts["dst_port"].astype(str)
        fig_ports = px.bar(
            port_counts, y="dst_port", x="count", orientation="h",
            color="count", color_continuous_scale=["#1a56a0", "#ffb74d"],
            height=240,
        )
        fig_ports.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d8e8", xaxis_title="Alert Count", yaxis_title="Port",
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False,
        )
        st.plotly_chart(fig_ports, use_container_width=True)


# ─── LIVE ALERT TABLE ─────────────────────────────────────────────────────────
st.markdown("#### Live Alert Feed — latest 50")

raw_rows = df.head(50).to_dict("records")
display_df = df.head(50).copy()

# Add top SHAP / z-score feature column
display_df["top_feature"] = [_format_shap(r) for r in raw_rows]

display_cols = ["timestamp", "severity", "attack_type", "src_ip", "dst_ip",
                "dst_port", "confidence", "top_feature"]
for col in display_cols:
    if col not in display_df.columns:
        display_df[col] = "—"

if "timestamp" in display_df.columns:
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")
if "confidence" in display_df.columns:
    display_df["confidence"] = display_df["confidence"].apply(
        lambda x: f"{x:.1%}" if isinstance(x, (int, float)) and x > 0 else "—"
    )

styled = (
    display_df[display_cols]
    .style
    .map(style_severity, subset=["severity"])
    .map(style_attack_type, subset=["attack_type"])
)
st.dataframe(styled, use_container_width=True, hide_index=True, height=400)


# ─── IPS PANEL ────────────────────────────────────────────────────────────────
def _render_ips_panel() -> None:
    """Render the IPS activity panel (only shown when ips_audit.jsonl exists)."""
    audit_path = Path(config.IPS_AUDIT_LOG)
    if not audit_path.exists():
        return

    st.markdown("<hr style='margin:18px 0;border-color:#243141;'>", unsafe_allow_html=True)
    st.markdown("#### 🛡️ Intrusion Prevention (IPS)")

    records = []
    try:
        with audit_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        st.warning("Could not read IPS audit log.")
        return

    if not records:
        st.info("No IPS actions yet.")
        return

    ips_df = pd.DataFrame(records)
    ips_df["timestamp"] = pd.to_datetime(ips_df["timestamp"], errors="coerce", utc=True)

    total_blocks   = int(ips_df["action"].isin(["BLOCK", "BLOCK_DRYRUN"]).sum())
    total_unblocks = int(ips_df["action"].isin(["UNBLOCK", "UNBLOCK_DRYRUN"]).sum())
    active_blocks  = total_blocks - total_unblocks
    unique_ips     = int(ips_df[ips_df["action"] == "BLOCK"]["ip"].nunique()) if "ip" in ips_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Blocks",     active_blocks)
    c2.metric("Total Blocks",      total_blocks)
    c3.metric("Total Unblocks",    total_unblocks)
    c4.metric("Unique IPs Blocked", unique_ips)

    if "ip" in ips_df.columns:
        st.markdown("**Currently Blocked IPs**")
        blocked = ips_df[ips_df["action"].isin(["BLOCK", "BLOCK_DRYRUN"])]
        unblocked_ips = set(ips_df[ips_df["action"].isin(["UNBLOCK", "UNBLOCK_DRYRUN"])]["ip"])
        active_df = blocked[~blocked["ip"].isin(unblocked_ips)].drop_duplicates("ip")
        if not active_df.empty:
            show_cols = [c for c in ["ip", "timestamp", "reason"] if c in active_df.columns]
            st.dataframe(active_df[show_cols], use_container_width=True, hide_index=True)
        else:
            st.write("No active blocks.")

    st.markdown("**Recent IPS Activity (last 20)**")
    recent = ips_df.sort_values("timestamp", ascending=False).head(20)
    show_cols = [c for c in ["timestamp", "action", "ip", "reason"] if c in recent.columns]
    st.dataframe(recent[show_cols], use_container_width=True, hide_index=True)


_render_ips_panel()


# ─── AUTO-REFRESH ─────────────────────────────────────────────────────────────
time.sleep(config.REFRESH_INTERVAL)
st.rerun()


def render_ips_panel():
    """Render the IPS activity panel in the Streamlit dashboard."""
    import streamlit as st
    import pandas as pd
    from pathlib import Path
    import json

    st.subheader("🛡️ Intrusion Prevention (IPS)")

    audit_path = Path("logs/ips_audit.jsonl")
    if not audit_path.exists():
        st.info("IPS audit log not found — run with --ips flag to enable.")
        return

    # Load all audit records
    records = []
    with open(audit_path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        st.info("No IPS actions yet.")
        return

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # KPI row
    total_blocks = len(df[df["action"].isin(["BLOCK", "BLOCK_DRYRUN"])])
    total_unblocks = len(df[df["action"].isin(["UNBLOCK", "UNBLOCK_DRYRUN"])])
    active_blocks = total_blocks - total_unblocks
    unique_ips_blocked = df[df["action"] == "BLOCK"]["ip"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Blocks", active_blocks)
    col2.metric("Total Blocks", total_blocks)
    col3.metric("Total Unblocks", total_unblocks)
    col4.metric("Unique IPs Blocked", unique_ips_blocked)

    # Currently active blocks
    st.markdown("**Currently Blocked IPs**")
    active_df = df[df["action"] == "BLOCK"].groupby("ip").last().reset_index()
    if not active_df.empty:
        st.dataframe(active_df[["ip", "timestamp", "reason"]],
                     use_container_width=True)
    else:
        st.write("No active blocks.")

    # Recent audit log entries
    st.markdown("**Recent IPS Activity (last 20)**")
    recent = df.sort_values("timestamp", ascending=False).head(20)
    st.dataframe(recent[["timestamp", "action", "ip", "reason"]],
                 use_container_width=True)

