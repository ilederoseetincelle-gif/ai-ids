"""
extractor.py — Live network flow feature extraction.

Captures packets using Scapy and groups them into flows based on the
5-tuple (src_ip, dst_ip, src_port, dst_port, protocol). Once a flow is
idle for FLOW_TIMEOUT_SEC, it is finalized and its 20 features are
computed — matching the CICIDS2017 feature definitions.

This module is the bridge between raw network traffic and the ML model.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import config


@dataclass
class PacketMeta:
    """Lightweight record of a single packet in a flow."""
    timestamp: float
    length: int
    direction: str  # "fwd" or "bwd"
    tcp_flags: int = 0


@dataclass
class Flow:
    """Accumulator for packets belonging to a single 5-tuple."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    packets: List[PacketMeta] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def add(self, pkt: PacketMeta) -> None:
        """Append a packet to the flow and update timestamps."""
        if not self.packets:
            self.first_seen = pkt.timestamp
        self.last_seen = pkt.timestamp
        self.packets.append(pkt)

    def is_idle(self, now: float, timeout: float) -> bool:
        """True if no packets have arrived in the last `timeout` seconds."""
        return (now - self.last_seen) > timeout


class FlowExtractor:
    """
    Groups live packets into flows and extracts CICIDS2017-compatible features.

    Usage:
        def on_flow_complete(features: dict):
            print(features)

        fe = FlowExtractor(on_flow_complete)
        fe.process_packet(scapy_packet)
        # Periodically call fe.flush_idle_flows() to emit timed-out flows
    """

    def __init__(self, on_flow_complete: Callable[[dict], None]):
        self.flows: Dict[Tuple, Flow] = {}
        self.on_flow_complete = on_flow_complete
        self._lock = threading.Lock()

    def process_packet(self, packet) -> None:
        """
        Extract metadata from a Scapy packet and update the matching flow.
        Silently ignores non-IP or unparseable packets.

        Args:
            packet: A Scapy Packet object.
        """
        try:
            # Lazy import so this module can be imported without scapy installed
            from scapy.layers.inet import IP, TCP, UDP
        except ImportError:
            return

        if IP not in packet:
            return

        ip = packet[IP]
        src_ip, dst_ip = ip.src, ip.dst
        proto = ip.proto

        src_port = dst_port = 0
        tcp_flags = 0
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            tcp_flags = int(packet[TCP].flags)
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport

        # Canonical flow key (direction-aware): the first-seen direction is "fwd"
        key_fwd = (src_ip, dst_ip, src_port, dst_port, proto)
        key_bwd = (dst_ip, src_ip, dst_port, src_port, proto)

        with self._lock:
            if key_fwd in self.flows:
                flow = self.flows[key_fwd]
                direction = "fwd"
            elif key_bwd in self.flows:
                flow = self.flows[key_bwd]
                direction = "bwd"
            else:
                flow = Flow(src_ip, dst_ip, src_port, dst_port, proto)
                self.flows[key_fwd] = flow
                direction = "fwd"

            flow.add(PacketMeta(
                timestamp=float(packet.time),
                length=len(packet),
                direction=direction,
                tcp_flags=tcp_flags,
            ))

    def flush_idle_flows(self, now: Optional[float] = None) -> int:
        """
        Finalize flows that have been idle for more than FLOW_TIMEOUT_SEC,
        compute their features, and call on_flow_complete for each.

        Args:
            now: Current timestamp (default: time.time()).

        Returns:
            Number of flows flushed.
        """
        now = now if now is not None else time.time()
        to_finalize = []

        with self._lock:
            for key, flow in self.flows.items():
                idle = flow.is_idle(now, config.FLOW_TIMEOUT_SEC)
                too_long = (now - flow.first_seen) >= config.MAX_FLOW_DURATION
                if idle or too_long:
                    to_finalize.append((key, flow))
            for key, _ in to_finalize:
                del self.flows[key]

        count = 0
        for _, flow in to_finalize:
            features = self.extract_features(flow)
            if features is not None:
                self.on_flow_complete(features)
                count += 1
        return count

    def flush_all(self) -> int:
        """Force-finalize every flow (used at capture shutdown)."""
        with self._lock:
            flows_snapshot = list(self.flows.values())
            self.flows.clear()

        count = 0
        for flow in flows_snapshot:
            features = self.extract_features(flow)
            if features is not None:
                self.on_flow_complete(features)
                count += 1
        return count

    def extract_features(self, flow: Flow) -> Optional[dict]:
        """
        Compute the 20 selected features for a completed flow.
        Returns None if the flow is too short to produce meaningful features.

        The feature names match config.SELECTED_FEATURES exactly, so the
        output dict can be fed directly to the scaler and model.

        Args:
            flow: A Flow object with packets collected.

        Returns:
            Dictionary of feature name → float, or None if insufficient data.
        """
        if len(flow.packets) < 2:
            return None

        fwd = [p for p in flow.packets if p.direction == "fwd"]
        bwd = [p for p in flow.packets if p.direction == "bwd"]

        timestamps = [p.timestamp for p in flow.packets]
        duration_sec = flow.last_seen - flow.first_seen
        duration_us = max(duration_sec * 1e6, 1.0)  # microseconds, avoid div/0

        all_lengths = [p.length for p in flow.packets]
        fwd_lengths = [p.length for p in fwd] or [0]
        bwd_lengths = [p.length for p in bwd] or [0]

        # Inter-arrival times
        iats = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        iats_us = [x * 1e6 for x in iats] or [0]

        fwd_times = [p.timestamp for p in fwd]
        bwd_times = [p.timestamp for p in bwd]
        fwd_iats = [(fwd_times[i] - fwd_times[i - 1]) * 1e6
                    for i in range(1, len(fwd_times))] or [0]
        bwd_iats = [(bwd_times[i] - bwd_times[i - 1]) * 1e6
                    for i in range(1, len(bwd_times))] or [0]

        # TCP flag counts
        TCP_FIN, TCP_PSH, TCP_ACK = 0x01, 0x08, 0x10
        fin_count = sum(1 for p in flow.packets if p.tcp_flags & TCP_FIN)
        psh_count = sum(1 for p in flow.packets if p.tcp_flags & TCP_PSH)
        ack_count = sum(1 for p in flow.packets if p.tcp_flags & TCP_ACK)

        total_bytes = sum(all_lengths)
        total_packets = len(flow.packets)

        features = {
            "Flow Duration": duration_us,
            "Total Fwd Packets": float(len(fwd)),
            "Bwd Packets/s": len(bwd) / (duration_sec if duration_sec > 0 else 1),
            "Total Length of Fwd Packets": float(sum(fwd_lengths)),
            "Min Packet Length": float(min(all_lengths)),
            "Fwd Packet Length Max": float(max(fwd_lengths)),
            "Fwd Packet Length Min": float(min(fwd_lengths)),
            "Fwd Packet Length Mean": float(sum(fwd_lengths) / len(fwd_lengths)),
            "Bwd Packet Length Max": float(max(bwd_lengths)),
            "Bwd Packet Length Min": float(min(bwd_lengths)),
            "Bwd Packet Length Mean": float(sum(bwd_lengths) / len(bwd_lengths)),
            "Flow Bytes/s": total_bytes / (duration_sec if duration_sec > 0 else 1),
            "Flow Packets/s": total_packets / (duration_sec if duration_sec > 0 else 1),
            "Flow IAT Mean": float(sum(iats_us) / len(iats_us)),
            "Flow IAT Std": float(_std(iats_us)),
            "Fwd IAT Mean": float(sum(fwd_iats) / len(fwd_iats)),
            "Bwd IAT Mean": float(sum(bwd_iats) / len(bwd_iats)),
            "PSH Flag Count": float(psh_count),
            "FIN Flag Count": float(fin_count),
            "ACK Flag Count": float(ack_count),
            # Metadata for alerts (not fed to model)
            "_src_ip": flow.src_ip,
            "_dst_ip": flow.dst_ip,
            "_src_port": flow.src_port,
            "_dst_port": flow.dst_port,
            "_protocol": flow.protocol,
        }
        return features


def _std(values: List[float]) -> float:
    """Population standard deviation without numpy (lightweight)."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5
