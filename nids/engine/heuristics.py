"""Rules for attacks that only show up across many flows.

The model scores one flow at a time. A port scan or SYN flood is thousands of
tiny flows that each look unremarkable, and many of them are single packets,
which the model never saw in training (the dataset drops one-packet flows).
These two counters watch the pattern instead.
"""

from __future__ import annotations

from collections import deque

from nids.engine.flowmeter import TCP, Flow

MAX_TRACKED = 50_000  # sources or services remembered at once


def _forget_stale(seen: dict, now: float, window: float) -> None:
    """Keep the tables bounded: drop entries with nothing recent in them."""
    for key in [k for k, v in seen.items() if not v or _time(v[-1]) < now - window]:
        del seen[key]


def _time(entry) -> float:
    return entry[0] if isinstance(entry, tuple) else entry


class SynFloodDetector:
    """Many short, reset or unanswered SYN flows from one source to one service."""

    label = "DDoS"
    description = "SYN flood (rule)"

    def __init__(self, window_seconds: float = 10, threshold: int = 20):
        self.window = window_seconds
        self.threshold = threshold
        self._seen: dict[tuple, deque[float]] = {}

    def check(self, flow: Flow) -> bool:
        if not (flow.total_syn_flags and flow.total_packets <= 5 and (flow.total_rst_flags or flow.total_packets <= 2)):
            return False
        now = flow.last_seen / 1e6
        if len(self._seen) > MAX_TRACKED:
            _forget_stale(self._seen, now, self.window)
        times = self._seen.setdefault((flow.src_ip, flow.dst_ip, flow.dst_port), deque())
        times.append(now)
        while times[0] < now - self.window:
            times.popleft()
        return len(times) >= self.threshold


class PortScanDetector:
    """One source making empty connection attempts to many different (host, port) pairs."""

    label = "Portscan"
    description = "Port scan (rule)"

    def __init__(self, window_seconds: float = 30, threshold: int = 15):
        self.window = window_seconds
        self.threshold = threshold
        self._seen: dict[str, deque[tuple[float, str, int]]] = {}

    def check(self, flow: Flow) -> bool:
        # A probe: a TCP connection attempt that never carried any data. UDP is left
        # out because DNS, NTP and service discovery look exactly like this.
        if not (flow.protocol == TCP and flow.total_packets <= 3 and flow.total_syn_flags and not flow.all_len.total):
            return False
        now = flow.last_seen / 1e6
        if len(self._seen) > MAX_TRACKED:
            _forget_stale(self._seen, now, self.window)
        contacts = self._seen.setdefault(flow.src_ip, deque())
        contacts.append((now, flow.dst_ip, flow.dst_port))
        while contacts[0][0] < now - self.window:
            contacts.popleft()
        return len({(host, port) for _, host, port in contacts}) >= self.threshold
