"""Turn packets into flows and flows into the 84 CICFlowMeter features.

The model was trained on flows produced by the dataset authors' fork of
CICFlowMeter (https://github.com/GintsEngelen/CICFlowMeter). A model only works
on live traffic if live flows are built and measured the same way, so this
module follows that tool's logic step by step, including its quirks. Each quirk
is marked "CICFlowMeter:" below. `scripts/parity_report.py` checks the result
against the dataset's own flows for the same packets.

No third-party imports: this runs anywhere Python does.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

from nids.features import FEATURE_NAMES_84

TCP, UDP, ICMP = 6, 17, 1
FIN, SYN, RST, PSH, ACK, URG, ECE, CWR = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80

# The dataset's settings (microseconds).
FLOW_TIMEOUT_US = 120_000_000  # a flow is cut 120 s after it started
ACTIVITY_TIMEOUT_US = 5_000_000  # a gap this long counts as "idle"
BULK_GAP_US = 1_000_000  # packets further apart than this start a new bulk
BULK_MIN_PACKETS = 4

# TCP teardown, as tracked by CICFlowMeter.
_OPEN, _FIRST_FIN, _SECOND_FIN, _TERMINATED = 0, 1, 2, 3


@dataclass(slots=True)
class Packet:
    """The fields of one packet that flow features depend on."""

    ts: int  # microseconds since the epoch
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    payload: int  # bytes after the TCP/UDP header (0 for ICMP)
    header: int  # TCP header incl. options, 8 for UDP, 0 for ICMP
    flags: int = 0  # TCP flag byte
    window: int = 0
    icmp_type: int = -1
    icmp_code: int = -1


class _Stats:
    """Running count / sum / min / max / sample standard deviation (Welford)."""

    __slots__ = ("n", "total", "lo", "hi", "_mean", "_m2")

    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.lo = math.inf
        self.hi = -math.inf
        self._mean = 0.0
        self._m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        self.total += x
        if x < self.lo:
            self.lo = x
        if x > self.hi:
            self.hi = x
        delta = x - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (x - self._mean)

    @property
    def mean(self) -> float:
        return self._mean if self.n else 0.0

    @property
    def variance(self) -> float:
        return self._m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def summary(self) -> tuple[float, float, float, float]:
        """(mean, std, max, min), all 0 when empty."""
        if not self.n:
            return 0.0, 0.0, 0.0, 0.0
        return self._mean, self.std, self.hi, self.lo


class _Bulk:
    """Bulk transfer tracking for one direction: runs of 4+ data packets under 1 s apart."""

    __slots__ = ("start", "last", "run_packets", "run_bytes", "count", "packets", "bytes", "duration")

    def __init__(self) -> None:
        self.start = 0
        self.last = 0
        self.run_packets = 0
        self.run_bytes = 0
        self.count = 0
        self.packets = 0
        self.bytes = 0
        self.duration = 0

    def update(self, ts: int, size: int, other_last: int) -> None:
        if other_last > self.start:  # the other side sent data in between: run broken
            self.start = 0
        if size <= 0:
            return
        if self.start == 0 or ts - self.last > BULK_GAP_US:
            self.start = self.last = ts
            self.run_packets, self.run_bytes = 1, size
            return
        self.run_packets += 1
        self.run_bytes += size
        if self.run_packets == BULK_MIN_PACKETS:
            self.count += 1
            self.packets += self.run_packets
            self.bytes += self.run_bytes
            self.duration += ts - self.start
        elif self.run_packets > BULK_MIN_PACKETS:
            self.packets += 1
            self.bytes += size
            self.duration += ts - self.last
        self.last = ts

    def features(self) -> tuple[int, int, int]:
        """(bytes per bulk, packets per bulk, bytes per second while in bulk).

        CICFlowMeter (dataset build): all three are truncated to whole numbers.
        """
        per_bulk = (self.bytes // self.count, self.packets // self.count) if self.count else (0, 0)
        rate = int(self.bytes / (self.duration / 1e6)) if self.duration else 0
        return per_bulk[0], per_bulk[1], rate


@dataclass(slots=True)
class Flow:
    """One bidirectional flow. "Forward" is the direction of the side that opened it."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    start: int  # microseconds
    last_seen: int
    carried_duration: int = 0  # time already spent in earlier, timed-out parts of this connection

    fwd_len: _Stats = field(default_factory=_Stats)
    bwd_len: _Stats = field(default_factory=_Stats)
    all_len: _Stats = field(default_factory=_Stats)
    flow_iat: _Stats = field(default_factory=_Stats)
    fwd_iat: _Stats = field(default_factory=_Stats)
    bwd_iat: _Stats = field(default_factory=_Stats)
    active: _Stats = field(default_factory=_Stats)
    idle: _Stats = field(default_factory=_Stats)
    fwd_bulk: _Bulk = field(default_factory=_Bulk)
    bwd_bulk: _Bulk = field(default_factory=_Bulk)

    fwd_last: int = 0
    bwd_last: int = 0
    fwd_header: int = 0
    bwd_header: int = 0
    fwd_seg_min: int = 0
    fwd_data_packets: int = 0
    fwd_init_window: int = 0
    bwd_init_window: int = 0
    flag_counts: list[int] = field(default_factory=lambda: [0] * 8)  # FIN SYN RST PSH ACK URG ECE CWR
    fwd_psh: int = 0
    bwd_psh: int = 0
    fwd_urg: int = 0
    bwd_urg: int = 0
    fwd_rst: int = 0
    bwd_rst: int = 0
    fwd_fin: int = 0
    bwd_fin: int = 0
    icmp_type: int = -1
    icmp_code: int = -1
    tcp_state: int = _OPEN
    active_start: int = 0
    active_end: int = 0

    # ------------------------------------------------------------------ build

    @classmethod
    def open(cls, p: Packet, like: Flow | None = None) -> Flow:
        """Start a flow with packet p. The sender of the first packet is the flow's
        source, unless `like` is given: then the new flow keeps that flow's direction."""
        if like is not None:
            ends = (like.src_ip, like.dst_ip, like.src_port, like.dst_port)
        else:
            ends = (p.src, p.dst, p.sport, p.dport)
        flow = cls(*ends, protocol=p.proto, start=p.ts, last_seen=p.ts)
        flow.active_start = flow.active_end = p.ts
        flow.icmp_type, flow.icmp_code = p.icmp_type, p.icmp_code
        forward = flow._is_forward(p)
        flow._bulk(p, forward)
        flow._count_flags(p, forward)
        if p.flags & RST:
            # CICFlowMeter (dataset build): a reset on a flow's first packet is counted
            # in "RST Flag Count" but not in the per-direction count.
            flow.fwd_rst = flow.bwd_rst = 0
        flow.all_len.add(p.payload)
        if forward:
            flow.fwd_seg_min = p.header
            flow.fwd_init_window = p.window
            flow.fwd_len.add(p.payload)
            flow.fwd_header = p.header
            flow.fwd_last = p.ts
            # CICFlowMeter (dataset build): "Fwd Act Data Pkts" skips the flow's first packet.
        else:
            # CICFlowMeter: when the first packet is a reply, the forward minimum
            # header size and initial window are never set and stay 0.
            flow.bwd_init_window = p.window
            flow.bwd_len.add(p.payload)
            flow.bwd_header = p.header
            flow.bwd_last = p.ts
        return flow

    def add(self, p: Packet) -> None:
        self._note_activity(p.ts)
        forward = self._is_forward(p)
        self._bulk(p, forward)
        self._count_flags(p, forward)
        self.all_len.add(p.payload)
        if forward:
            self.fwd_data_packets += p.payload >= 1
            self.fwd_len.add(p.payload)
            self.fwd_header += p.header
            if self.fwd_len.n > 1:
                self.fwd_iat.add(p.ts - self.fwd_last)
            self.fwd_last = p.ts
            self.fwd_seg_min = min(self.fwd_seg_min, p.header)
        else:
            self.bwd_len.add(p.payload)
            # CICFlowMeter (dataset build): despite the name, this ends up as the
            # window of the last backward packet, not the first.
            self.bwd_init_window = p.window
            self.bwd_header += p.header
            if self.bwd_len.n > 1:
                self.bwd_iat.add(p.ts - self.bwd_last)
            self.bwd_last = p.ts
        self.flow_iat.add(p.ts - self.last_seen)
        self.last_seen = p.ts

    def _is_forward(self, p: Packet) -> bool:
        # CICFlowMeter compares addresses only. The port check matters when both
        # ends share an address (loopback), where it would call every packet forward.
        return p.src == self.src_ip and (p.src != p.dst or p.sport == self.src_port)

    def _note_activity(self, ts: int) -> None:
        if ts - self.active_end > ACTIVITY_TIMEOUT_US:
            if self.active_end > self.active_start:
                self.active.add(self.active_end - self.active_start)
            self.idle.add(ts - self.active_end)
            self.active_start = ts
        # CICFlowMeter: the last active period is never closed, so a flow with
        # no 5 s gap reports all-zero active and idle times.
        self.active_end = ts

    def _bulk(self, p: Packet, forward: bool) -> None:
        if forward:
            self.fwd_bulk.update(p.ts, p.payload, self.bwd_bulk.last)
        else:
            self.bwd_bulk.update(p.ts, p.payload, self.fwd_bulk.last)

    def _count_flags(self, p: Packet, forward: bool) -> None:
        f = p.flags
        if not f:
            return
        counts = self.flag_counts
        for i in range(8):
            if f >> i & 1:
                counts[i] += 1
        if forward:
            self.fwd_psh += f >> 3 & 1
            self.fwd_urg += f >> 5 & 1
            self.fwd_rst += f >> 2 & 1
            self.fwd_fin += f & 1
        else:
            self.bwd_psh += f >> 3 & 1
            self.bwd_urg += f >> 5 & 1
            self.bwd_rst += f >> 2 & 1
            self.bwd_fin += f & 1

    # ------------------------------------------------------------------- read

    @property
    def flow_id(self) -> str:
        return f"{self.src_ip}-{self.dst_ip}-{self.src_port}-{self.dst_port}-{self.protocol}"

    @property
    def duration(self) -> int:
        return self.last_seen - self.start

    @property
    def total_packets(self) -> int:
        return self.all_len.n

    @property
    def total_syn_flags(self) -> int:
        return self.flag_counts[1]

    @property
    def total_rst_flags(self) -> int:
        return self.flag_counts[2]

    def features(self) -> dict[str, float]:
        """The 84 features, named and ordered as in nids.features.FEATURE_NAMES_84.

        Byte and packet rates are NaN for a zero-length flow (CICFlowMeter divides
        by zero there); the model imputes NaN, as it did in training.
        """
        dur = self.duration
        seconds = dur / 1e6
        n_fwd, n_bwd, n_all = self.fwd_len.n, self.bwd_len.n, self.all_len.n
        fwd_bytes, bwd_bytes = self.fwd_len.total, self.bwd_len.total
        fwd_mean, fwd_std, fwd_max, fwd_min = self.fwd_len.summary()
        bwd_mean, bwd_std, bwd_max, bwd_min = self.bwd_len.summary()
        all_mean, all_std, all_max, all_min = self.all_len.summary()
        iat_mean, iat_std, iat_max, iat_min = self.flow_iat.summary()
        fi_mean, fi_std, fi_max, fi_min = self.fwd_iat.summary()
        bi_mean, bi_std, bi_max, bi_min = self.bwd_iat.summary()
        act_mean, act_std, act_max, act_min = self.active.summary()
        idle_mean, idle_std, idle_max, idle_min = self.idle.summary()
        fb_bytes, fb_packets, fb_rate = self.fwd_bulk.features()
        bb_bytes, bb_packets, bb_rate = self.bwd_bulk.features()
        fin, syn, rst, psh, ack, urg, ece, cwr = self.flag_counts
        values = (
            self.src_port, self.dst_port, self.protocol, dur,
            n_fwd, n_bwd, fwd_bytes, bwd_bytes,
            fwd_max, fwd_min, fwd_mean, fwd_std,
            bwd_max, bwd_min, bwd_mean, bwd_std,
            (fwd_bytes + bwd_bytes) / seconds if dur else math.nan,
            n_all / seconds if dur else math.nan,
            iat_mean, iat_std, iat_max, iat_min,
            self.fwd_iat.total, fi_mean, fi_std, fi_max, fi_min,
            self.bwd_iat.total, bi_mean, bi_std, bi_max, bi_min,
            self.fwd_psh, self.bwd_psh, self.fwd_urg, self.bwd_urg,
            self.fwd_rst, self.bwd_rst,
            self.fwd_header, self.bwd_header,
            n_fwd / seconds if dur else 0.0,
            n_bwd / seconds if dur else 0.0,
            all_min, all_max, all_mean, all_std, self.all_len.variance,
            fin, syn, rst, psh, ack, urg, cwr, ece,
            n_bwd / n_fwd if n_fwd else 0.0,
            self.all_len.total / n_all if n_all else 0.0,
            fwd_mean, bwd_mean,
            fb_bytes, fb_packets, fb_rate,
            bb_bytes, bb_packets, bb_rate,
            # CICFlowMeter (the build that made the dataset): "subflow" values are
            # the totals integer-divided by the flow's packet count.
            n_fwd // n_all, fwd_bytes // n_all, n_bwd // n_all, bwd_bytes // n_all,
            self.fwd_init_window, self.bwd_init_window,
            self.fwd_data_packets, self.fwd_seg_min,
            act_mean, act_std, act_max, act_min,
            idle_mean, idle_std, idle_max, idle_min,
            self.icmp_code, self.icmp_type,
            self.carried_duration + dur if self.protocol == TCP else 0,
        )
        return dict(zip(FEATURE_NAMES_84, map(float, values), strict=True))


class FlowMeter:
    """Groups packets into flows the way CICFlowMeter does.

    `add(packet)` returns the flows that packet finished (usually none).
    `flush()` returns everything still open, for the end of a capture file.

    A flow ends when a packet arrives more than 120 s after the flow started,
    or when a new SYN arrives on a TCP connection that already closed. That is
    all CICFlowMeter does, and it is enough for a capture file. Live traffic
    also needs flows to end when nothing more arrives, which is what
    `expire(now)` is for:

    - `idle_timeout`: end a flow that has been silent this long. The dataset
      has no such rule, so a flow with a longer pause inside 120 s is reported
      as two flows live but one in the dataset. Higher is more faithful, lower
      detects sooner.
    - `close_grace`: end a closed TCP connection this long after its last packet.
    """

    def __init__(
        self,
        flow_timeout: float = FLOW_TIMEOUT_US / 1e6,
        idle_timeout: float | None = None,
        close_grace: float | None = None,
        max_flows: int = 200_000,
    ):
        self.flow_timeout = int(flow_timeout * 1e6)
        self.idle_timeout = None if idle_timeout is None else int(idle_timeout * 1e6)
        self.close_grace = None if close_grace is None else int(close_grace * 1e6)
        self.max_flows = max_flows
        self._flows: dict[tuple, Flow] = {}
        self.packets = 0
        self.flows_started = 0

    def __len__(self) -> int:
        return len(self._flows)

    def add(self, p: Packet) -> list[Flow]:
        self.packets += 1
        flows = self._flows
        key = (p.src, p.sport, p.dst, p.dport, p.proto)
        flow = flows.get(key)
        if flow is None:
            key = (p.dst, p.dport, p.src, p.sport, p.proto)
            flow = flows.get(key)
        if flow is None:
            done = self._evict() if len(flows) >= self.max_flows else []
            # CICFlowMeter (dataset build): the first packet's sender is the source even
            # when that packet is a SYN-ACK (later versions of the tool swap the two).
            flows[(p.src, p.sport, p.dst, p.dport, p.proto)] = Flow.open(p)
            self.flows_started += 1
            return done

        closed_and_reopened = flow.tcp_state == _TERMINATED and p.flags & SYN
        if p.ts - flow.start > self.flow_timeout or closed_and_reopened:
            del flows[key]
            # CICFlowMeter (dataset build): a closed connection reopened by a plain SYN is
            # a new flow, but if the capture shows the SYN-ACK first it is treated as a
            # continuation of the old one (same direction, duration carried over).
            fresh_start = (
                (closed_and_reopened and not p.flags & ACK)
                or (flow.protocol == UDP and p.ts - flow.last_seen > self.flow_timeout)
                or p.proto not in (TCP, UDP)
            )
            if fresh_start:
                new = Flow.open(p)
            else:
                # The same conversation, cut only because it ran past 120 s.
                new = Flow.open(p, like=flow)
                new.carried_duration = flow.carried_duration + flow.duration + (p.ts - flow.last_seen)
            flows[key] = new
            self.flows_started += 1
            return [flow]

        flags = p.flags
        if flags & FIN:
            flow.add(p)
            if flow.tcp_state == _OPEN:
                flow.tcp_state = _FIRST_FIN
            elif flow.tcp_state == _FIRST_FIN and flow.fwd_fin and flow.bwd_fin:
                flow.tcp_state = _SECOND_FIN
        elif flags & RST:
            flow.add(p)
            flow.tcp_state = _TERMINATED
        elif flags & ACK:
            flow.add(p)
            if flow.tcp_state == _SECOND_FIN:
                flow.tcp_state = _TERMINATED
        elif flow.protocol == ICMP and flow.icmp_code != p.icmp_code and flow.icmp_type != p.icmp_type:
            flows[key] = Flow.open(p)
            self.flows_started += 1
            return [flow]
        else:
            flow.add(p)
        return []

    def expire(self, now: int) -> list[Flow]:
        """Flows that should be scored now even though no packet has ended them. `now` is microseconds."""
        done = []
        for key, flow in list(self._flows.items()):
            quiet = now - flow.last_seen
            if (
                now - flow.start > self.flow_timeout
                or (self.idle_timeout is not None and quiet > self.idle_timeout)
                or (self.close_grace is not None and flow.tcp_state == _TERMINATED and quiet > self.close_grace)
            ):
                del self._flows[key]
                done.append(flow)
        return done

    def flush(self) -> list[Flow]:
        done = list(self._flows.values())
        self._flows.clear()
        return done

    def _evict(self) -> list[Flow]:
        """Table full: give up the flow that has been quiet longest."""
        key = min(self._flows, key=lambda k: self._flows[k].last_seen)
        return [self._flows.pop(key)]


def in_time_order(packets: Iterator[Packet], window: int = 4096) -> Iterator[Packet]:
    """Fix small timestamp inversions in a capture (packets up to `window` places out of order).

    Capture files from mirrored ports often have neighbouring packets swapped by a
    few microseconds, which would give negative gaps between packets. The dataset
    was built from time-sorted captures.
    """
    heap: list[tuple[int, int, Packet]] = []
    for i, p in enumerate(packets):
        heapq.heappush(heap, (p.ts, i, p))
        if len(heap) > window:
            yield heapq.heappop(heap)[2]
    while heap:
        yield heapq.heappop(heap)[2]


def flows_from_packets(packets: Iterator[Packet], **meter_options) -> Iterator[Flow]:
    """All flows in a packet stream, in the order they finish."""
    meter = FlowMeter(**meter_options)
    live = meter.idle_timeout is not None or meter.close_grace is not None
    next_sweep = 0
    for p in packets:
        yield from meter.add(p)
        if live and p.ts >= next_sweep:  # use packet time as the clock
            yield from meter.expire(p.ts)
            next_sweep = p.ts + 1_000_000
    yield from meter.flush()
