"""
Flow Aggregator Module for NIDS
===============================
Groups individual packets into bidirectional network flows.

A flow is defined by the 5-tuple: (src_ip, dst_ip, src_port, dst_port, protocol)
Packets are tracked as either forward (src->dst) or backward (dst->src).
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from nids.engine.packet_capture import CapturedPacket

# Configure logging
logger = logging.getLogger("nids.flow")


@dataclass
class PacketInfo:
    """
    Simplified packet information for flow statistics.
    """
    timestamp: float
    length: int
    payload_length: int
    header_length: int
    tcp_flags: dict[str, int] = field(default_factory=dict)
    window_size: int = 0


@dataclass
class Flow:
    """
    Represents a bidirectional network flow.

    A flow groups all packets between two endpoints, tracking
    forward (src->dst) and backward (dst->src) traffic separately.

    Attributes:
        flow_id: Unique identifier for this flow
        src_ip: Source IP (flow initiator)
        dst_ip: Destination IP
        src_port: Source port
        dst_port: Destination port
        protocol: Protocol number (6=TCP, 17=UDP, 1=ICMP)
        start_time: First packet timestamp
        last_time: Most recent packet timestamp
        fwd_packets: List of forward packets
        bwd_packets: List of backward packets
        is_complete: Whether flow is complete (timeout/FIN)
    """
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    start_time: float
    last_time: float
    fwd_packets: list[PacketInfo] = field(default_factory=list)
    bwd_packets: list[PacketInfo] = field(default_factory=list)
    is_complete: bool = False

    # Aggregated flag counts (across all packets)
    fwd_fin_flags: int = 0
    fwd_syn_flags: int = 0
    fwd_rst_flags: int = 0
    fwd_psh_flags: int = 0
    fwd_ack_flags: int = 0
    fwd_urg_flags: int = 0
    fwd_ece_flags: int = 0
    fwd_cwr_flags: int = 0

    bwd_fin_flags: int = 0
    bwd_syn_flags: int = 0
    bwd_rst_flags: int = 0
    bwd_psh_flags: int = 0
    bwd_ack_flags: int = 0
    bwd_urg_flags: int = 0
    bwd_ece_flags: int = 0
    bwd_cwr_flags: int = 0

    # Initial TCP window sizes
    fwd_init_win_bytes: int = 0
    bwd_init_win_bytes: int = 0

    # ICMP info
    icmp_type: int = 0
    icmp_code: int = 0

    # Activity tracking (for active/idle time calculation)
    active_start: float = 0.0
    active_times: list[float] = field(default_factory=list)
    idle_times: list[float] = field(default_factory=list)
    last_active_time: float = 0.0

    def add_forward_packet(self, packet: PacketInfo) -> None:
        """Add a forward direction packet to the flow."""
        # Track initial window size
        if len(self.fwd_packets) == 0 and packet.window_size > 0:
            self.fwd_init_win_bytes = packet.window_size

        self.fwd_packets.append(packet)
        self._update_flags(packet.tcp_flags, is_forward=True)
        self._update_activity(packet.timestamp)
        self.last_time = packet.timestamp

    def add_backward_packet(self, packet: PacketInfo) -> None:
        """Add a backward direction packet to the flow."""
        # Track initial window size
        if len(self.bwd_packets) == 0 and packet.window_size > 0:
            self.bwd_init_win_bytes = packet.window_size

        self.bwd_packets.append(packet)
        self._update_flags(packet.tcp_flags, is_forward=False)
        self._update_activity(packet.timestamp)
        self.last_time = packet.timestamp

    def _update_flags(self, flags: dict[str, int], is_forward: bool) -> None:
        """Update TCP flag counts."""
        if not flags:
            return

        if is_forward:
            self.fwd_fin_flags += flags.get('FIN', 0)
            self.fwd_syn_flags += flags.get('SYN', 0)
            self.fwd_rst_flags += flags.get('RST', 0)
            self.fwd_psh_flags += flags.get('PSH', 0)
            self.fwd_ack_flags += flags.get('ACK', 0)
            self.fwd_urg_flags += flags.get('URG', 0)
            self.fwd_ece_flags += flags.get('ECE', 0)
            self.fwd_cwr_flags += flags.get('CWR', 0)
        else:
            self.bwd_fin_flags += flags.get('FIN', 0)
            self.bwd_syn_flags += flags.get('SYN', 0)
            self.bwd_rst_flags += flags.get('RST', 0)
            self.bwd_psh_flags += flags.get('PSH', 0)
            self.bwd_ack_flags += flags.get('ACK', 0)
            self.bwd_urg_flags += flags.get('URG', 0)
            self.bwd_ece_flags += flags.get('ECE', 0)
            self.bwd_cwr_flags += flags.get('CWR', 0)

    def _update_activity(self, timestamp: float) -> None:
        """Track active and idle periods for the flow."""
        IDLE_THRESHOLD = 1.0  # 1 second threshold for idle

        if self.last_active_time == 0:
            self.last_active_time = timestamp
            self.active_start = timestamp
            return

        time_diff = timestamp - self.last_active_time

        if time_diff > IDLE_THRESHOLD:
            # End of active period, start of idle
            if self.active_start > 0:
                active_duration = self.last_active_time - self.active_start
                if active_duration > 0:
                    self.active_times.append(active_duration)
            self.idle_times.append(time_diff)
            self.active_start = timestamp

        self.last_active_time = timestamp

    @property
    def duration(self) -> float:
        """Flow duration in microseconds."""
        return (self.last_time - self.start_time) * 1_000_000

    @property
    def total_packets(self) -> int:
        """Total number of packets in flow."""
        return len(self.fwd_packets) + len(self.bwd_packets)

    @property
    def total_fin_flags(self) -> int:
        """Total FIN flags in flow."""
        return self.fwd_fin_flags + self.bwd_fin_flags

    @property
    def total_syn_flags(self) -> int:
        """Total SYN flags in flow."""
        return self.fwd_syn_flags + self.bwd_syn_flags

    @property
    def total_rst_flags(self) -> int:
        """Total RST flags in flow."""
        return self.fwd_rst_flags + self.bwd_rst_flags

    @property
    def total_psh_flags(self) -> int:
        """Total PSH flags in flow."""
        return self.fwd_psh_flags + self.bwd_psh_flags

    @property
    def total_ack_flags(self) -> int:
        """Total ACK flags in flow."""
        return self.fwd_ack_flags + self.bwd_ack_flags

    @property
    def total_urg_flags(self) -> int:
        """Total URG flags in flow."""
        return self.fwd_urg_flags + self.bwd_urg_flags

    @property
    def total_ece_flags(self) -> int:
        """Total ECE flags in flow."""
        return self.fwd_ece_flags + self.bwd_ece_flags

    @property
    def total_cwr_flags(self) -> int:
        """Total CWR flags in flow."""
        return self.fwd_cwr_flags + self.bwd_cwr_flags


class FlowAggregator:
    """
    Aggregates packets into bidirectional network flows.

    Manages active flows, handles timeouts, and provides completed
    flows for feature extraction.

    Example:
        aggregator = FlowAggregator(flow_timeout=120.0)
        for packet in captured_packets:
            aggregator.process_packet(packet)
        completed = aggregator.get_completed_flows()
    """

    def __init__(
        self,
        flow_timeout: float = 120.0,
        max_packets_per_flow: int = 1000,
        max_flows: int = 100000
    ):
        """
        Initialize the flow aggregator.

        Args:
            flow_timeout: Seconds of inactivity before flow expires
            max_packets_per_flow: Maximum packets per flow before completion
            max_flows: Maximum active flows to track
        """
        self.flow_timeout = flow_timeout
        self.max_packets_per_flow = max_packets_per_flow
        self.max_flows = max_flows

        # Active flows: flow_id -> Flow
        self._flows: dict[str, Flow] = {}

        # Completed flows ready for processing
        self._completed_flows: list[Flow] = []

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._total_packets_processed = 0
        self._total_flows_created = 0
        self._total_flows_completed = 0

        logger.info(f"FlowAggregator initialized: timeout={flow_timeout}s, max_packets={max_packets_per_flow}")

    def _create_flow_id(self, packet: CapturedPacket) -> tuple[str, bool]:
        """
        Create a unique flow ID from packet 5-tuple.

        Returns:
            Tuple of (flow_id, is_forward_direction)
        """
        # Create canonical flow ID (sorted by IP to handle bidirectional)
        # Forward: src -> dst
        fwd_id = f"{packet.src_ip}:{packet.src_port}-{packet.dst_ip}:{packet.dst_port}-{packet.protocol}"
        # Backward: dst -> src
        bwd_id = f"{packet.dst_ip}:{packet.dst_port}-{packet.src_ip}:{packet.src_port}-{packet.protocol}"

        # Check if flow exists in either direction
        if fwd_id in self._flows:
            return fwd_id, True
        elif bwd_id in self._flows:
            return bwd_id, False
        else:
            # New flow - use forward direction as canonical
            return fwd_id, True

    def process_packet(self, packet: CapturedPacket) -> str | None:
        """
        Process a captured packet and add to appropriate flow.

        Args:
            packet: The captured packet to process

        Returns:
            Flow ID if packet was processed, None otherwise
        """
        with self._lock:
            self._total_packets_processed += 1

            # Get or create flow
            flow_id, is_forward = self._create_flow_id(packet)

            # Create packet info
            pkt_info = PacketInfo(
                timestamp=packet.timestamp,
                length=packet.length,
                payload_length=packet.payload_length,
                header_length=packet.header_length,
                tcp_flags=packet.tcp_flags,
                window_size=packet.window_size
            )

            if flow_id not in self._flows:
                # Check max flows limit
                if len(self._flows) >= self.max_flows:
                    # Remove oldest flow
                    oldest_id = min(self._flows.keys(),
                                   key=lambda k: self._flows[k].last_time)
                    old_flow = self._flows.pop(oldest_id)
                    old_flow.is_complete = True
                    self._completed_flows.append(old_flow)

                # Create new flow
                flow = Flow(
                    flow_id=flow_id,
                    src_ip=packet.src_ip,
                    dst_ip=packet.dst_ip,
                    src_port=packet.src_port,
                    dst_port=packet.dst_port,
                    protocol=packet.protocol,
                    start_time=packet.timestamp,
                    last_time=packet.timestamp,
                    icmp_type=packet.icmp_type,
                    icmp_code=packet.icmp_code
                )
                self._flows[flow_id] = flow
                self._total_flows_created += 1
            else:
                flow = self._flows[flow_id]

            # Add packet to flow
            if is_forward:
                flow.add_forward_packet(pkt_info)
            else:
                flow.add_backward_packet(pkt_info)

            # Check if flow is complete
            if self._should_complete_flow(flow):
                flow.is_complete = True
                self._completed_flows.append(flow)
                del self._flows[flow_id]
                self._total_flows_completed += 1

            return flow_id

    def _should_complete_flow(self, flow: Flow) -> bool:
        """
        Check if a flow should be marked as complete.

        Conditions:
        - Max packets reached
        - TCP connection closed (FIN or RST from both sides)
        """
        # Max packets reached
        if flow.total_packets >= self.max_packets_per_flow:
            return True

        # TCP connection properly closed
        if flow.protocol == 6:  # TCP
            # Both sides sent FIN
            if flow.fwd_fin_flags > 0 and flow.bwd_fin_flags > 0:
                return True
            # RST received
            if flow.total_rst_flags > 0:
                return True

        return False

    def cleanup_expired_flows(self) -> int:
        """
        Remove flows that have timed out.

        Returns:
            Number of flows cleaned up
        """
        current_time = time.time()
        expired_ids = []

        with self._lock:
            for flow_id, flow in self._flows.items():
                if current_time - flow.last_time > self.flow_timeout:
                    expired_ids.append(flow_id)

            for flow_id in expired_ids:
                flow = self._flows.pop(flow_id)
                flow.is_complete = True
                self._completed_flows.append(flow)
                self._total_flows_completed += 1

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired flows")

        return len(expired_ids)

    def get_completed_flows(self) -> list[Flow]:
        """
        Retrieve and clear all completed flows.

        Returns:
            List of completed Flow objects
        """
        with self._lock:
            flows = self._completed_flows.copy()
            self._completed_flows.clear()
            return flows

    def force_complete_all(self) -> list[Flow]:
        """
        Force complete all active flows and return them.

        Useful for shutdown or periodic processing.

        Returns:
            List of all flows (both active and completed)
        """
        with self._lock:
            all_flows = self._completed_flows.copy()
            self._completed_flows.clear()

            for flow in self._flows.values():
                flow.is_complete = True
                all_flows.append(flow)
                self._total_flows_completed += 1

            self._flows.clear()

            return all_flows

    def get_statistics(self) -> dict[str, Any]:
        """
        Get aggregator statistics.

        Returns:
            Dictionary with aggregator statistics
        """
        with self._lock:
            return {
                "active_flows": len(self._flows),
                "completed_flows_pending": len(self._completed_flows),
                "total_packets_processed": self._total_packets_processed,
                "total_flows_created": self._total_flows_created,
                "total_flows_completed": self._total_flows_completed
            }


if __name__ == "__main__":
    # Test flow aggregator with simulated packets
    logging.basicConfig(level=logging.INFO)

    print("Testing FlowAggregator...")

    aggregator = FlowAggregator(flow_timeout=5.0, max_packets_per_flow=10)

    # Simulate packets for a TCP flow
    base_time = time.time()
    test_packets = [
        CapturedPacket(
            timestamp=base_time + i * 0.1,
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            src_port=45678,
            dst_port=80,
            protocol=6,
            length=100 + i * 10,
            payload_length=50 + i * 5,
            header_length=40,
            tcp_flags={'SYN': 1 if i == 0 else 0, 'ACK': 1 if i > 0 else 0},
            window_size=65535
        )
        for i in range(15)  # 15 packets - should trigger max_packets completion
    ]

    # Process packets
    for pkt in test_packets:
        flow_id = aggregator.process_packet(pkt)
        print(f"Processed packet -> Flow: {flow_id[:30]}...")

    # Get completed flows
    completed = aggregator.get_completed_flows()
    print(f"\nCompleted flows: {len(completed)}")

    for flow in completed:
        print(f"  Flow: {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port}")
        print(f"  Packets: {len(flow.fwd_packets)} fwd, {len(flow.bwd_packets)} bwd")
        print(f"  Duration: {flow.duration:.2f} us")

    # Print statistics
    stats = aggregator.get_statistics()
    print(f"\nStatistics: {stats}")
