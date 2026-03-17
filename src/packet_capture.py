"""
Packet Capture Module for NIDS
==============================
Captures network packets using Scapy and stores them in a thread-safe queue.

This module provides low-level packet capture functionality that feeds into
the flow aggregation pipeline.

Author: C00286234
"""

# --- Imports and dependencies: standard library, Scapy (with graceful fallback) ---
import threading
import queue
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("WARNING: Scapy not installed. Run: pip install scapy")

logger = logging.getLogger("nids.capture")


# --- CapturedPacket dataclass: structured representation of a single captured packet ---
@dataclass
class CapturedPacket:
    """
    Represents a captured network packet with extracted metadata.

    Attributes:
        timestamp: When the packet was captured
        src_ip: Source IP address
        dst_ip: Destination IP address
        src_port: Source port (0 for ICMP)
        dst_port: Destination port (0 for ICMP)
        protocol: Protocol number (6=TCP, 17=UDP, 1=ICMP)
        length: Total packet length in bytes
        payload_length: Length of payload (excluding headers)
        tcp_flags: Dictionary of TCP flags (if TCP packet)
        header_length: Length of IP + transport headers
        window_size: TCP window size (if TCP)
        icmp_type: ICMP type (if ICMP packet)
        icmp_code: ICMP code (if ICMP packet)
    """
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    length: int
    payload_length: int = 0
    tcp_flags: Dict[str, int] = field(default_factory=dict)
    header_length: int = 0
    window_size: int = 0
    icmp_type: int = 0
    icmp_code: int = 0


class PacketCapture:
    """
    Network packet capture using Scapy.

    Captures packets from a specified network interface and stores them
    in a thread-safe queue for processing by the flow aggregator.

    Example:
        capture = PacketCapture(interface="Ethernet", promiscuous=True)
        capture.start()
        time.sleep(10)
        packets = capture.get_packets(max_count=100)
        capture.stop()
    """

    # --- Initialisation: configure interface, queue, and capture thread controls ---
    def __init__(
        self,
        interface: Optional[str] = None,
        promiscuous: bool = True,
        max_queue_size: int = 50000,
        bpf_filter: str = "ip"
    ):
        """
        Initialize the packet capture.

        Args:
            interface: Network interface name (None for default)
            promiscuous: Enable promiscuous mode to capture all traffic
            max_queue_size: Maximum packets to buffer (oldest dropped when full)
            bpf_filter: Berkeley Packet Filter expression (default: only IP packets)
        """
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is not installed. Run: pip install scapy")

        self.interface = interface
        self.promiscuous = promiscuous
        self.bpf_filter = bpf_filter
        self.max_queue_size = max_queue_size

        # Thread-safe packet queue
        self._packet_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)

        # Capture thread control
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False

        # Statistics
        self._packets_captured = 0
        self._packets_dropped = 0
        self._start_time: Optional[float] = None

        logger.info(f"PacketCapture initialized: interface={interface}, promiscuous={promiscuous}")

    # --- Packet callback: extract IP/TCP/UDP/ICMP fields and enqueue the packet ---
    def _packet_callback(self, packet) -> None:
        """
        Callback function called by Scapy for each captured packet.

        Extracts relevant information and adds to the queue.
        """
        try:
            # Only process IP packets
            if not packet.haslayer(IP):
                return

            ip_layer = packet[IP]

            # Extract basic info
            captured = CapturedPacket(
                timestamp=float(packet.time),
                src_ip=ip_layer.src,
                dst_ip=ip_layer.dst,
                src_port=0,
                dst_port=0,
                protocol=ip_layer.proto,
                length=len(packet),
                payload_length=0,
                header_length=ip_layer.ihl * 4  # IP header length
            )

            # Extract TCP information
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                captured.src_port = tcp_layer.sport
                captured.dst_port = tcp_layer.dport
                captured.header_length += tcp_layer.dataofs * 4  # TCP header length
                captured.window_size = tcp_layer.window

                # Extract TCP flags
                flags = tcp_layer.flags
                captured.tcp_flags = {
                    'FIN': 1 if flags.F else 0,
                    'SYN': 1 if flags.S else 0,
                    'RST': 1 if flags.R else 0,
                    'PSH': 1 if flags.P else 0,
                    'ACK': 1 if flags.A else 0,
                    'URG': 1 if flags.U else 0,
                    'ECE': 1 if flags.E else 0,
                    'CWR': 1 if flags.C else 0
                }

                # Calculate payload length
                if tcp_layer.payload:
                    captured.payload_length = len(tcp_layer.payload)

            # Extract UDP information
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                captured.src_port = udp_layer.sport
                captured.dst_port = udp_layer.dport
                captured.header_length += 8  # UDP header is always 8 bytes

                if udp_layer.payload:
                    captured.payload_length = len(udp_layer.payload)

            # Extract ICMP information
            elif packet.haslayer(ICMP):
                icmp_layer = packet[ICMP]
                captured.icmp_type = icmp_layer.type
                captured.icmp_code = icmp_layer.code
                captured.header_length += 8  # ICMP header

            # Try to add to queue
            try:
                self._packet_queue.put_nowait(captured)
                self._packets_captured += 1
            except queue.Full:
                # Queue full - drop oldest packet and add new one
                try:
                    self._packet_queue.get_nowait()
                    self._packet_queue.put_nowait(captured)
                    self._packets_dropped += 1
                except queue.Empty:
                    pass

        except Exception as e:
            logger.debug(f"Error processing packet: {e}")

    # --- Capture loop: Scapy sniff running in a background thread ---
    def _capture_loop(self) -> None:
        """
        Main capture loop running in background thread.
        """
        logger.info(f"Starting packet capture on interface: {self.interface or 'default'}")

        try:
            # Configure Scapy
            if self.promiscuous:
                conf.promisc = True

            # Start sniffing - this blocks until stop_filter returns True
            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                store=False,  # Don't store packets in memory
                filter=self.bpf_filter,
                stop_filter=lambda x: self._stop_event.is_set()
            )
        except PermissionError:
            logger.error("Permission denied. Run as Administrator/root for packet capture.")
        except Exception as e:
            logger.error(f"Capture error: {e}")
        finally:
            logger.info("Packet capture stopped")

    # --- Start/stop control: launch and gracefully terminate the capture thread ---
    def start(self) -> None:
        """
        Start packet capture in background thread.

        Raises:
            RuntimeError: If capture is already running
        """
        if self._is_running:
            raise RuntimeError("Packet capture is already running")

        self._stop_event.clear()
        self._start_time = time.time()
        self._packets_captured = 0
        self._packets_dropped = 0

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="PacketCaptureThread",
            daemon=True
        )
        self._capture_thread.start()
        self._is_running = True

        logger.info("Packet capture started")

    def stop(self) -> None:
        """
        Stop packet capture gracefully.
        """
        if not self._is_running:
            return

        logger.info("Stopping packet capture...")
        self._stop_event.set()

        # Wait for thread to finish (with timeout)
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)

        self._is_running = False
        logger.info(f"Capture stopped. Total packets: {self._packets_captured}, Dropped: {self._packets_dropped}")

    # --- Packet retrieval: drain packets from the thread-safe queue ---
    def get_packets(self, max_count: int = 100) -> List[CapturedPacket]:
        """
        Retrieve captured packets from the queue.

        Args:
            max_count: Maximum number of packets to retrieve

        Returns:
            List of CapturedPacket objects
        """
        packets = []

        for _ in range(max_count):
            try:
                packet = self._packet_queue.get_nowait()
                packets.append(packet)
            except queue.Empty:
                break

        return packets

    def get_queue_size(self) -> int:
        """Get current number of packets waiting in queue."""
        return self._packet_queue.qsize()

    # --- Statistics: capture counts, drop rate, and throughput ---
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get capture statistics.

        Returns:
            Dictionary with capture statistics
        """
        elapsed = time.time() - self._start_time if self._start_time else 0

        return {
            "is_running": self._is_running,
            "interface": self.interface,
            "packets_captured": self._packets_captured,
            "packets_dropped": self._packets_dropped,
            "queue_size": self._packet_queue.qsize(),
            "elapsed_seconds": elapsed,
            "packets_per_second": self._packets_captured / elapsed if elapsed > 0 else 0
        }

    @property
    def is_running(self) -> bool:
        """Check if capture is currently running."""
        return self._is_running


# --- Interface listing: enumerate available network interfaces (cross-platform) ---
def list_interfaces() -> List[str]:
    """
    List available network interfaces.

    Returns:
        List of interface names
    """
    if not SCAPY_AVAILABLE:
        return []

    try:
        from scapy.arch.windows import get_windows_if_list
        interfaces = get_windows_if_list()
        return [iface['name'] for iface in interfaces]
    except ImportError:
        # Non-Windows system
        from scapy.all import get_if_list
        return get_if_list()
    except Exception as e:
        logger.error(f"Error listing interfaces: {e}")
        return []


# --- Standalone test: capture packets for manual verification ---
if __name__ == "__main__":
    # Test packet capture
    logging.basicConfig(level=logging.INFO)

    print("Available network interfaces:")
    for i, iface in enumerate(list_interfaces()):
        print(f"  {i}: {iface}")

    print("\nStarting packet capture (Ctrl+C to stop)...")

    capture = PacketCapture(interface=None, promiscuous=True)
    capture.start()

    try:
        while True:
            time.sleep(2)
            packets = capture.get_packets(max_count=50)
            stats = capture.get_statistics()

            print(f"\nCaptured: {stats['packets_captured']}, Queue: {stats['queue_size']}, "
                  f"Rate: {stats['packets_per_second']:.1f} pps")

            for pkt in packets[:5]:  # Show first 5 packets
                print(f"  {pkt.src_ip}:{pkt.src_port} -> {pkt.dst_ip}:{pkt.dst_port} "
                      f"proto={pkt.protocol} len={pkt.length}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.stop()
