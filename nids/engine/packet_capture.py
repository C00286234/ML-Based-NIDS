"""Live packet capture with Scapy.

Scapy only does the sniffing. Each captured frame is decoded by the same
function that reads capture files (nids.engine.pcap.parse_frame), so live
traffic and files go through identical code.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from nids.engine.flowmeter import Packet
from nids.engine.pcap import LINK_ETHERNET, LINK_LINUX_SLL, LINK_NULL, parse_frame, parse_ipv4

logger = logging.getLogger("nids.capture")

# Scapy's name for the outermost layer -> pcap link type
_LINK_TYPES = {"Ether": LINK_ETHERNET, "Loopback": LINK_NULL, "CookedLinux": LINK_LINUX_SLL}


def to_packet(scapy_packet) -> Packet | None:
    """Convert a sniffed Scapy packet into a flowmeter Packet (None if it is not IPv4 TCP/UDP/ICMP)."""
    ts = int(scapy_packet.time * 1_000_000)
    raw = getattr(scapy_packet, "original", None)
    link = _LINK_TYPES.get(type(scapy_packet).__name__)
    if raw and link is not None:
        return parse_frame(bytes(raw), ts, link)
    ip = scapy_packet.getlayer("IP")
    return parse_ipv4(bytes(ip), ts) if ip is not None else None


class PacketCapture:
    """Sniffs an interface on a background thread and queues decoded packets."""

    def __init__(
        self,
        interface: str | None = None,
        promiscuous: bool = True,
        max_queue_size: int = 50_000,
        bpf_filter: str = "ip",
    ):
        self.interface = interface
        self.promiscuous = promiscuous
        self.bpf_filter = bpf_filter
        self._queue: queue.Queue[Packet] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started: float | None = None
        self.captured = 0
        self.dropped = 0
        self.error: Exception | None = None

    def _on_packet(self, scapy_packet) -> None:
        try:
            packet = to_packet(scapy_packet)
        except Exception as e:  # a malformed frame must never stop the capture
            logger.debug("Could not decode packet: %s", e)
            return
        if packet is None:
            return
        try:
            self._queue.put_nowait(packet)
            self.captured += 1
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        from scapy.all import conf, sniff

        conf.promisc = self.promiscuous
        try:
            sniff(
                iface=self.interface,
                prn=self._on_packet,
                store=False,
                filter=self.bpf_filter,
                stop_filter=lambda _: self._stop.is_set(),
            )
        except PermissionError as e:
            self.error = e
            logger.error("Permission denied: packet capture needs root (sudo).")
        except Exception as e:
            self.error = e
            logger.error("Capture error: %s", e)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Packet capture is already running")
        try:
            import scapy.all  # noqa: F401
        except ImportError as e:
            raise RuntimeError("Live capture needs Scapy: install the 'capture' extra (uv sync --extra capture)") from e
        self._stop.clear()
        self._started = time.time()
        self._thread = threading.Thread(target=self._run, name="packet-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_packets(self, max_count: int = 500) -> list[Packet]:
        packets = []
        try:
            for _ in range(max_count):
                packets.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return packets

    def get_statistics(self) -> dict[str, Any]:
        elapsed = time.time() - self._started if self._started else 0.0
        return {
            "interface": self.interface,
            "packets_captured": self.captured,
            "packets_dropped": self.dropped,
            "queue_size": self._queue.qsize(),
            "packets_per_second": self.captured / elapsed if elapsed > 0 else 0.0,
        }


def list_interfaces() -> list[str]:
    try:
        from scapy.all import get_if_list
    except ImportError:
        return []
    return list(get_if_list())
