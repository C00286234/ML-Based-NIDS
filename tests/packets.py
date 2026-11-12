"""Helpers for building packets and capture files in tests."""

from __future__ import annotations

import struct

from nids.engine.flowmeter import ACK, FIN, ICMP, PSH, RST, SYN, TCP, UDP, Packet

CLIENT, SERVER = "10.0.0.1", "10.0.0.2"
SECOND = 1_000_000


def tcp(ts: float, flags: int, payload: int = 0, *, reply: bool = False, header: int = 20, window: int = 1000,
        sport: int = 40000, dport: int = 80) -> Packet:
    """A TCP packet from CLIENT to SERVER (or back, with reply=True). ts is in seconds."""
    src, dst, sp, dp = (SERVER, CLIENT, dport, sport) if reply else (CLIENT, SERVER, sport, dport)
    return Packet(int(ts * SECOND), src, dst, sp, dp, TCP, payload, header, flags, window)


def udp(ts: float, payload: int = 30, *, reply: bool = False, sport: int = 5000, dport: int = 53) -> Packet:
    src, dst, sp, dp = (SERVER, CLIENT, dport, sport) if reply else (CLIENT, SERVER, sport, dport)
    return Packet(int(ts * SECOND), src, dst, sp, dp, UDP, payload, 8)


def icmp(ts: float, kind: int = 8, code: int = 0, *, reply: bool = False) -> Packet:
    src, dst = (SERVER, CLIENT) if reply else (CLIENT, SERVER)
    return Packet(int(ts * SECOND), src, dst, 0, 0, ICMP, 0, 0, icmp_type=kind, icmp_code=code)


def http_exchange(start: float = 0.0, sport: int = 40000) -> list[Packet]:
    """A complete connection: handshake, a request, a two-packet response, and a clean close."""
    t = start
    return [
        tcp(t, SYN, header=40, window=64240, sport=sport),
        tcp(t + 0.01, SYN | ACK, reply=True, header=40, window=65535, sport=sport),
        tcp(t + 0.02, ACK, sport=sport),
        tcp(t + 0.03, PSH | ACK, 100, sport=sport),
        tcp(t + 0.04, ACK, 1000, reply=True, window=500, sport=sport),
        tcp(t + 0.05, PSH | ACK, 500, reply=True, window=501, sport=sport),
        tcp(t + 0.06, ACK, sport=sport),
        tcp(t + 0.07, FIN | ACK, sport=sport),
        tcp(t + 0.08, FIN | ACK, reply=True, window=502, sport=sport),
        tcp(t + 0.09, ACK, sport=sport),
    ]


def syn_probe(ts: float, dport: int, sport: int = 41000) -> list[Packet]:
    """What a port scanner produces on a closed port: SYN, then RST from the target."""
    return [tcp(ts, SYN, sport=sport, dport=dport), tcp(ts + 0.001, RST | ACK, reply=True, sport=sport, dport=dport)]


# ---------------------------------------------------------------- raw frames


def _ip(addr: str) -> bytes:
    return bytes(int(x) for x in addr.split("."))


def ipv4(proto: int, src: str, dst: str, transport: bytes, frag: int = 0) -> bytes:
    header = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(transport), 1, frag, 64, proto, 0, _ip(src), _ip(dst))
    return header + transport


def tcp_segment(sport: int, dport: int, flags: int, payload: bytes = b"", options: bytes = b"", window: int = 8192) -> bytes:
    offset = (20 + len(options)) // 4
    return struct.pack("!HHIIHHHH", sport, dport, 1, 2, (offset << 12) | flags, window, 0, 0) + options + payload


def udp_datagram(sport: int, dport: int, payload: bytes = b"") -> bytes:
    return struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload


def ethernet(ip_packet: bytes, vlan: int | None = None, ethertype: int = 0x0800) -> bytes:
    tag = struct.pack("!HH", 0x8100, vlan) if vlan is not None else b""
    return b"\x02" * 6 + b"\x04" * 6 + tag + struct.pack("!H", ethertype) + ip_packet


def write_pcap(path, frames: list[tuple[int, bytes]], linktype: int = 1, endian: str = "<", nanos: bool = False) -> None:
    """frames: (timestamp in microseconds, bytes)."""
    with open(path, "wb") as f:
        f.write(struct.pack(endian + "IHHiIII", 0xA1B23C4D if nanos else 0xA1B2C3D4, 2, 4, 0, 0, 65535, linktype))
        for ts, data in frames:
            frac = ts % SECOND * (1000 if nanos else 1)
            f.write(struct.pack(endian + "IIII", ts // SECOND, frac, len(data), len(data)) + data)


def write_pcapng(path, frames: list[tuple[int, bytes]], linktype: int = 1, tsresol: int | None = None) -> None:
    def block(kind: int, body: bytes) -> bytes:
        body += b"\x00" * (-len(body) % 4)
        length = 12 + len(body)
        return struct.pack("<II", kind, length) + body + struct.pack("<I", length)

    options = struct.pack("<HHB3x", 9, 1, tsresol) + struct.pack("<HH", 0, 0) if tsresol is not None else b""
    per_second = 10 ** (tsresol if tsresol is not None else 6)
    with open(path, "wb") as f:
        f.write(block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1)))
        f.write(block(1, struct.pack("<HHI", linktype, 0, 65535) + options))
        f.write(block(5, b"\x00" * 12))  # an interface-statistics block: must be skipped
        for ts, data in frames:
            ticks = ts * per_second // SECOND
            f.write(block(6, struct.pack("<IIIII", 0, ticks >> 32, ticks & 0xFFFFFFFF, len(data), len(data)) + data))
