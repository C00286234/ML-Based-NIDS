"""Read capture files (pcap and pcapng) and decode frames into flowmeter Packets.

Written with `struct` only, so analysing a capture file needs no extra
packages and runs at several hundred thousand packets per second.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

from nids.engine.flowmeter import ICMP, TCP, UDP, Packet

# Link types (https://www.tcpdump.org/linktypes.html)
LINK_NULL, LINK_ETHERNET, LINK_RAW, LINK_LINUX_SLL, LINK_IPV4 = 0, 1, 101, 113, 228
_ETH_IPV4, _ETH_VLAN, _ETH_QINQ = 0x0800, 0x8100, 0x88A8


def parse_frame(frame: bytes, ts: int, linktype: int = LINK_ETHERNET) -> Packet | None:
    """Decode one captured frame. Returns None for anything that is not IPv4 TCP, UDP or ICMP."""
    if linktype == LINK_ETHERNET:
        if len(frame) < 14:
            return None
        ethertype = (frame[12] << 8) | frame[13]
        offset = 14
        while ethertype in (_ETH_VLAN, _ETH_QINQ) and len(frame) >= offset + 4:
            ethertype = (frame[offset + 2] << 8) | frame[offset + 3]
            offset += 4
        if ethertype != _ETH_IPV4:
            return None
    elif linktype == LINK_LINUX_SLL:
        if len(frame) < 16 or (frame[14] << 8) | frame[15] != _ETH_IPV4:
            return None
        offset = 16
    elif linktype == LINK_NULL:  # BSD/macOS loopback: 4-byte address family, host byte order
        if len(frame) < 4 or 2 not in (frame[0], frame[3]):
            return None
        offset = 4
    elif linktype in (LINK_RAW, LINK_IPV4):
        offset = 0
    else:
        return None
    return parse_ipv4(frame, ts, offset)


def parse_ipv4(buf: bytes, ts: int, offset: int = 0) -> Packet | None:
    """Decode an IPv4 packet starting at `offset` in buf."""
    if len(buf) < offset + 20 or buf[offset] >> 4 != 4:
        return None
    ihl = (buf[offset] & 0x0F) * 4
    total_len, _, frag, _, proto = struct.unpack_from("!HHHBB", buf, offset + 2)
    if frag & 0x1FFF:  # a later fragment has no transport header to read
        return None
    if total_len == 0:  # segmentation offload: the real length is whatever was captured
        total_len = len(buf) - offset
    src = ".".join(map(str, buf[offset + 12 : offset + 16]))
    dst = ".".join(map(str, buf[offset + 16 : offset + 20]))
    l4 = offset + ihl
    if proto == TCP:
        if len(buf) < l4 + 20:
            return None
        sport, dport, _, _, off_flags, window = struct.unpack_from("!HHIIHH", buf, l4)
        header = (off_flags >> 12) * 4
        return Packet(ts, src, dst, sport, dport, TCP, max(total_len - ihl - header, 0), header,
                      off_flags & 0xFF, window)
    if proto == UDP:
        if len(buf) < l4 + 8:
            return None
        sport, dport, length = struct.unpack_from("!HHH", buf, l4)
        return Packet(ts, src, dst, sport, dport, UDP, max(length - 8, 0), 8)
    if proto == ICMP:
        if len(buf) < l4 + 2:
            return None
        # CICFlowMeter gives ICMP flows no ports, no payload and no header bytes.
        return Packet(ts, src, dst, 0, 0, ICMP, 0, 0, icmp_type=buf[l4], icmp_code=buf[l4 + 1])
    return None


def read_frames(path: str | Path) -> Iterator[tuple[int, bytes, int]]:
    """Yield (timestamp in microseconds, frame bytes, link type) for each packet in a capture file."""
    with open(path, "rb") as f:
        magic = f.read(4)
        f.seek(0)
        if magic == b"\x0a\x0d\x0d\x0a":
            yield from _read_pcapng(f)
        elif magic in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"):
            yield from _read_pcap(f)
        else:
            raise ValueError(f"{path} is not a pcap or pcapng file")


def read_packets(path: str | Path) -> Iterator[Packet]:
    """Yield the IPv4 TCP/UDP/ICMP packets in a capture file."""
    for ts, frame, linktype in read_frames(path):
        packet = parse_frame(frame, ts, linktype)
        if packet is not None:
            yield packet


def _read_pcap(f) -> Iterator[tuple[int, bytes, int]]:
    header = f.read(24)
    magic = header[:4]
    endian = "<" if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
    nanos = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    linktype = struct.unpack(endian + "I", header[20:24])[0] & 0xFFFF
    record = struct.Struct(endian + "IIII")
    while True:
        head = f.read(16)
        if len(head) < 16:
            return
        sec, frac, captured, _ = record.unpack(head)
        frame = f.read(captured)
        if len(frame) < captured:  # file cut off mid-packet
            return
        yield sec * 1_000_000 + (frac // 1000 if nanos else frac), frame, linktype


def _read_pcapng(f) -> Iterator[tuple[int, bytes, int]]:
    endian = "<"
    interfaces: list[tuple[int, int]] = []  # (link type, timestamp units per second)
    while True:
        head = f.read(8)
        if len(head) < 8:
            return
        if head[:4] == b"\x0a\x0d\x0d\x0a":  # section header: sets byte order, resets interfaces
            bom = f.read(4)
            endian = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            length = struct.unpack(endian + "I", head[4:])[0]
            f.seek(length - 12, 1)
            interfaces = []
            continue
        kind, length = struct.unpack(endian + "II", head)
        body = f.read(length - 8)
        if len(body) < length - 8:
            return
        if kind == 1:  # interface description
            linktype = struct.unpack_from(endian + "H", body)[0]
            interfaces.append((linktype, _ts_resolution(body[8:-4], endian)))
        elif kind == 6:  # enhanced packet
            iface, high, low, captured = struct.unpack_from(endian + "IIII", body)
            linktype, per_second = interfaces[iface] if iface < len(interfaces) else (LINK_ETHERNET, 1_000_000)
            ticks = (high << 32) | low
            ts = ticks if per_second == 1_000_000 else ticks * 1_000_000 // per_second
            yield ts, body[20 : 20 + captured], linktype


def _ts_resolution(options: bytes, endian: str) -> int:
    """Timestamp units per second from an interface block's options (default: microseconds)."""
    pos = 0
    while pos + 4 <= len(options):
        code, length = struct.unpack_from(endian + "HH", options, pos)
        if code == 0:
            break
        if code == 9 and length >= 1:  # if_tsresol
            value = options[pos + 4]
            return 2 ** (value & 0x7F) if value & 0x80 else 10**value
        pos += 4 + (length + 3) // 4 * 4
    return 1_000_000
