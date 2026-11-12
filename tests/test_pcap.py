"""Reading capture files and decoding frames."""

import pytest

from nids.engine.flowmeter import ACK, ICMP, PSH, SYN, TCP, UDP
from nids.engine.pcap import (
    LINK_LINUX_SLL,
    LINK_NULL,
    LINK_RAW,
    parse_frame,
    parse_ipv4,
    read_frames,
    read_packets,
)

from .packets import ethernet, ipv4, tcp_segment, udp_datagram, write_pcap, write_pcapng

A, B = "192.168.1.10", "93.184.216.34"
SYN_FRAME = ethernet(ipv4(6, A, B, tcp_segment(50000, 443, SYN, options=b"\x02\x04\x05\xb4" + b"\x01" * 8, window=64240)))
DATA_FRAME = ethernet(ipv4(6, A, B, tcp_segment(50000, 443, PSH | ACK, b"x" * 120)))
DNS_FRAME = ethernet(ipv4(17, A, "8.8.8.8", udp_datagram(5353, 53, b"q" * 33)))


def test_tcp_frame():
    p = parse_frame(SYN_FRAME, 1_500_000)
    assert (p.src, p.dst, p.sport, p.dport, p.proto) == (A, B, 50000, 443, TCP)
    assert (p.payload, p.header, p.flags, p.window, p.ts) == (0, 32, SYN, 64240, 1_500_000)
    assert (p.icmp_type, p.icmp_code) == (-1, -1)
    assert parse_frame(DATA_FRAME, 0).payload == 120


def test_ethernet_padding_is_not_payload():
    padded = ethernet(ipv4(6, A, B, tcp_segment(1, 2, ACK))) + b"\x00" * 6
    assert parse_frame(padded, 0).payload == 0


def test_udp_and_icmp_frames():
    p = parse_frame(DNS_FRAME, 0)
    assert (p.proto, p.sport, p.dport, p.payload, p.header) == (UDP, 5353, 53, 33, 8)
    q = parse_frame(ethernet(ipv4(1, A, B, bytes([3, 1, 0, 0]) + b"\x00" * 28)), 0)
    assert (q.proto, q.icmp_type, q.icmp_code, q.sport, q.payload, q.header) == (ICMP, 3, 1, 0, 0, 0)


def test_vlan_tagged_frame():
    tagged = ethernet(ipv4(6, A, B, tcp_segment(50000, 443, SYN)), vlan=100)
    assert parse_frame(tagged, 0).dport == 443


def test_segmentation_offload_frame_uses_captured_length():
    frame = bytearray(DATA_FRAME)
    frame[16:18] = b"\x00\x00"  # IP total length 0, as written by a NIC doing the segmenting
    assert parse_frame(bytes(frame), 0).payload == 120


@pytest.mark.parametrize("frame", [
    b"",
    b"\x00" * 10,
    ethernet(b"\x00" * 28, ethertype=0x0806),  # ARP
    ethernet(b"\x60" + b"\x00" * 60, ethertype=0x86DD),  # IPv6
    ethernet(ipv4(47, A, B, b"\x00" * 20)),  # GRE
    ethernet(ipv4(6, A, B, tcp_segment(1, 2, ACK), frag=100)),  # a later fragment
    ethernet(ipv4(6, A, B, b"\x00" * 8)),  # truncated TCP header
])
def test_frames_we_do_not_score(frame):
    assert parse_frame(frame, 0) is None


def test_other_link_types():
    ip = ipv4(17, A, B, udp_datagram(1, 2, b"hi"))
    assert parse_frame(ip, 0, LINK_RAW).payload == 2
    assert parse_frame(b"\x02\x00\x00\x00" + ip, 0, LINK_NULL).payload == 2
    assert parse_frame(b"\x00" * 14 + b"\x08\x00" + ip, 0, LINK_LINUX_SLL).payload == 2
    assert parse_frame(ip, 0, 999) is None
    assert parse_ipv4(b"\x45" + b"\x00" * 5, 0) is None


FRAMES = [(1_700_000_000_000_001, SYN_FRAME), (1_700_000_000_250_000, DATA_FRAME), (1_700_000_001_000_000, DNS_FRAME)]


@pytest.mark.parametrize("endian,nanos", [("<", False), (">", False), ("<", True)])
def test_read_pcap(tmp_path, endian, nanos):
    path = tmp_path / "t.pcap"
    write_pcap(path, FRAMES, endian=endian, nanos=nanos)
    assert [(ts, frame) for ts, frame, _ in read_frames(path)] == FRAMES
    assert [p.proto for p in read_packets(path)] == [TCP, TCP, UDP]


@pytest.mark.parametrize("tsresol", [None, 6, 9, 3])
def test_read_pcapng(tmp_path, tsresol):
    path = tmp_path / "t.pcapng"
    frames = [(ts // 1000 * 1000, frame) for ts, frame in FRAMES] if tsresol == 3 else FRAMES
    write_pcapng(path, frames, tsresol=tsresol)
    assert [(ts, frame) for ts, frame, _ in read_frames(path)] == frames


def test_truncated_file_stops_cleanly(tmp_path):
    path = tmp_path / "t.pcap"
    write_pcap(path, FRAMES)
    path.write_bytes(path.read_bytes()[:-10])
    assert len(list(read_frames(path))) == 2


def test_not_a_capture_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello world")
    with pytest.raises(ValueError, match="not a pcap"):
        list(read_frames(path))
