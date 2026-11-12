"""The flow builder must reproduce CICFlowMeter's definitions: these pin down each one."""

import math

import pytest

from nids.engine.flowmeter import ACK, FIN, PSH, RST, SYN, FlowMeter, flows_from_packets, in_time_order
from nids.features import FEATURE_NAMES_84

from .packets import CLIENT, SECOND, SERVER, http_exchange, icmp, syn_probe, tcp, udp


def one_flow(packets, **options):
    flows = list(flows_from_packets(iter(packets), **options))
    assert len(flows) == 1
    return flows[0]


def test_features_are_the_84_names_in_order():
    features = one_flow(http_exchange()).features()
    assert list(features) == FEATURE_NAMES_84
    assert all(isinstance(v, float) for v in features.values())


def test_counts_and_lengths_use_payload_bytes_not_frame_size():
    f = one_flow(http_exchange()).features()
    assert (f["Total Fwd Packet"], f["Total Bwd packets"]) == (6, 4)
    assert f["Total Length of Fwd Packet"] == 100
    assert f["Total Length of Bwd Packet"] == 1500
    assert (f["Fwd Packet Length Max"], f["Fwd Packet Length Min"]) == (100, 0)
    assert f["Bwd Packet Length Mean"] == 375
    assert f["Packet Length Max"] == 1000
    assert f["Average Packet Size"] == 160
    assert f["Down/Up Ratio"] == pytest.approx(4 / 6)


def test_header_features_use_transport_header_only():
    f = one_flow(http_exchange()).features()
    assert f["Fwd Header Length"] == 40 + 5 * 20
    assert f["Bwd Header Length"] == 40 + 3 * 20
    assert f["Fwd Seg Size Min"] == 20  # the smallest header, despite the name


def test_flag_counts():
    f = one_flow(http_exchange()).features()
    assert (f["SYN Flag Count"], f["FIN Flag Count"], f["RST Flag Count"]) == (2, 2, 0)
    assert f["ACK Flag Count"] == 9
    assert (f["Fwd PSH Flags"], f["Bwd PSH Flags"], f["PSH Flag Count"]) == (1, 1, 2)


def test_duration_and_rates_are_in_microseconds_and_per_second():
    f = one_flow(http_exchange()).features()
    assert f["Flow Duration"] == 90_000
    assert f["Flow Packets/s"] == pytest.approx(10 / 0.09)
    assert f["Flow Bytes/s"] == pytest.approx(1600 / 0.09)
    assert f["Flow IAT Mean"] == pytest.approx(10_000)
    assert f["Fwd IAT Total"] == 90_000
    assert f["Total TCP Flow Time"] == 90_000


def test_standard_deviation_is_the_sample_one():
    f = one_flow([udp(0, 48), udp(1, 50)]).features()
    assert f["Fwd Packet Length Std"] == pytest.approx(math.sqrt(2))


def test_dataset_quirks():
    f = one_flow(http_exchange()).features()
    assert f["Fwd Act Data Pkts"] == 1  # data packets, not counting the flow's first packet
    assert f["FWD Init Win Bytes"] == 64240
    assert f["Bwd Init Win Bytes"] == 502  # the last backward window, not the first
    # "subflow" values are totals integer-divided by the packet count
    assert (f["Subflow Fwd Packets"], f["Subflow Fwd Bytes"], f["Subflow Bwd Bytes"]) == (0, 10, 150)
    assert (f["ICMP Code"], f["ICMP Type"]) == (-1, -1)


def test_first_packet_with_data_is_not_an_active_data_packet():
    assert one_flow([udp(0), udp(1), udp(2)]).features()["Fwd Act Data Pkts"] == 2


def test_idle_needs_a_five_second_gap():
    quiet = one_flow([udp(0), udp(4), udp(8)]).features()
    assert quiet["Idle Max"] == 0 and quiet["Active Max"] == 0
    f = one_flow([udp(0), udp(1), udp(7), udp(8)]).features()
    assert (f["Idle Mean"], f["Idle Min"], f["Idle Max"]) == (6 * SECOND,) * 3
    assert f["Active Mean"] == 1 * SECOND  # the period before the gap; the last one is never closed


def test_zero_duration_flow_has_undefined_rates():
    f = one_flow([udp(1), udp(1)]).features()
    assert math.isnan(f["Flow Bytes/s"]) and math.isnan(f["Flow Packets/s"])
    assert f["Fwd Packets/s"] == 0


def test_udp_and_icmp_headers():
    assert one_flow([udp(0), udp(1, reply=True)]).features()["Fwd Header Length"] == 8
    f = one_flow([icmp(0, 3, 1), icmp(1, 3, 1)]).features()
    assert (f["ICMP Type"], f["ICMP Code"], f["Fwd Header Length"], f["Total Length of Fwd Packet"]) == (3, 1, 0, 0)
    assert f["Total TCP Flow Time"] == 0


def test_bulk_needs_four_data_packets_in_a_row():
    burst = [tcp(0, SYN)] + [tcp(0.1 * i, ACK, 1000) for i in range(1, 6)]
    f = one_flow(burst).features()
    assert f["Fwd Bytes/Bulk Avg"] == 5000
    assert f["Fwd Packet/Bulk Avg"] == 5
    assert f["Fwd Bulk Rate Avg"] == 12_500  # 5000 bytes over 0.4 s
    interrupted = [tcp(0, SYN), tcp(0.1, ACK, 1000), tcp(0.2, ACK, 1000), tcp(0.25, ACK, 10, reply=True),
                   tcp(0.3, ACK, 1000), tcp(0.4, ACK, 1000)]
    assert one_flow(interrupted).features()["Fwd Bytes/Bulk Avg"] == 0


def test_first_packet_sender_is_the_source_even_for_a_syn_ack():
    flow = one_flow([tcp(0, SYN | ACK, reply=True), tcp(0.001, SYN), tcp(0.002, ACK)])
    assert (flow.src_ip, flow.dst_ip) == (SERVER, CLIENT)
    assert flow.features()["Fwd Seg Size Min"] == 20


def test_flow_is_cut_120_seconds_after_it_started():
    packets = [tcp(0, SYN, header=32)] + [tcp(t, ACK, 10, reply=t % 20 == 0) for t in range(10, 200, 10)]
    first, second = flows_from_packets(iter(packets))
    assert first.duration == 120 * SECOND
    assert second.start == 130 * SECOND
    assert (second.src_ip, second.src_port) == (CLIENT, 40000)  # keeps the original direction
    assert second.features()["Total TCP Flow Time"] == 190 * SECOND  # counted from the connection's first packet
    assert second.features()["Flow Duration"] == 60 * SECOND


def test_closed_connection_reopened_by_syn_is_a_new_flow():
    first, second = flows_from_packets(iter(http_exchange(0) + http_exchange(5)))
    assert first.total_packets == second.total_packets == 10
    assert second.features()["Total TCP Flow Time"] == second.duration


def test_packets_after_close_stay_in_the_flow_until_a_syn():
    packets = http_exchange() + [tcp(1, ACK), tcp(2, RST)]
    assert one_flow(packets).total_packets == 12


def test_reset_on_first_packet_is_not_counted_per_direction():
    f = one_flow([tcp(0, RST | ACK), tcp(1, RST)]).features()
    assert (f["RST Flag Count"], f["Fwd RST Flags"]) == (2, 1)


def test_udp_conversation_restarts_after_a_long_silence():
    first, second = flows_from_packets(iter([udp(0), udp(1, reply=True), udp(300, reply=True), udp(301)]))
    assert first.total_packets == 2
    assert second.src_ip == SERVER  # a fresh conversation: whoever speaks first is the source


def test_icmp_flow_splits_when_type_and_code_both_change():
    assert len(list(flows_from_packets(iter([icmp(0, 8, 0), icmp(1, 3, 1)])))) == 2
    assert len(list(flows_from_packets(iter([icmp(0, 8, 0), icmp(1, 8, 0)])))) == 1


def test_loopback_direction_uses_ports():
    a = tcp(0, SYN)
    b = tcp(0.01, SYN | ACK, reply=True)
    a.dst = b.src = b.dst = a.src  # both ends on one address
    f = one_flow([a, b]).features()
    assert (f["Total Fwd Packet"], f["Total Bwd packets"]) == (1, 1)


def test_live_expiry_by_idle_and_by_close():
    meter = FlowMeter(idle_timeout=15, close_grace=1)
    for p in http_exchange(0):
        assert meter.add(p) == []
    meter.add(udp(0.5))
    assert meter.expire(int(0.9 * SECOND)) == []
    closed = meter.expire(int(1.2 * SECOND))  # the TCP flow closed at 0.09 s
    assert [f.protocol for f in closed] == [6]
    assert meter.expire(10 * SECOND) == []
    assert [f.protocol for f in meter.expire(16 * SECOND)] == [17]
    assert len(meter) == 0


def test_exact_mode_never_expires_early():
    meter = FlowMeter()
    for p in http_exchange(0):
        meter.add(p)
    assert meter.expire(60 * SECOND) == []
    assert len(meter.expire(121 * SECOND)) == 1


def test_full_table_gives_up_the_quietest_flow():
    meter = FlowMeter(max_flows=2)
    meter.add(udp(0, sport=1))
    meter.add(udp(1, sport=2))
    evicted = meter.add(udp(2, sport=3))
    assert [f.src_port for f in evicted] == [1]
    assert len(meter) == 2


def test_in_time_order_fixes_small_inversions():
    packets = [udp(0.000003), udp(0.000001), udp(0.000002), udp(0.000004)]
    assert [p.ts for p in in_time_order(iter(packets), window=3)] == [1, 2, 3, 4]


def test_probe_flows_carry_the_flags_the_rules_need():
    flow = one_flow(syn_probe(0, 443))
    assert (flow.total_packets, flow.total_syn_flags, flow.total_rst_flags) == (2, 1, 1)
    assert flow.flow_id == f"{CLIENT}-{SERVER}-41000-443-6"
    assert PSH and FIN  # imported for readability of the helpers
