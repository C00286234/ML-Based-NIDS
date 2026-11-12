"""The capture engine, the cross-flow rules and capture-file analysis."""

import csv
import json
from pathlib import Path

import pytest

from nids import paths
from nids.engine import engine as engine_module
from nids.engine.analyze import analyze
from nids.engine.engine import Engine, load_config
from nids.engine.flowmeter import flows_from_packets, in_time_order
from nids.engine.heuristics import PortScanDetector, SynFloodDetector
from nids.engine.pcap import read_packets
from nids.features import FEATURE_NAMES_84

from .packets import ethernet, http_exchange, ipv4, syn_probe, tcp_segment, udp

SAMPLE = Path(__file__).parent / "data" / "parity_sample.pcap"


def flows_of(packets):
    return list(flows_from_packets(iter(packets)))


# ------------------------------------------------------------------- rules


def test_port_scan_rule_fires_on_many_empty_probes():
    rule = PortScanDetector(window_seconds=30, threshold=15)
    probes = flows_of([p for port in range(1, 21) for p in syn_probe(port * 0.1, dport=port, sport=41000 + port)])
    fired = [rule.check(f) for f in probes]
    assert fired.index(True) == 14 and all(fired[14:])


def test_port_scan_rule_ignores_normal_traffic():
    rule = PortScanDetector(threshold=5)
    web = flows_of([p for i in range(20) for p in http_exchange(i, sport=42000 + i)])
    dns = flows_of([udp(i, sport=6000 + i, dport=53 + i) for i in range(20) for _ in range(2)])
    assert not any(rule.check(f) for f in web + dns)


def test_port_scan_rule_forgets_old_probes():
    rule = PortScanDetector(window_seconds=30, threshold=15)
    slow = flows_of([p for port in range(1, 31) for p in syn_probe(port * 10, dport=port, sport=41000 + port)])
    assert not any(rule.check(f) for f in slow)


def test_syn_flood_rule_counts_resets_to_one_service():
    rule = SynFloodDetector(window_seconds=10, threshold=20)
    flood = flows_of([p for i in range(25) for p in syn_probe(i * 0.01, dport=80, sport=43000 + i)])
    assert [rule.check(f) for f in flood].count(True) == 6
    assert not any(SynFloodDetector().check(f) for f in flows_of(http_exchange()))


# ------------------------------------------------------------------ engine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    eng = Engine(load_config(None), quiet=True)
    eng.sent = []

    def fake_predict(flow, explain=True):
        eng.sent.append(flow)
        scan = flow.total_packets == 2 and flow.total_rst_flags
        return {"prediction": "Portscan" if scan else "BENIGN", "confidence": 0.95 if scan else 0.99,
                "is_attack": bool(scan), "top_features": []}

    monkeypatch.setattr(eng, "_predict", fake_predict)
    return eng


def test_engine_scores_flows_and_logs_attacks(engine):
    packets = http_exchange(0) + [p for port in range(1, 31) for p in syn_probe(1 + port * 0.01, dport=port, sport=41000 + port)]
    packets.append(udp(5, sport=7000))  # a lone packet: counted, never sent to the model
    engine.run_packets(iter(packets))
    summary = engine.close()

    assert summary["flows"] == 32
    assert summary["single_packet_flows"] == 1
    assert summary["predictions"] == len(engine.sent) == 31
    assert summary["attacks"] == 30
    assert summary["rule_alerts"] == 1  # the port-scan rule, once for the source rather than once per flow
    assert summary["attacks_by_type"]["Portscan"] == 30

    log = [json.loads(line) for line in (engine.session_dir / "detections.jsonl").read_text().splitlines()]
    assert len(log) == 31
    assert {entry["source"] for entry in log} == {"model", "rule"}
    assert json.loads((engine.session_dir / "summary.json").read_text())["packets"] == len(packets)


def test_engine_respects_the_alert_threshold(engine, monkeypatch):
    monkeypatch.setattr(engine, "_predict", lambda flow, explain=True: {
        "prediction": "DDoS", "confidence": 0.5, "is_attack": True, "top_features": []})
    engine.run_packets(iter(http_exchange()))
    assert engine.close().get("attacks", 0) == 0


def test_engine_sends_null_for_undefined_rates(engine, monkeypatch):
    posted = {}

    class Response:
        status_code = 200

        def json(self):
            return {"prediction": "BENIGN", "confidence": 1.0, "is_attack": False}

    class Session:
        def post(self, url, json, **kwargs):
            posted.update(url=url, body=json, params=kwargs["params"])
            return Response()

    monkeypatch.undo()  # use the real _predict
    monkeypatch.setattr(paths, "ROOT", engine.session_dir.parents[2])
    monkeypatch.setattr(engine_module.requests, "Session", Session)
    engine.run_packets(iter([udp(1), udp(1)]))  # zero duration: rates are NaN
    engine.close()

    assert posted["url"].endswith("/predict")
    assert set(posted["body"]["features"]) == set(FEATURE_NAMES_84)
    assert posted["body"]["features"]["Flow Bytes/s"] is None
    assert posted["body"]["flow_id"] == "10.0.0.1-10.0.0.2-5000-53-17"


def test_engine_on_real_packets(engine):
    engine.run_packets(in_time_order(read_packets(SAMPLE)))
    summary = engine.close()
    assert summary["packets"] == 3283
    assert summary["predictions"] >= 219


def test_load_config_overlays_file_on_defaults(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("flow:\n  idle_timeout: 60\napi:\n  url: http://api:8000\nalerting:\n  slack:\n    enabled: false\n")
    config = load_config(cfg)
    assert config["flow"]["idle_timeout"] == 60 and config["flow"]["timeout"] == 120.0
    assert config["api"]["url"] == "http://api:8000" and config["api"]["workers"] == 8
    assert "slack" in config["alerting"]
    assert load_config(tmp_path / "missing.yaml")["flow"]["timeout"] == 120.0


def test_shipped_config_matches_the_training_data():
    config = load_config(paths.CONFIG_PATH)
    assert config["flow"]["timeout"] == 120.0  # anything else builds flows the model was not trained on


# ---------------------------------------------------------------- analysis


def test_analyze_capture_file(tiny_model, tmp_path):
    out = tmp_path / "flows.csv"
    report = analyze(SAMPLE, model=tiny_model, out=out)
    assert report.packets == 3283
    assert report.flows == sum(report.predictions.values()) >= 219
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == report.flows
    assert rows[0]["start_utc"].startswith("2017-07-0")
    summary = report.to_dict()
    assert summary["attack_flows"] == report.attacks
    assert report.path in report.text()


def test_analyze_min_confidence_turns_weak_attacks_benign(tiny_model):
    assert analyze(SAMPLE, model=tiny_model, min_confidence=1.01).attacks == 0


def test_analyze_live_timeouts_cut_more_flows(tiny_model):
    exact = analyze(SAMPLE, model=tiny_model)
    live = analyze(SAMPLE, model=tiny_model, idle_timeout=5.0, close_grace=1.0)
    assert live.flows + live.single_packet_flows > exact.flows + exact.single_packet_flows


# ----------------------------------------------------------------- capture


def test_scapy_packets_decode_like_capture_files():
    scapy = pytest.importorskip("scapy.all")
    from nids.engine.packet_capture import to_packet
    from nids.engine.pcap import parse_frame

    frame = ethernet(ipv4(6, "10.1.1.1", "10.2.2.2", tcp_segment(1234, 443, 0x18, b"hello", window=777)))
    sniffed = scapy.Ether(frame)
    sniffed.time = 12.5
    ours = to_packet(sniffed)
    assert ours == parse_frame(frame, 12_500_000)
    assert (ours.payload, ours.window, ours.dport) == (5, 777, 443)

    no_link_layer = scapy.IP(frame[14:])  # e.g. a tunnel interface: falls back to the IP layer
    no_link_layer.time = 1
    assert to_packet(no_link_layer).payload == 5
    assert to_packet(scapy.Ether(ethernet(b"\x00" * 28, ethertype=0x0806))) is None
