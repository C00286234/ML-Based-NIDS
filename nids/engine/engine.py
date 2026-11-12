"""The capture engine: packets -> flows -> features -> prediction API -> alerts.

    sudo nids engine -i en0              # watch a network interface
    nids engine --pcap capture.pcap      # run a capture file through the same pipeline

Every flow is sent to the prediction API, so detections appear on the dashboard.
A capture file is processed using the packets' own timestamps as the clock, so
the result does not depend on how fast the file is read.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

from nids import paths
from nids.engine.alerting import AlertManager
from nids.engine.flowmeter import Flow, FlowMeter, Packet, in_time_order
from nids.engine.heuristics import PortScanDetector, SynFloodDetector
from nids.engine.packet_capture import PacketCapture, list_interfaces
from nids.engine.pcap import read_packets

logger = logging.getLogger("nids.engine")

PROTOCOLS = {1: "ICMP", 6: "TCP", 17: "UDP"}
RULE_ALERT_COOLDOWN = 60.0  # seconds between repeats of the same rule for the same source

DEFAULT_CONFIG: dict[str, Any] = {
    "network": {"interface": None, "promiscuous": True, "bpf_filter": "ip"},
    "flow": {"timeout": 120.0, "idle_timeout": 15.0, "close_grace": 1.0, "max_flows": 200_000},
    "api": {"url": "http://localhost:8000", "timeout": 10.0, "retry_count": 3, "workers": 8, "max_backlog": 5000},
    "alerts": {"confidence_threshold": 0.8},
    "heuristics": {
        "syn_flood_window": 10, "syn_flood_threshold": 20,
        "port_scan_window": 30, "port_scan_threshold": 15,
    },
    "performance": {"packet_batch_size": 500, "stats_interval": 10},
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    """DEFAULT_CONFIG overlaid with the YAML file, section by section."""
    config = {section: dict(values) for section, values in DEFAULT_CONFIG.items()}
    if path and Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        for section, values in loaded.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section].update(values)
            else:
                config[section] = values
    elif path:
        logger.warning("Config file %s not found; using defaults", path)
    return config


def describe(flow: Flow) -> str:
    proto = PROTOCOLS.get(flow.protocol, str(flow.protocol))
    return f"{flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port} ({proto}, {flow.total_packets} pkts)"


class Engine:
    def __init__(self, config: dict[str, Any], quiet: bool = False):
        self.config = config
        self.quiet = quiet
        flow = config["flow"]
        self.meter = FlowMeter(
            flow_timeout=flow["timeout"],
            idle_timeout=flow["idle_timeout"],
            close_grace=flow["close_grace"],
            max_flows=flow["max_flows"],
        )
        rules = config["heuristics"]
        self.rules = [
            SynFloodDetector(rules["syn_flood_window"], rules["syn_flood_threshold"]),
            PortScanDetector(rules["port_scan_window"], rules["port_scan_threshold"]),
        ]
        self._rule_last_fired: dict[tuple[str, str], float] = {}
        self.alerts = AlertManager(config)

        api = config["api"]
        self.api_url = api["url"].rstrip("/")
        self._headers = {"X-API-Key": os.environ["NIDS_API_KEY"]} if os.getenv("NIDS_API_KEY") else {}
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=api["workers"], thread_name_prefix="nids-predict")
        self._backlog = threading.Semaphore(api["max_backlog"])
        self._pending = 0
        self._running = False

        self.stats: Counter[str] = Counter()
        self.attack_counts: Counter[str] = Counter()
        self._lock = threading.Lock()
        self.started = time.time()

        self.session_dir = paths.ROOT / "logs" / "sessions" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._detections = open(self.session_dir / "detections.jsonl", "a", encoding="utf-8")  # noqa: SIM115

    # ------------------------------------------------------------------ sources

    def run_live(self, capture: PacketCapture) -> None:
        """Process packets from a live capture until it stops or stop() is called."""
        perf = self.config["performance"]
        capture.start()
        last_sweep = last_stats = time.time()
        self._running = True
        try:
            while self._running and (capture.is_running or capture.get_statistics()["queue_size"]):
                packets = capture.get_packets(perf["packet_batch_size"])
                for packet in packets:
                    self._on_flows(self.meter.add(packet))
                now = time.time()
                if now - last_sweep >= 1.0:
                    self._on_flows(self.meter.expire(int(now * 1e6)))
                    last_sweep = now
                if now - last_stats >= perf["stats_interval"]:
                    self.print_stats(capture)
                    last_stats = now
                if not packets:
                    time.sleep(0.02)
        finally:
            capture.stop()
            self.stats["packets_dropped"] = capture.dropped

    def run_packets(self, packets: Iterable[Packet], speed: float | None = None) -> None:
        """Process a recorded packet stream. `speed` replays it in (scaled) real time; None = as fast as possible."""
        first_ts = next_sweep = None
        wall_start = time.time()
        self._running = True
        for packet in packets:
            if not self._running:
                break
            if first_ts is None:
                first_ts, next_sweep = packet.ts, packet.ts + 1_000_000
            if speed:
                wait = (packet.ts - first_ts) / 1e6 / speed - (time.time() - wall_start)
                if wait > 0:
                    time.sleep(wait)
            self._on_flows(self.meter.add(packet))
            if packet.ts >= next_sweep:
                self._on_flows(self.meter.expire(packet.ts))
                next_sweep = packet.ts + 1_000_000

    def stop(self) -> None:
        self._running = False

    def close(self) -> dict[str, Any]:
        """Score whatever is still open, wait for the API, and write the session summary."""
        self._on_flows(self.meter.flush())
        self._pool.shutdown(wait=True)
        self._detections.close()
        summary = {
            "started": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "seconds": round(time.time() - self.started, 1),
            "packets": self.meter.packets,
            **dict(self.stats),
            "attacks_by_type": dict(self.attack_counts),
        }
        (self.session_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # ------------------------------------------------------------------- flows

    def _on_flows(self, flows: list[Flow]) -> None:
        for flow in flows:
            self.stats["flows"] += 1
            for rule in self.rules:
                if rule.check(flow):
                    self._rule_fired(rule, flow)
            if flow.total_packets < 2:
                # The dataset has no one-packet flows, so the model has never seen
                # one. The rules above still count them.
                self.stats["single_packet_flows"] += 1
                continue
            if not self._backlog.acquire(blocking=False):
                self.stats["flows_dropped_api_backlog"] += 1
                continue
            with self._lock:
                self._pending += 1
            self._pool.submit(self._score, flow)

    def _score(self, flow: Flow) -> None:
        try:
            # Skip the SHAP explanation when the API is falling behind.
            result = self._predict(flow, explain=self._pending < 200)
            if result is not None:
                self._on_prediction(flow, result)
        except Exception:
            logger.exception("Error scoring flow %s", flow.flow_id)
        finally:
            with self._lock:
                self._pending -= 1
            self._backlog.release()

    def _predict(self, flow: Flow, explain: bool = True) -> dict[str, Any] | None:
        api = self.config["api"]
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = requests.Session()
        # JSON has no NaN: send null and let the model impute, as in training.
        features = {k: (None if math.isnan(v) else v) for k, v in flow.features().items()}
        body = {"features": features, "flow_id": flow.flow_id}
        for attempt in range(api["retry_count"]):
            try:
                r = session.post(f"{self.api_url}/predict", json=body, headers=self._headers,
                                 params={"explain": str(explain).lower()}, timeout=api["timeout"])
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    time.sleep(1.0)
                    continue
                logger.warning("API returned %s: %s", r.status_code, r.text[:120])
                if r.status_code < 500:
                    break
            except requests.RequestException as e:
                logger.warning("API request failed (attempt %d): %s", attempt + 1, e)
                time.sleep(0.5)
        with self._lock:
            self.stats["api_errors"] += 1
        return None

    def _on_prediction(self, flow: Flow, result: dict[str, Any]) -> None:
        label, confidence = result["prediction"], result["confidence"]
        alert = result["is_attack"] and confidence >= self.config["alerts"]["confidence_threshold"]
        with self._lock:
            self.stats["predictions"] += 1
            if alert:
                self.stats["attacks"] += 1
                self.attack_counts[label] += 1
            self._record(flow, label, confidence, "model", result.get("top_features", []))
        if alert:
            self._say(f"  [!] {label:<24} {confidence:>4.0%}  {describe(flow)}")
            self.alerts.send_attack_alert(
                attack_type=label, confidence=confidence, source_ip=flow.src_ip, destination_ip=flow.dst_ip,
                source_port=flow.src_port, destination_port=flow.dst_port, protocol=flow.protocol,
                features=flow.features(), flow_id=flow.flow_id,
            )
        elif logger.isEnabledFor(logging.DEBUG):
            self._say(f"  [ ] {label:<24} {confidence:>4.0%}  {describe(flow)}")

    def _rule_fired(self, rule, flow: Flow) -> None:
        now = flow.last_seen / 1e6
        key = (rule.label, flow.src_ip)
        if now - self._rule_last_fired.get(key, -math.inf) < RULE_ALERT_COOLDOWN:
            return
        self._rule_last_fired[key] = now
        with self._lock:
            self.stats["rule_alerts"] += 1
            self.attack_counts[rule.description] += 1
            self._record(flow, rule.label, 1.0, "rule", [])
        self._say(f"  [!] {rule.description:<24} rule  {describe(flow)}")
        self.alerts.send_attack_alert(
            attack_type=rule.label, confidence=1.0, source_ip=flow.src_ip, destination_ip=flow.dst_ip,
            source_port=flow.src_port, destination_port=flow.dst_port, protocol=flow.protocol, flow_id=flow.flow_id,
        )

    def _record(self, flow: Flow, label: str, confidence: float, source: str, top_features: list) -> None:
        if label == "BENIGN":
            return
        self._detections.write(json.dumps({
            "time_utc": datetime.fromtimestamp(flow.last_seen / 1e6, UTC).isoformat(timespec="milliseconds"),
            "label": label, "confidence": round(confidence, 4), "source": source, "flow_id": flow.flow_id,
            "packets": flow.total_packets, "top_features": top_features,
        }) + "\n")
        self._detections.flush()

    # ------------------------------------------------------------------ output

    def _say(self, line: str) -> None:
        if not self.quiet:
            print(line, flush=True)

    def print_stats(self, capture: PacketCapture | None = None) -> None:
        s = self.stats
        line = (f"-- {self.meter.packets:,} packets | {len(self.meter):,} open flows | {s['flows']:,} finished | "
                f"{s['predictions']:,} scored | {s['attacks']:,} attacks | {s['rule_alerts']:,} rule alerts")
        if capture is not None:
            line += f" | {capture.dropped:,} packets dropped"
        if s["api_errors"] or s["flows_dropped_api_backlog"]:
            line += f" | {s['api_errors']:,} API errors, {s['flows_dropped_api_backlog']:,} flows skipped"
        self._say(line)

    def api_healthy(self) -> bool:
        try:
            r = requests.get(f"{self.api_url}/health", timeout=5)
            return r.status_code == 200 and r.json().get("models_loaded", False)
        except requests.RequestException:
            return False


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="nids engine", description="Capture traffic, build flows and score them with the API")
    p.add_argument("config", nargs="?", default=str(paths.CONFIG_PATH), help="YAML config (default: config/nids_config.yaml)")
    p.add_argument("-i", "--interface", help="network interface to capture from")
    p.add_argument("--pcap", metavar="FILE", help="process a capture file instead of a live interface")
    p.add_argument("--speed", type=float, help="with --pcap: replay at this multiple of real time (default: as fast as possible)")
    p.add_argument("--api-url", help="prediction API (default: from config, http://localhost:8000)")
    p.add_argument("--list-interfaces", action="store_true")
    args = p.parse_args(argv)

    if args.list_interfaces:
        for name in list_interfaces():
            print(name)
        return

    config = load_config(args.config)
    if args.api_url:
        config["api"]["url"] = args.api_url
    if args.pcap and not Path(args.pcap).exists():
        raise SystemExit(f"Capture file not found: {args.pcap}")

    engine = Engine(config)
    if not engine.api_healthy():
        raise SystemExit(f"The prediction API at {engine.api_url} is not reachable. Start it with `nids serve`.")

    try:
        if args.pcap:
            print(f"Processing {args.pcap} -> {engine.api_url}")
            engine.run_packets(in_time_order(read_packets(args.pcap)), speed=args.speed)
        else:
            interface = args.interface or config["network"]["interface"]
            capture = PacketCapture(interface, config["network"]["promiscuous"], bpf_filter=config["network"]["bpf_filter"])
            print(f"Capturing on {interface or 'the default interface'} -> {engine.api_url}. Ctrl+C to stop.")
            engine.run_live(capture)
            if capture.error:
                print(f"Capture stopped: {capture.error}")
    except KeyboardInterrupt:
        print("\nStopping...")
        engine.stop()
    finally:
        summary = engine.close()
        engine.print_stats()
        for label, count in sorted(summary["attacks_by_type"].items(), key=lambda kv: -kv[1]):
            print(f"   {label}: {count:,}")
        print(f"Session log: {engine.session_dir}")


if __name__ == "__main__":
    main()
