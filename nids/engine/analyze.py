"""Analyse a capture file with the model, in one process: no API or dashboard needed.

    nids pcap capture.pcap
    nids pcap capture.pcap --out flows.csv
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nids.engine.flowmeter import Flow, flows_from_packets, in_time_order
from nids.engine.heuristics import PortScanDetector, SynFloodDetector
from nids.engine.pcap import read_packets
from nids.features import ATTACK_SEVERITY, BENIGN, FEATURE_NAMES_84
from nids.ml.model import NIDSModel

BATCH = 5000
COLUMNS = ["start_utc", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "packets", "duration_s",
           "prediction", "confidence", "severity"]


@dataclass
class Report:
    path: str
    packets: int = 0
    flows: int = 0
    single_packet_flows: int = 0
    first_packet: int | None = None
    last_packet: int | None = None
    seconds: float = 0.0
    predictions: Counter = field(default_factory=Counter)
    attack_sources: Counter = field(default_factory=Counter)
    attack_targets: Counter = field(default_factory=Counter)
    rule_alerts: Counter = field(default_factory=Counter)

    @property
    def attacks(self) -> int:
        return sum(n for label, n in self.predictions.items() if label != BENIGN)

    def to_dict(self) -> dict:
        return {
            "file": self.path, "packets": self.packets, "flows_scored": self.flows,
            "single_packet_flows_not_scored": self.single_packet_flows,
            "capture_start_utc": _utc(self.first_packet), "capture_end_utc": _utc(self.last_packet),
            "analysis_seconds": round(self.seconds, 2), "attack_flows": self.attacks,
            "predictions": dict(self.predictions.most_common()),
            "rule_alerts": dict(self.rule_alerts),
            "top_attack_sources": dict(self.attack_sources.most_common(10)),
            "top_attack_targets": dict(self.attack_targets.most_common(10)),
        }

    def text(self) -> str:
        span = (self.last_packet - self.first_packet) / 1e6 if self.first_packet is not None else 0.0
        lines = [
            f"{self.path}",
            f"  {self.packets:,} packets over {span:,.0f} s (from {_utc(self.first_packet)} UTC), "
            f"analysed in {self.seconds:.1f} s",
            f"  {self.flows:,} flows scored; {self.single_packet_flows:,} one-packet flows not scored "
            f"(the model was trained on flows of 2+ packets)",
            "",
        ]
        if not self.attacks and not self.rule_alerts:
            return "\n".join(lines + ["  No attacks detected."])
        lines.append(f"  {self.attacks:,} flows flagged as attacks ({self.attacks / max(self.flows, 1):.1%} of flows)")
        for label, n in self.predictions.most_common():
            if label != BENIGN:
                lines.append(f"    {label:<26} {n:>8,}   severity {ATTACK_SEVERITY.get(label, 'medium')}")
        for label, n in self.rule_alerts.items():
            lines.append(f"    {label:<26} {n:>8,}   sources")
        if self.attack_sources:
            lines += ["", "  Top sources of flagged flows:"]
            lines += [f"    {ip:<18} {n:>8,}" for ip, n in self.attack_sources.most_common(5)]
            lines += ["  Top targets:"]
            lines += [f"    {ip:<18} {n:>8,}" for ip, n in self.attack_targets.most_common(5)]
        return "\n".join(lines)


def _utc(ts: int | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts / 1e6, UTC).strftime("%Y-%m-%d %H:%M:%S")


def analyze(
    path: str | Path,
    model: NIDSModel | None = None,
    out: str | Path | None = None,
    min_confidence: float = 0.0,
    **meter_options,
) -> Report:
    """Build flows from a capture file and classify them.

    Flows are cut exactly as in the training data (see nids.engine.flowmeter).
    Pass idle_timeout / close_grace to cut them the way the live engine does.
    An attack prediction below `min_confidence` is reported as BENIGN.
    """
    model = model or NIDSModel.load()
    report = Report(str(path))
    rules = [SynFloodDetector(), PortScanDetector()]
    rule_sources: dict[str, set] = {r.description: set() for r in rules}
    started = time.time()

    def packets():
        for p in in_time_order(read_packets(path)):
            report.packets += 1
            if report.first_packet is None:
                report.first_packet = p.ts
            report.last_packet = p.ts
            yield p

    out_file = open(out, "w", newline="", encoding="utf-8") if out else None  # noqa: SIM115
    writer = csv.writer(out_file) if out_file else None
    if writer:
        writer.writerow(COLUMNS)

    def score(batch: list[Flow]) -> None:
        if not batch:
            return
        X = pd.DataFrame([f.features() for f in batch], columns=FEATURE_NAMES_84)
        for flow, pred in zip(batch, model.predict_flows(X), strict=True):
            label = pred.label if (not pred.is_attack or pred.confidence >= min_confidence) else BENIGN
            report.predictions[label] += 1
            if label != BENIGN:
                report.attack_sources[flow.src_ip] += 1
                report.attack_targets[flow.dst_ip] += 1
            if writer:
                writer.writerow([
                    _utc(flow.start), flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port, flow.protocol,
                    flow.total_packets, round(flow.duration / 1e6, 6), label, round(pred.confidence, 4),
                    ATTACK_SEVERITY.get(label, "medium"),
                ])

    try:
        batch: list[Flow] = []
        for flow in flows_from_packets(packets(), **meter_options):
            for rule in rules:
                if rule.check(flow):
                    rule_sources[rule.description].add(flow.src_ip)
            if flow.total_packets < 2:
                report.single_packet_flows += 1
                continue
            report.flows += 1
            batch.append(flow)
            if len(batch) >= BATCH:
                score(batch)
                batch = []
        score(batch)
    finally:
        if out_file:
            out_file.close()
    report.rule_alerts = Counter({name: len(sources) for name, sources in rule_sources.items() if sources})
    report.seconds = time.time() - started
    return report
