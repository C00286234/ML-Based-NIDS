"""Regression test against real traffic.

tests/data/parity_sample.pcap holds the packets (headers only) of 219 flows from
the CIC-IDS2017 captures, covering every attack class in the slices used by
scripts/parity_report.py. tests/data/parity_expected.csv holds the feature rows
the dataset's CICFlowMeter produced for them. Our flow builder must reproduce
those rows. Rebuild the fixture with scripts/make_parity_fixture.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids.engine.flowmeter import flows_from_packets, in_time_order
from nids.engine.pcap import read_packets
from nids.features import FEATURE_NAMES_84, LABEL_COLUMN

DATA = Path(__file__).parent / "data"
KEY = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "start"]


@pytest.fixture(scope="module")
def compared():
    expected = pd.read_csv(DATA / "parity_expected.csv")
    flows = list(flows_from_packets(in_time_order(read_packets(DATA / "parity_sample.pcap"))))
    ours = pd.DataFrame([f.features() for f in flows], columns=FEATURE_NAMES_84)
    ours["Src IP"] = [f.src_ip for f in flows]
    ours["Dst IP"] = [f.dst_ip for f in flows]
    ours["start"] = [f.start for f in flows]
    return expected, expected.merge(ours, on=KEY, suffixes=("", "_ours"))


def test_fixture_covers_the_attack_classes(compared):
    expected, _ = compared
    assert {"BENIGN", "DDoS", "DoS Hulk", "DoS GoldenEye", "DoS Slowloris", "FTP-Patator", "Portscan",
            "Infiltration - Portscan", "Botnet", "Web Attack"} <= set(expected[LABEL_COLUMN])


def test_every_dataset_flow_is_rebuilt_with_the_same_start_time(compared):
    expected, both = compared
    assert len(both) == len(expected)


def test_every_feature_matches_the_dataset(compared):
    _, both = compared
    theirs = both[FEATURE_NAMES_84].to_numpy(float)
    mine = both[[f if f in KEY else f + "_ours" for f in FEATURE_NAMES_84]].to_numpy(float)
    same = np.isclose(theirs, mine, rtol=1e-4, atol=1e-2) | (np.isnan(mine) & ~np.isfinite(theirs))
    worst = {FEATURE_NAMES_84[i]: float(same[:, i].mean()) for i in np.argsort(same.mean(axis=0))[:3]}
    assert same.all(), f"least-matching features: {worst}"
