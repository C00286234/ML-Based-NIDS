"""Build the small real-traffic fixture used by tests/test_parity.py.

Picks a few connections of every class from the capture slices in data/pcap/,
writes their packets (headers only) to tests/data/parity_sample.pcap, and the
rows CICFlowMeter produced for them to tests/data/parity_expected.csv.

    python scripts/make_parity_fixture.py
"""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd
from parity_report import KEY, PCAP_DIR, _canonical, comparable, load_day, rebuild

from nids import paths
from nids.engine.pcap import parse_frame, read_frames
from nids.features import FEATURE_NAMES_84, LABEL_COLUMN

OUT = paths.ROOT / "tests" / "data"
SNAPLEN = 96  # Ethernet + the largest IPv4 header + the fixed part of TCP: all the flow builder reads
PER_CLASS = 12  # connections per class per slice
MAX_PACKETS = 400  # skip very long connections to keep the file small
SEED = 7


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    expected, frames = [], []
    for pcap in sorted(PCAP_DIR.glob("*.pcap")):
        ours = rebuild(pcap)
        ref = comparable(load_day(pcap.name.split("_")[0]), int(ours["start"].min()), int(ours["end"].max()))
        packets_per_tuple = (ref["Total Fwd Packet"] + ref["Total Bwd packets"]).groupby(ref["tuple"]).sum()
        small = set(packets_per_tuple[packets_per_tuple <= MAX_PACKETS].index)
        chosen: set = set()
        for _, group in ref[ref["tuple"].isin(small)].groupby(LABEL_COLUMN):
            tuples = sorted(set(group["tuple"]))
            chosen.update(tuples[i] for i in rng.choice(len(tuples), min(PER_CLASS, len(tuples)), replace=False))
        expected.append(ref[ref["tuple"].isin(chosen)])
        for ts, frame, linktype in read_frames(pcap):
            p = parse_frame(frame, ts, linktype)
            if p is not None and _canonical(p.src, p.sport, p.dst, p.dport, p.proto) in chosen:
                frames.append((ts, frame[:SNAPLEN], len(frame)))
        print(f"{pcap.name}: {len(chosen)} connections")

    frames.sort(key=lambda f: f[0])
    with open(OUT / "parity_sample.pcap", "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, SNAPLEN, 1))
        for ts, data, original in frames:
            f.write(struct.pack("<IIII", ts // 1_000_000, ts % 1_000_000, len(data), original))
            f.write(data)
    rows = pd.concat(expected)[KEY + [c for c in FEATURE_NAMES_84 if c not in KEY] + [LABEL_COLUMN]]
    rows.to_csv(OUT / "parity_expected.csv", index=False)
    print(f"{len(rows)} flows, {len(frames):,} packets -> {OUT}")


if __name__ == "__main__":
    main()
