"""Download a few minutes of a very large remote PCAP without fetching the whole file.

The CIC-IDS2017 captures are 8-13 GB per day. This finds the byte offset of a
start time by bisection (HTTP range requests), downloads from there, and writes
a classic PCAP of whole packets (the source files are pcapng).

    python scripts/pcap_slice.py tuesday 2017-07-04T12:15:00 --minutes 12

Times are UTC (the dataset's CSV timestamps are UTC too).
"""

from __future__ import annotations

import argparse
import struct
from datetime import UTC, datetime
from pathlib import Path

import requests

MIRROR = "https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/pcaps/"
DAYS = {
    "monday": "Monday-WorkingHours.pcap",
    "tuesday": "Tuesday-WorkingHours.pcap",
    "wednesday": "Wednesday-workingHours.pcap",
    "thursday": "Thursday-WorkingHours.pcap",
    "friday": "Friday-WorkingHours.pcap",
}
EPB = 6  # pcapng Enhanced Packet Block
EPB_FIXED = 28  # type, length, interface, ts high, ts low, captured length, original length
MAX_PACKET = 262_144
PROBE = 1 << 20
# What we write: a classic little-endian, microsecond pcap of Ethernet frames.
PCAP_HEADER = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, MAX_PACKET, 1)


class RemotePcapng:
    """Random access into a remote pcapng file (microsecond timestamps, as CIC-IDS2017 uses)."""

    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        r = self.session.get(url, headers={"Range": "bytes=0-4095"}, timeout=60)
        r.raise_for_status()
        if struct.unpack_from("<I", r.content)[0] != 0x0A0D0D0A:
            raise ValueError("not a pcapng file")
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.day_start = 0
        first = self.next_block(r.content)
        self.first_packet = first
        self.day_start = self.timestamp(r.content, first)

    def read(self, offset: int, length: int) -> bytes:
        r = self.session.get(
            self.url, headers={"Range": f"bytes={offset}-{offset + length - 1}"}, timeout=120
        )
        r.raise_for_status()
        return r.content

    @staticmethod
    def timestamp(buf: bytes, pos: int) -> int:
        """Microseconds since the epoch (kept as an integer: floats lose the last digit)."""
        high, low = struct.unpack_from("<II", buf, pos + 12)
        return (high << 32) | low

    def _valid(self, buf: bytes, pos: int) -> int:
        """Length of the packet block at pos, or 0 if this is not one."""
        if pos + EPB_FIXED > len(buf):
            return 0
        kind, total, _, _, _, captured, original = struct.unpack_from("<IIIIIII", buf, pos)
        if kind != EPB or total % 4 or not (0 < captured <= original <= MAX_PACKET):
            return 0
        if not (EPB_FIXED + 4 + captured <= total <= EPB_FIXED + 4 + captured + 64):
            return 0
        if self.day_start and not (0 <= self.timestamp(buf, pos) - self.day_start + 60_000_000 <= 86_460_000_000):
            return 0
        # A real block repeats its length at the end.
        if pos + total <= len(buf) and struct.unpack_from("<I", buf, pos + total - 4)[0] != total:
            return 0
        return total

    def next_block(self, buf: bytes) -> int:
        """Offset in buf of the first real packet block: one followed by three more."""
        for start in range(0, len(buf) - EPB_FIXED):
            pos, ok = start, 0
            while ok < 4 and (step := self._valid(buf, pos)):
                pos += step
                ok += 1
            if ok == 4:
                return start
        raise ValueError("could not find a packet boundary")

    def offset_for(self, when: int) -> int:
        """Byte offset shortly before the first packet at or after `when`."""
        lo, hi = self.first_packet, self.size - PROBE
        while hi - lo > 4 * PROBE:
            mid = (lo + hi) // 2
            buf = self.read(mid, PROBE)
            if self.timestamp(buf, self.next_block(buf)) < when:
                lo = mid
            else:
                hi = mid
        return lo


def fetch(day: str, start: datetime, minutes: float, out: Path, max_mb: int = 1500) -> Path:
    pcap = RemotePcapng(MIRROR + DAYS[day])
    t0 = int(start.replace(tzinfo=UTC).timestamp()) * 1_000_000
    t1 = t0 + int(minutes * 60_000_000)
    begin = pcap.offset_for(t0)
    end = min(pcap.offset_for(t1) + 8 * PROBE, pcap.size)
    if end - begin > max_mb << 20:
        raise ValueError(f"slice would be {(end - begin) >> 20} MB; raise --max-mb to allow it")
    print(f"Downloading bytes {begin:,}-{end:,} ({(end - begin) >> 20} MB) of {DAYS[day]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out, "wb") as f:
        f.write(PCAP_HEADER)
        buf = b""
        synced = False
        with pcap.session.get(
            pcap.url, headers={"Range": f"bytes={begin}-{end - 1}"}, stream=True, timeout=120
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(4 << 20):
                buf += chunk
                if not synced:
                    if len(buf) < PROBE:
                        continue
                    buf = buf[pcap.next_block(buf[:PROBE]) :]
                    synced = True
                pos = 0
                while pos + EPB_FIXED <= len(buf):
                    kind, total = struct.unpack_from("<II", buf, pos)
                    if pos + total > len(buf):
                        break
                    if kind == EPB:
                        ts = pcap.timestamp(buf, pos)
                        captured, original = struct.unpack_from("<II", buf, pos + 20)
                        if t0 <= ts < t1:
                            f.write(struct.pack("<IIII", ts // 1_000_000, ts % 1_000_000, captured, original))
                            f.write(buf[pos + EPB_FIXED : pos + EPB_FIXED + captured])
                            kept += 1
                    pos += total
                buf = buf[pos:]
    print(f"Wrote {kept:,} packets to {out} ({out.stat().st_size >> 20} MB)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("day", choices=DAYS)
    p.add_argument("start", help="UTC start time, e.g. 2017-07-04T12:15:00")
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--out", type=Path)
    p.add_argument("--max-mb", type=int, default=1500)
    args = p.parse_args()
    start = datetime.fromisoformat(args.start)
    out = args.out or Path("data/pcap") / f"{args.day}_{start:%H%M}_{args.minutes:g}min.pcap"
    fetch(args.day, start, args.minutes, out, args.max_mb)


if __name__ == "__main__":
    main()
