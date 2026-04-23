#!/usr/bin/env python3
"""
PCAP to API: Send CICFlowMeter-extracted flows to the NIDS API for prediction.

Usage:
    python pcap_to_api.py "path/to/file.pcap"
    python pcap_to_api.py "path/to/file.pcap" --api-url http://localhost:8000 --batch-size 500
"""

import sys
import os
import argparse
import logging
import tempfile
import time
import math
import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Import the authoritative feature names
sys.path.insert(0, str(Path(__file__).parent))
from src.constants import FEATURE_NAMES_84

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# CIC column names → FEATURE_NAMES_84 canonical names
CIC_TO_CANONICAL = {
    "src_port":        "Src Port",
    "dst_port":        "Dst Port",
    "protocol":        "Protocol",
    "flow_duration":   "Flow Duration",
    "tot_fwd_pkts":    "Total Fwd Packet",
    "tot_bwd_pkts":    "Total Bwd packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packet",
    "totlen_bwd_pkts": "Total Length of Bwd Packet",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean":"Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean":"Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s":     "Flow Bytes/s",
    "flow_pkts_s":     "Flow Packets/s",
    "flow_iat_mean":   "Flow IAT Mean",
    "flow_iat_std":    "Flow IAT Std",
    "flow_iat_max":    "Flow IAT Max",
    "flow_iat_min":    "Flow IAT Min",
    "fwd_iat_tot":     "Fwd IAT Total",
    "fwd_iat_mean":    "Fwd IAT Mean",
    "fwd_iat_std":     "Fwd IAT Std",
    "fwd_iat_max":     "Fwd IAT Max",
    "fwd_iat_min":     "Fwd IAT Min",
    "bwd_iat_tot":     "Bwd IAT Total",
    "bwd_iat_mean":    "Bwd IAT Mean",
    "bwd_iat_std":     "Bwd IAT Std",
    "bwd_iat_max":     "Bwd IAT Max",
    "bwd_iat_min":     "Bwd IAT Min",
    "fwd_psh_flags":   "Fwd PSH Flags",
    "bwd_psh_flags":   "Bwd PSH Flags",
    "fwd_urg_flags":   "Fwd URG Flags",
    "bwd_urg_flags":   "Bwd URG Flags",
    "fwd_header_len":  "Fwd Header Length",
    "bwd_header_len":  "Bwd Header Length",
    "fwd_pkts_s":      "Fwd Packets/s",
    "bwd_pkts_s":      "Bwd Packets/s",
    "pkt_len_min":     "Packet Length Min",
    "pkt_len_max":     "Packet Length Max",
    "pkt_len_mean":    "Packet Length Mean",
    "pkt_len_std":     "Packet Length Std",
    "pkt_len_var":     "Packet Length Variance",
    "fin_flag_cnt":    "FIN Flag Count",
    "syn_flag_cnt":    "SYN Flag Count",
    "rst_flag_cnt":    "RST Flag Count",
    "psh_flag_cnt":    "PSH Flag Count",
    "ack_flag_cnt":    "ACK Flag Count",
    "urg_flag_cnt":    "URG Flag Count",
    "cwr_flag_count":  "CWR Flag Count",
    "ece_flag_cnt":    "ECE Flag Count",
    "down_up_ratio":   "Down/Up Ratio",
    "pkt_size_avg":    "Average Packet Size",
    "fwd_seg_size_avg":"Fwd Segment Size Avg",
    "bwd_seg_size_avg":"Bwd Segment Size Avg",
    "fwd_byts_b_avg":  "Fwd Bytes/Bulk Avg",
    "fwd_pkts_b_avg":  "Fwd Packet/Bulk Avg",
    "fwd_blk_rate_avg":"Fwd Bulk Rate Avg",
    "bwd_byts_b_avg":  "Bwd Bytes/Bulk Avg",
    "bwd_pkts_b_avg":  "Bwd Packet/Bulk Avg",
    "bwd_blk_rate_avg":"Bwd Bulk Rate Avg",
    "subflow_fwd_pkts":"Subflow Fwd Packets",
    "subflow_fwd_byts":"Subflow Fwd Bytes",
    "subflow_bwd_pkts":"Subflow Bwd Packets",
    "subflow_bwd_byts":"Subflow Bwd Bytes",
    "init_fwd_win_byts":"FWD Init Win Bytes",
    "init_bwd_win_byts":"Bwd Init Win Bytes",
    "fwd_act_data_pkts":"Fwd Act Data Pkts",
    "fwd_seg_size_min": "Fwd Seg Size Min",
    "active_mean":     "Active Mean",
    "active_std":      "Active Std",
    "active_max":      "Active Max",
    "active_min":      "Active Min",
    "idle_mean":       "Idle Mean",
    "idle_std":        "Idle Std",
    "idle_max":        "Idle Max",
    "idle_min":        "Idle Min",
}

# Features that must be zero-filled (absent from CIC output)
ZERO_FILL = ["Fwd RST Flags", "Bwd RST Flags", "ICMP Code", "ICMP Type"]

# Unit conversion multipliers (CIC uses seconds, model trained on microseconds)
UNIT_SCALE_MICROSECONDS = [
    "Flow Duration",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def check_api_health(api_url):
    """Verify API is accessible and healthy."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        resp.raise_for_status()
        logger.info(f"✓ API health check passed: {api_url}")
    except requests.RequestException as e:
        logger.error(f"✗ API unreachable at {api_url}: {e}")
        sys.exit(1)


def run_cicflowmeter(pcap_path, verbose=False):
    """
    Read PCAP and extract flows using CICFlowMeter's FlowSession.
    Uses Scapy PcapReader (no tcpdump needed).
    Returns a pandas DataFrame with CIC flow features.
    """
    from scapy.utils import PcapReader
    from cicflowmeter.flow_session import FlowSession

    if not os.path.exists(pcap_path):
        logger.error(f"PCAP file not found: {pcap_path}")
        sys.exit(1)

    logger.info(f"Processing PCAP: {pcap_path}")

    # Create temp CSV for FlowSession output
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_file = tmp.name

        logger.info("Reading PCAP and extracting flows...")

        # Instantiate FlowSession (output_mode required)
        session = FlowSession(output_mode="csv", output=tmp_file)

        # Read PCAP and process packets
        packet_count = 0
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                session.process(pkt)
                packet_count += 1
                if verbose and packet_count % 10000 == 0:
                    logger.debug(f"Processed {packet_count:,} packets...")

        # Flush remaining open flows
        session.flush_flows()
        logger.info(f"Processed {packet_count:,} packets")

        # Read the CSV
        df = pd.read_csv(tmp_file)
        logger.info(f"✓ Extracted {len(df)} flows from PCAP")

        if len(df) == 0:
            logger.warning("No flows extracted from PCAP (may be empty or malformed)")
            return df

        return df

    except Exception as e:
        logger.error(f"PCAP processing error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.unlink(tmp_file)
            except:
                pass


def map_cic_row_to_features(row):
    """
    Convert a single CIC flow row to the 84-feature dict.
    Handles unit conversions, zero-fills, and derivations.
    """
    features = {}

    # Direct mapping
    for cic_key, canonical in CIC_TO_CANONICAL.items():
        raw = row.get(cic_key, 0.0)
        try:
            val = float(raw)
        except (ValueError, TypeError):
            val = 0.0

        # Sanitize inf/nan
        if math.isnan(val) or math.isinf(val):
            val = 0.0

        # Apply unit scaling (CIC seconds → microseconds)
        if canonical in UNIT_SCALE_MICROSECONDS:
            val = val * 1_000_000

        features[canonical] = val

    # Zero-fill absent features
    for name in ZERO_FILL:
        features[name] = 0.0

    # Derive Total TCP Flow Time
    protocol = features.get("Protocol", 0.0)
    flow_duration = features.get("Flow Duration", 0.0)
    features["Total TCP Flow Time"] = flow_duration if protocol == 6.0 else 0.0

    return features


def map_dataframe(df, verbose=False):
    """
    Convert all rows to 84-feature dicts.
    Returns (list of valid feature dicts, count of skipped rows).
    """
    feature_dicts = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            features = map_cic_row_to_features(row)

            # Validate: must have exactly 84 keys matching canonical names
            if len(features) != 84:
                logger.warning(f"Row {idx}: invalid feature count ({len(features)} != 84), skipping")
                skipped += 1
                continue

            if set(features.keys()) != set(FEATURE_NAMES_84):
                logger.warning(f"Row {idx}: feature name mismatch, skipping")
                skipped += 1
                continue

            feature_dicts.append(features)

            if verbose and idx % 1000 == 0:
                logger.debug(f"Mapped {idx} rows...")

        except Exception as e:
            logger.warning(f"Row {idx}: mapping error ({e}), skipping")
            skipped += 1
            continue

    if skipped > 0:
        logger.info(f"Skipped {skipped} rows due to mapping errors")

    return feature_dicts, skipped


def flows_to_batch_payload(feature_dicts):
    """
    Convert list of feature dicts to /batch_predict payload.
    Each flow is a list of 84 floats in FEATURE_NAMES_84 order.
    """
    flows = []
    for fd in feature_dicts:
        ordered = [fd[name] for name in FEATURE_NAMES_84]
        flows.append({"features": ordered})
    return {"flows": flows}


def send_batch(session, api_url, feature_dicts, batch_num, results_tracker):
    """Send one batch to /batch_predict and update results."""
    payload = flows_to_batch_payload(feature_dicts)
    try:
        resp = session.post(f"{api_url}/batch_predict", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results_tracker["total"] += data["total_flows"]
        results_tracker["attacks"] += data["attack_count"]

        # Track attack types
        for pred in data["predictions"]:
            if pred.get("is_attack", False):
                attack_type = pred.get("prediction", "Unknown")
                results_tracker["attack_types"][attack_type] += 1

        elapsed = data.get("processing_time_ms", 0)
        logger.info(f"Batch {batch_num}: {data['total_flows']} flows, "
                   f"{data['attack_count']} attacks ({elapsed:.0f}ms)")
        return True

    except requests.HTTPError as e:
        logger.error(f"Batch {batch_num} HTTP {e.response.status_code}: "
                    f"{e.response.text[:200]}")
        results_tracker["errors"] += len(feature_dicts)
        return False
    except requests.RequestException as e:
        logger.error(f"Batch {batch_num} connection error: {e}")
        results_tracker["errors"] += len(feature_dicts)
        return False


def send_all_batches(feature_dicts, api_url, batch_size):
    """Send all flows in batches to the API."""
    if not feature_dicts:
        logger.warning("No flows to send")
        return {"total": 0, "attacks": 0, "errors": 0, "attack_types": {}}

    session = requests.Session()
    results_tracker = {
        "total": 0,
        "attacks": 0,
        "errors": 0,
        "attack_types": defaultdict(int),
    }

    num_batches = (len(feature_dicts) + batch_size - 1) // batch_size
    logger.info(f"Sending {len(feature_dicts)} flows in {num_batches} batches (size={batch_size})")

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(feature_dicts))
        batch = feature_dicts[start_idx:end_idx]

        send_batch(session, api_url, batch, batch_num + 1, results_tracker)

        # Rate limiting: 30 req/min = 2 sec/req. Sleep to avoid 429.
        if batch_num < num_batches - 1:
            time.sleep(2)

    return results_tracker


def print_summary(pcap_path, feature_dicts, results, skipped_count):
    """Print a formatted summary of the analysis."""
    total_flows = len(feature_dicts)
    sent_flows = results["total"]
    attack_count = results["attacks"]
    benign_count = sent_flows - attack_count
    attack_rate = (attack_count / sent_flows * 100) if sent_flows > 0 else 0.0

    print("\n" + "=" * 70)
    print("PCAP ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"PCAP file:         {os.path.basename(pcap_path)}")
    print(f"Total extracted:   {total_flows:,}")
    print(f"Rows skipped:      {skipped_count:,} (validation errors)")
    print(f"Flows sent:        {sent_flows:,}")
    print(f"API errors:        {results['errors']:,}")
    print("-" * 70)
    print(f"BENIGN:            {benign_count:,} ({100-attack_rate:.1f}%)")
    print(f"ATTACKS total:     {attack_count:,} ({attack_rate:.1f}%)")
    for attack_type in sorted(results["attack_types"].keys()):
        count = results["attack_types"][attack_type]
        pct = (count / sent_flows * 100) if sent_flows > 0 else 0.0
        print(f"  • {attack_type:20s} {count:>6,} ({pct:>5.1f}%)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Send CICFlowMeter-extracted PCAP flows to the NIDS prediction API."
    )
    parser.add_argument("pcap", help="Path to PCAP file")
    parser.add_argument("--api-url", default="http://localhost:8000",
                       help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--batch-size", type=int, default=500,
                       help="Flows per batch (max 1000, default: 500)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"PCAP to API Converter v1.0")
    logger.info(f"Python: {sys.version.split()[0]}")

    # Step 1: Check API
    check_api_health(args.api_url)

    # Step 2: Run CICFlowMeter
    df = run_cicflowmeter(args.pcap, verbose=args.verbose)
    if df is None or len(df) == 0:
        logger.error("No flows extracted, exiting")
        sys.exit(1)

    # Step 3: Map features
    logger.info("Mapping CIC columns to canonical feature names...")
    feature_dicts, skipped = map_dataframe(df, verbose=args.verbose)
    logger.info(f"✓ Mapped {len(feature_dicts)} flows")

    # Step 4: Send batches
    logger.info(f"Connecting to API at {args.api_url}...")
    results = send_all_batches(feature_dicts, args.api_url, args.batch_size)

    # Step 5: Summary
    print_summary(args.pcap, feature_dicts, results, skipped)


if __name__ == "__main__":
    main()
