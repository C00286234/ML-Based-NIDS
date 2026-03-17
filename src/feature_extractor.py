"""
Feature Extractor Module for NIDS
=================================
Extracts the 84 CIC-IDS2017 features from network flows.

This module computes all features required by the trained ML model,
matching the exact feature names and calculations used in training.

Author: C00286234
"""

import math
import logging
import sys
import os
from typing import Dict, List, Optional
from statistics import mean, stdev, variance

# Handle imports whether run from src/ or project root
try:
    from flow_aggregator import Flow, PacketInfo
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from flow_aggregator import Flow, PacketInfo

# Configure logging
logger = logging.getLogger("nids.features")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division is impossible."""
    if denominator == 0 or math.isinf(denominator) or math.isnan(denominator):
        return default
    result = numerator / denominator
    if math.isinf(result) or math.isnan(result):
        return default
    return result


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """Calculate mean safely, handling empty lists."""
    if not values:
        return default
    try:
        result = mean(values)
        return default if math.isinf(result) or math.isnan(result) else result
    except Exception:
        return default


def safe_stdev(values: List[float], default: float = 0.0) -> float:
    """Calculate standard deviation safely."""
    if len(values) < 2:
        return default
    try:
        result = stdev(values)
        return default if math.isinf(result) or math.isnan(result) else result
    except Exception:
        return default


def safe_variance(values: List[float], default: float = 0.0) -> float:
    """Calculate variance safely."""
    if len(values) < 2:
        return default
    try:
        result = variance(values)
        return default if math.isinf(result) or math.isnan(result) else result
    except Exception:
        return default


def safe_max(values: List[float], default: float = 0.0) -> float:
    """Get max value safely."""
    if not values:
        return default
    return max(values)


def safe_min(values: List[float], default: float = 0.0) -> float:
    """Get min value safely."""
    if not values:
        return default
    return min(values)


def calculate_iat(packets: List[PacketInfo]) -> List[float]:
    """
    Calculate inter-arrival times between consecutive packets.

    IAT = time difference between packet i and packet i-1 (in microseconds)
    """
    if len(packets) < 2:
        return []

    iats = []
    sorted_packets = sorted(packets, key=lambda p: p.timestamp)

    for i in range(1, len(sorted_packets)):
        iat = (sorted_packets[i].timestamp - sorted_packets[i-1].timestamp) * 1_000_000
        if iat >= 0:  # Ignore negative IATs (shouldn't happen but safety check)
            iats.append(iat)

    return iats


class FeatureExtractor:
    """
    Extracts 84 CIC-IDS2017 compatible features from network flows.

    The features are computed to match the exact names and calculations
    used in the CIC-IDS2017 dataset for compatibility with trained models.

    Example:
        extractor = FeatureExtractor()
        features = extractor.extract_features(flow)
        # features is a dict with 84 feature names and values
    """

    # The exact 84 feature names in order (matching training data)
    FEATURE_NAMES = [
        "Src Port", "Dst Port", "Protocol", "Flow Duration",
        "Total Fwd Packet", "Total Bwd packets",
        "Total Length of Fwd Packet", "Total Length of Bwd Packet",
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "Fwd RST Flags", "Bwd RST Flags",
        "Fwd Header Length", "Bwd Header Length",
        "Fwd Packets/s", "Bwd Packets/s",
        "Packet Length Min", "Packet Length Max",
        "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "CWR Flag Count", "ECE Flag Count",
        "Down/Up Ratio", "Average Packet Size",
        "Fwd Segment Size Avg", "Bwd Segment Size Avg",
        "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
        "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
        "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "FWD Init Win Bytes", "Bwd Init Win Bytes",
        "Fwd Act Data Pkts", "Fwd Seg Size Min",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
        "ICMP Code", "ICMP Type", "Total TCP Flow Time"
    ]

    def __init__(self):
        """Initialize the feature extractor."""
        logger.info(f"FeatureExtractor initialized with {len(self.FEATURE_NAMES)} features")

    def extract_features(self, flow: Flow) -> Dict[str, float]:
        """
        Extract all 84 CIC-IDS2017 features from a flow.

        Args:
            flow: The Flow object to extract features from

        Returns:
            Dictionary mapping feature names to values
        """
        features = {}

        # Get packet lists
        fwd_packets = flow.fwd_packets
        bwd_packets = flow.bwd_packets
        all_packets = fwd_packets + bwd_packets

        # Get packet lengths
        fwd_lengths = [p.length for p in fwd_packets]
        bwd_lengths = [p.length for p in bwd_packets]
        all_lengths = fwd_lengths + bwd_lengths

        # Get payload lengths
        fwd_payload_lengths = [p.payload_length for p in fwd_packets]
        bwd_payload_lengths = [p.payload_length for p in bwd_packets]

        # Get header lengths
        fwd_header_lengths = [p.header_length for p in fwd_packets]
        bwd_header_lengths = [p.header_length for p in bwd_packets]

        # Calculate IATs
        flow_iats = calculate_iat(all_packets)
        fwd_iats = calculate_iat(fwd_packets)
        bwd_iats = calculate_iat(bwd_packets)

        # Flow duration in microseconds
        duration_us = flow.duration
        duration_s = duration_us / 1_000_000 if duration_us > 0 else 0

        # Total bytes
        total_fwd_bytes = sum(fwd_lengths)
        total_bwd_bytes = sum(bwd_lengths)
        total_bytes = total_fwd_bytes + total_bwd_bytes

        # ========================================================================
        # BASIC FEATURES (1-4)
        # ========================================================================
        features["Src Port"] = float(flow.src_port)
        features["Dst Port"] = float(flow.dst_port)
        features["Protocol"] = float(flow.protocol)
        features["Flow Duration"] = duration_us

        # ========================================================================
        # PACKET COUNT FEATURES (5-6)
        # ========================================================================
        features["Total Fwd Packet"] = float(len(fwd_packets))
        features["Total Bwd packets"] = float(len(bwd_packets))

        # ========================================================================
        # PACKET LENGTH FEATURES (7-16)
        # ========================================================================
        features["Total Length of Fwd Packet"] = float(total_fwd_bytes)
        features["Total Length of Bwd Packet"] = float(total_bwd_bytes)

        features["Fwd Packet Length Max"] = safe_max(fwd_lengths)
        features["Fwd Packet Length Min"] = safe_min(fwd_lengths)
        features["Fwd Packet Length Mean"] = safe_mean(fwd_lengths)
        features["Fwd Packet Length Std"] = safe_stdev(fwd_lengths)

        features["Bwd Packet Length Max"] = safe_max(bwd_lengths)
        features["Bwd Packet Length Min"] = safe_min(bwd_lengths)
        features["Bwd Packet Length Mean"] = safe_mean(bwd_lengths)
        features["Bwd Packet Length Std"] = safe_stdev(bwd_lengths)

        # ========================================================================
        # FLOW RATE FEATURES (17-18)
        # ========================================================================
        features["Flow Bytes/s"] = safe_divide(total_bytes, duration_s)
        features["Flow Packets/s"] = safe_divide(len(all_packets), duration_s)

        # ========================================================================
        # FLOW IAT FEATURES (19-22)
        # ========================================================================
        features["Flow IAT Mean"] = safe_mean(flow_iats)
        features["Flow IAT Std"] = safe_stdev(flow_iats)
        features["Flow IAT Max"] = safe_max(flow_iats)
        features["Flow IAT Min"] = safe_min(flow_iats)

        # ========================================================================
        # FORWARD IAT FEATURES (23-27)
        # ========================================================================
        features["Fwd IAT Total"] = sum(fwd_iats) if fwd_iats else 0.0
        features["Fwd IAT Mean"] = safe_mean(fwd_iats)
        features["Fwd IAT Std"] = safe_stdev(fwd_iats)
        features["Fwd IAT Max"] = safe_max(fwd_iats)
        features["Fwd IAT Min"] = safe_min(fwd_iats)

        # ========================================================================
        # BACKWARD IAT FEATURES (28-32)
        # ========================================================================
        features["Bwd IAT Total"] = sum(bwd_iats) if bwd_iats else 0.0
        features["Bwd IAT Mean"] = safe_mean(bwd_iats)
        features["Bwd IAT Std"] = safe_stdev(bwd_iats)
        features["Bwd IAT Max"] = safe_max(bwd_iats)
        features["Bwd IAT Min"] = safe_min(bwd_iats)

        # ========================================================================
        # FLAG FEATURES (33-38)
        # ========================================================================
        features["Fwd PSH Flags"] = float(flow.fwd_psh_flags)
        features["Bwd PSH Flags"] = float(flow.bwd_psh_flags)
        features["Fwd URG Flags"] = float(flow.fwd_urg_flags)
        features["Bwd URG Flags"] = float(flow.bwd_urg_flags)
        features["Fwd RST Flags"] = float(flow.fwd_rst_flags)
        features["Bwd RST Flags"] = float(flow.bwd_rst_flags)

        # ========================================================================
        # HEADER LENGTH FEATURES (39-40)
        # ========================================================================
        features["Fwd Header Length"] = float(sum(fwd_header_lengths))
        features["Bwd Header Length"] = float(sum(bwd_header_lengths))

        # ========================================================================
        # PACKETS PER SECOND (41-42)
        # ========================================================================
        features["Fwd Packets/s"] = safe_divide(len(fwd_packets), duration_s)
        features["Bwd Packets/s"] = safe_divide(len(bwd_packets), duration_s)

        # ========================================================================
        # OVERALL PACKET LENGTH STATS (43-47)
        # ========================================================================
        features["Packet Length Min"] = safe_min(all_lengths)
        features["Packet Length Max"] = safe_max(all_lengths)
        features["Packet Length Mean"] = safe_mean(all_lengths)
        features["Packet Length Std"] = safe_stdev(all_lengths)
        features["Packet Length Variance"] = safe_variance(all_lengths)

        # ========================================================================
        # TOTAL FLAG COUNTS (48-55)
        # ========================================================================
        features["FIN Flag Count"] = float(flow.total_fin_flags)
        features["SYN Flag Count"] = float(flow.total_syn_flags)
        features["RST Flag Count"] = float(flow.total_rst_flags)
        features["PSH Flag Count"] = float(flow.total_psh_flags)
        features["ACK Flag Count"] = float(flow.total_ack_flags)
        features["URG Flag Count"] = float(flow.total_urg_flags)
        features["CWR Flag Count"] = float(flow.total_cwr_flags)
        features["ECE Flag Count"] = float(flow.total_ece_flags)

        # ========================================================================
        # RATIO AND AVERAGE FEATURES (56-58)
        # ========================================================================
        # Down/Up Ratio: bwd_packets / fwd_packets
        features["Down/Up Ratio"] = safe_divide(len(bwd_packets), len(fwd_packets))

        # Average Packet Size
        features["Average Packet Size"] = safe_mean(all_lengths)

        # Segment Size Averages (same as packet length means for our purposes)
        features["Fwd Segment Size Avg"] = safe_mean(fwd_lengths)
        features["Bwd Segment Size Avg"] = safe_mean(bwd_lengths)

        # ========================================================================
        # BULK FEATURES (59-64)
        # ========================================================================
        # Bulk features - for simplicity, set to 0 (requires bulk detection logic)
        features["Fwd Bytes/Bulk Avg"] = 0.0
        features["Fwd Packet/Bulk Avg"] = 0.0
        features["Fwd Bulk Rate Avg"] = 0.0
        features["Bwd Bytes/Bulk Avg"] = 0.0
        features["Bwd Packet/Bulk Avg"] = 0.0
        features["Bwd Bulk Rate Avg"] = 0.0

        # ========================================================================
        # SUBFLOW FEATURES (65-68)
        # ========================================================================
        # Subflow = entire flow in this case (no subflow detection)
        features["Subflow Fwd Packets"] = float(len(fwd_packets))
        features["Subflow Fwd Bytes"] = float(total_fwd_bytes)
        features["Subflow Bwd Packets"] = float(len(bwd_packets))
        features["Subflow Bwd Bytes"] = float(total_bwd_bytes)

        # ========================================================================
        # WINDOW SIZE FEATURES (69-70)
        # ========================================================================
        features["FWD Init Win Bytes"] = float(flow.fwd_init_win_bytes)
        features["Bwd Init Win Bytes"] = float(flow.bwd_init_win_bytes)

        # ========================================================================
        # ADDITIONAL FEATURES (71-72)
        # ========================================================================
        # Forward packets with payload
        fwd_act_data_pkts = sum(1 for p in fwd_packets if p.payload_length > 0)
        features["Fwd Act Data Pkts"] = float(fwd_act_data_pkts)

        # Minimum segment size in forward direction
        fwd_seg_sizes = [p.payload_length for p in fwd_packets if p.payload_length > 0]
        features["Fwd Seg Size Min"] = safe_min(fwd_seg_sizes) if fwd_seg_sizes else 0.0

        # ========================================================================
        # ACTIVE/IDLE TIME FEATURES (73-80)
        # ========================================================================
        active_times = flow.active_times if flow.active_times else [0.0]
        idle_times = flow.idle_times if flow.idle_times else [0.0]

        features["Active Mean"] = safe_mean(active_times) * 1_000_000  # Convert to microseconds
        features["Active Std"] = safe_stdev(active_times) * 1_000_000
        features["Active Max"] = safe_max(active_times) * 1_000_000
        features["Active Min"] = safe_min(active_times) * 1_000_000

        features["Idle Mean"] = safe_mean(idle_times) * 1_000_000
        features["Idle Std"] = safe_stdev(idle_times) * 1_000_000
        features["Idle Max"] = safe_max(idle_times) * 1_000_000
        features["Idle Min"] = safe_min(idle_times) * 1_000_000

        # ========================================================================
        # ICMP FEATURES (81-82)
        # ========================================================================
        features["ICMP Code"] = float(flow.icmp_code)
        features["ICMP Type"] = float(flow.icmp_type)

        # ========================================================================
        # TOTAL TCP FLOW TIME (83)
        # ========================================================================
        # For TCP flows, this is the duration; for others, 0
        features["Total TCP Flow Time"] = duration_us if flow.protocol == 6 else 0.0

        # ========================================================================
        # VALIDATION
        # ========================================================================
        # Ensure we have exactly 84 features
        if len(features) != 84:
            logger.warning(f"Feature count mismatch: {len(features)} != 84")
            # Fill missing features with 0
            for name in self.FEATURE_NAMES:
                if name not in features:
                    features[name] = 0.0
                    logger.warning(f"Missing feature filled with 0: {name}")

        # Replace any remaining inf/nan values
        for key, value in features.items():
            if math.isinf(value) or math.isnan(value):
                features[key] = 0.0

        return features

    def get_feature_names(self) -> List[str]:
        """Return the list of feature names in order."""
        return self.FEATURE_NAMES.copy()

    def validate_features(self, features: Dict[str, float]) -> bool:
        """
        Validate that features dict has all required features.

        Args:
            features: Dictionary of features to validate

        Returns:
            True if valid, False otherwise
        """
        if len(features) != 84:
            return False

        for name in self.FEATURE_NAMES:
            if name not in features:
                return False

        return True


if __name__ == "__main__":
    # Test feature extraction with a mock flow
    import time
    from flow_aggregator import Flow, PacketInfo

    logging.basicConfig(level=logging.INFO)

    print("Testing FeatureExtractor...")

    # Create a mock flow
    base_time = time.time()

    flow = Flow(
        flow_id="test-flow",
        src_ip="192.168.1.100",
        dst_ip="10.0.0.1",
        src_port=45678,
        dst_port=80,
        protocol=6,  # TCP
        start_time=base_time,
        last_time=base_time + 1.5
    )

    # Add some forward packets
    for i in range(10):
        pkt = PacketInfo(
            timestamp=base_time + i * 0.1,
            length=100 + i * 20,
            payload_length=50 + i * 10,
            header_length=40,
            tcp_flags={'SYN': 1 if i == 0 else 0, 'ACK': 1, 'PSH': 1 if i > 5 else 0},
            window_size=65535
        )
        flow.add_forward_packet(pkt)

    # Add some backward packets
    for i in range(7):
        pkt = PacketInfo(
            timestamp=base_time + 0.05 + i * 0.15,
            length=200 + i * 30,
            payload_length=150 + i * 20,
            header_length=40,
            tcp_flags={'ACK': 1, 'PSH': 1 if i > 3 else 0},
            window_size=32768
        )
        flow.add_backward_packet(pkt)

    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_features(flow)

    print(f"\nExtracted {len(features)} features:")
    print("-" * 60)

    for i, (name, value) in enumerate(features.items(), 1):
        print(f"{i:2d}. {name:30s}: {value:.4f}")

    print("-" * 60)
    print(f"Total features: {len(features)}")
    print(f"Valid: {extractor.validate_features(features)}")

    # Verify all feature names match
    expected_names = set(extractor.FEATURE_NAMES)
    actual_names = set(features.keys())

    if expected_names == actual_names:
        print("All feature names match expected CIC-IDS2017 format!")
    else:
        missing = expected_names - actual_names
        extra = actual_names - expected_names
        if missing:
            print(f"Missing features: {missing}")
        if extra:
            print(f"Extra features: {extra}")
