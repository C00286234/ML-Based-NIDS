"""Feature definitions shared by training, the API and the live engine."""

# The 84 CICFlowMeter feature names, in the order the API accepts them.
# The live engine (nids.engine.feature_extractor) produces exactly these.
FEATURE_NAMES_84 = [
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


# Features left out of training on purpose:
# - Src Port: an ephemeral, OS-chosen number. The model learns which ports the
#   lab's attack machines happened to use, which does not generalise.
# - Bulk features: the live engine cannot compute them (always 0), so training
#   on them would make live traffic look different from training traffic.
EXCLUDED_FROM_TRAINING = [
    "Src Port",
    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
]

TRAINING_FEATURES = [f for f in FEATURE_NAMES_84 if f not in EXCLUDED_FROM_TRAINING]

LABEL_COLUMN = "Label"
BENIGN = "BENIGN"

# Alert severity per class. Names must match the dataset labels exactly.
ATTACK_SEVERITY = {
    "BENIGN": "low",
    "Portscan": "low",
    "Infiltration - Portscan": "high",  # a port scan from an already-compromised internal host
    "FTP-Patator": "medium",
    "SSH-Patator": "medium",
    "Web Attack": "high",
    "DoS Slowloris": "medium",
    "DoS Slowhttptest": "medium",
    "DoS GoldenEye": "high",
    "DoS Hulk": "high",
    "Botnet": "high",
    "DDoS": "critical",
    "Heartbleed": "critical",
    "Infiltration": "critical",
}
