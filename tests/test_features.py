from nids.engine.heuristics import PortScanDetector, SynFloodDetector
from nids.features import ATTACK_SEVERITY, EXCLUDED_FROM_TRAINING, FEATURE_NAMES_84

DATASET_LABELS = [
    "BENIGN", "Portscan", "DoS Hulk", "DDoS", "Infiltration - Portscan", "DoS GoldenEye", "FTP-Patator",
    "DoS Slowloris", "SSH-Patator", "DoS Slowhttptest", "Botnet", "Web Attack", "Infiltration", "Heartbleed",
]


def test_there_are_84_distinct_features():
    assert len(FEATURE_NAMES_84) == len(set(FEATURE_NAMES_84)) == 84


def test_rule_alerts_use_real_class_names():
    assert {SynFloodDetector.label, PortScanDetector.label} <= set(ATTACK_SEVERITY)


def test_every_label_has_a_severity():
    missing = [label for label in DATASET_LABELS if label not in ATTACK_SEVERITY]
    assert not missing


def test_excluded_features_exist():
    assert set(EXCLUDED_FROM_TRAINING) <= set(FEATURE_NAMES_84)
