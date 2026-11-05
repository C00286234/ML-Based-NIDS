from nids.engine.feature_extractor import FeatureExtractor
from nids.features import ATTACK_SEVERITY, EXCLUDED_FROM_TRAINING, FEATURE_NAMES_84

DATASET_LABELS = [
    "BENIGN", "Portscan", "DoS Hulk", "DDoS", "Infiltration - Portscan", "DoS GoldenEye", "FTP-Patator",
    "DoS Slowloris", "SSH-Patator", "DoS Slowhttptest", "Botnet", "Web Attack", "Infiltration", "Heartbleed",
]


def test_live_extractor_matches_api_feature_order():
    assert FeatureExtractor.FEATURE_NAMES == FEATURE_NAMES_84
    assert len(FEATURE_NAMES_84) == 84


def test_every_label_has_a_severity():
    missing = [label for label in DATASET_LABELS if label not in ATTACK_SEVERITY]
    assert not missing


def test_excluded_features_exist():
    assert set(EXCLUDED_FROM_TRAINING) <= set(FEATURE_NAMES_84)
