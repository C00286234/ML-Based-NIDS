import pytest
from fastapi.testclient import TestClient

from nids.api import app as api
from nids.features import FEATURE_NAMES_84


@pytest.fixture()
def client(tiny_model):
    api.state.model = tiny_model
    with TestClient(api.app) as c:
        yield c


def _features(flows, i=0):
    return {f: float(flows.iloc[i][f]) for f in FEATURE_NAMES_84}


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy" and body["models_loaded"]


def test_predict_single_flow(client, flows):
    r = client.post("/predict", json={"features": _features(flows, 0), "flow_id": "10.0.0.1-10.0.0.2-6-1234-80"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == flows.iloc[0]["Label"]
    assert len(body["top_features"]) == 5
    assert 0 <= body["confidence"] <= 1
    recent = client.get("/api/recent?limit=1").json()
    assert recent[0]["src_ip"] == "10.0.0.1"


def test_predict_without_explanation(client, flows):
    r = client.post("/predict?explain=false", json={"features": _features(flows, 1)})
    assert r.status_code == 200 and r.json()["top_features"] == []


def test_null_features_are_imputed(client, flows):
    feats = _features(flows, 2)
    feats["Flow Bytes/s"] = None
    assert client.post("/predict", json={"features": feats}).status_code == 200


def test_missing_features_rejected(client, flows):
    feats = _features(flows)
    del feats["Dst Port"]
    assert client.post("/predict", json={"features": feats}).status_code == 422


def test_batch_predict(client, flows):
    rows = [[float(v) for v in flows.iloc[i][FEATURE_NAMES_84]] for i in range(0, 180, 30)]
    body = client.post("/batch_predict", json={"flows": [{"features": r} for r in rows]}).json()
    assert body["total_flows"] == 6
    assert body["attack_count"] == sum(p["prediction"] != "BENIGN" for p in body["predictions"])


def test_classes_and_model_card(client):
    classes = client.get("/classes").json()
    assert classes["total_classes"] == 3
    assert client.get("/model").json()["metadata"]["test"] is True
