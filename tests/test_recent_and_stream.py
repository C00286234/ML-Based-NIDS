"""Tests for the ring buffer, /api/recent, and /api/stream SSE endpoints."""

import json
from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient

from nids.api.app import DetectionRecord, _publish_detection, _recent_detections, app


def _make_record(prediction="DoS Hulk", confidence=0.99, severity="critical", idx=0):
    """Create a DetectionRecord for testing."""
    return DetectionRecord(
        id=f"test-{idx}",
        timestamp=datetime.now().isoformat(),
        flow_id=f"192.168.10.{idx % 256}-10.0.0.5-6-443-80",
        src_ip=f"192.168.10.{idx % 256}",
        dst_ip="10.0.0.5",
        prediction=prediction,
        confidence=confidence,
        is_attack=prediction != "BENIGN",
        severity=severity,
        model_agreement=True,
        top_features=[],
        probabilities={prediction: confidence},
    )


@pytest.fixture(autouse=True)
def _clear_buffer():
    """Clear the ring buffer before each test."""
    _recent_detections.clear()
    yield
    _recent_detections.clear()


class TestRingBuffer:
    def test_empty_buffer(self):
        assert len(_recent_detections) == 0

    def test_append_and_order(self):
        for i in range(5):
            _publish_detection(_make_record(idx=i))
        assert len(_recent_detections) == 5
        assert _recent_detections[0].id == "test-4"
        assert _recent_detections[4].id == "test-0"

    def test_max_capacity_200(self):
        for i in range(250):
            _publish_detection(_make_record(idx=i))
        assert len(_recent_detections) == 200
        ids = {r.id for r in _recent_detections}
        assert "test-0" not in ids
        assert "test-249" in ids


class TestRecentEndpoint:
    async def test_empty(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/recent")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_newest_first(self):
        for i in range(10):
            _publish_detection(_make_record(idx=i))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/recent?limit=3")
        data = resp.json()
        assert len(data) == 3
        assert data[0]["id"] == "test-9"
        assert data[2]["id"] == "test-7"

    async def test_severity_filter(self):
        _publish_detection(_make_record(prediction="BENIGN", severity="low", idx=0))
        _publish_detection(_make_record(prediction="DoS Hulk", severity="critical", idx=1))
        _publish_detection(_make_record(prediction="Portscan", severity="high", idx=2))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/recent?severity=critical")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["severity"] == "critical"


class TestDetectionRecord:
    def test_round_trip(self):
        record = _make_record(idx=42)
        d = record.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["id"] == "test-42"
        assert restored["confidence"] == 0.99
        assert restored["is_attack"] is True
