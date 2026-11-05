# Dashboard Refresh + Alerting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the polling-based Flask dashboard with an SSE-driven real-time detection feed, Layout C narrative design, expandable SHAP cards, and in-browser alerting.

**Architecture:** Add a ring buffer + pub/sub fan-out in `api/app.py`, expose `/api/recent` and `/api/stream` endpoints, rewrite the dashboard frontend to consume SSE events and render detection cards with SHAP explainability. Browser Notification API for medium+ severity alerts.

**Tech Stack:** Python 3 / FastAPI / Flask / Jinja2 / Chart.js / SSE (EventSource) / Notification API / pytest

---

## File structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `api/app.py` | Add `DetectionRecord`, ring buffer, `_subscribers` pub/sub, `/api/recent`, `/api/stream` |
| Modify | `dashboard/app.py` | Add SSE proxy route `/api/stream`, proxy `/api/recent` |
| Rewrite | `dashboard/templates/index.html` | Layout C structure, detection card markup, notification permission button |
| Modify | `dashboard/static/style.css` | Add detection card, feed, timeline, severity, top-talker, status-bar styles |
| Create | `dashboard/static/app.js` | SSE client, card rendering, chart updates, notification logic, scroll management |
| Create | `tests/test_recent_and_stream.py` | Backend tests for ring buffer, `/api/recent`, `/api/stream` |

---

### Task 1: DetectionRecord + ring buffer in api/app.py

**Files:**
- Modify: `api/app.py:1-16` (imports), `api/app.py:139-159` (ModelState/global state)

- [ ] **Step 1: Add imports and DetectionRecord dataclass**

At the top of `api/app.py`, add `asyncio` and `uuid` to imports, then add the `DetectionRecord` dataclass after the `ModelState` class and `state = ModelState()` line (around line 159):

```python
# Add to existing imports at top of file (line 10-15 area):
import asyncio
import uuid

# Add after line 159 (state = ModelState()):
from src.alerting import ATTACK_SEVERITY

@dataclass
class DetectionRecord:
    """A single detection result for the live feed."""
    id: str
    timestamp: str
    flow_id: str
    src_ip: str
    dst_ip: str
    prediction: str
    confidence: float
    is_attack: bool
    severity: str
    model_agreement: bool
    top_features: list
    probabilities: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "is_attack": self.is_attack,
            "severity": self.severity,
            "model_agreement": self.model_agreement,
            "top_features": [
                {"feature": f.feature, "value": f.value, "contribution": f.contribution}
                if hasattr(f, "feature") else f
                for f in self.top_features
            ],
            "probabilities": self.probabilities,
        }
```

Also add the `dataclasses` import at the top:
```python
from dataclasses import dataclass, field as dc_field
```

Note: `api/app.py` already imports `from dataclasses import ...` indirectly — check if it does. If not, add it. The `FeatureContribution` Pydantic model already exists, so `DetectionRecord.top_features` will hold either those objects (converted via `to_dict`) or plain dicts.

- [ ] **Step 2: Add ring buffer and subscriber set to global state**

After the `DetectionRecord` class definition, add:

```python
# Ring buffer for recent detections + SSE pub/sub
_recent_detections: deque = deque(maxlen=200)
_sse_subscribers: set = set()
```

`deque` is already imported (used by `ModelState.prediction_times`). `_sse_subscribers` holds `asyncio.Queue` instances — one per connected SSE client.

- [ ] **Step 3: Add helper to parse IPs from flow_id**

```python
def _parse_flow_ips(flow_id: Optional[str]) -> tuple:
    """Extract src_ip and dst_ip from a flow_id string like '192.168.1.1-10.0.0.5-6-443-80'.
    Returns (src_ip, dst_ip). Falls back to ('—', '—') if unparseable."""
    if not flow_id:
        return ("—", "—")
    parts = flow_id.split("-")
    if len(parts) >= 2:
        return (parts[0], parts[1])
    return ("—", "—")
```

- [ ] **Step 4: Add publish helper**

```python
def _publish_detection(record: DetectionRecord):
    """Append to ring buffer and fan out to all SSE subscribers."""
    _recent_detections.appendleft(record)
    dead = []
    for q in _sse_subscribers:
        try:
            q.put_nowait(record)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _sse_subscribers.discard(q)
```

- [ ] **Step 5: Wire publish into /predict endpoint**

In the `/predict` endpoint handler (around line 750, after `log_prediction(...)` and before the `return PredictionResponse(...)`), add:

```python
        # Publish to live feed
        src_ip, dst_ip = _parse_flow_ips(flow.flow_id)
        detection = DetectionRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            flow_id=flow.flow_id or "",
            src_ip=src_ip,
            dst_ip=dst_ip,
            prediction=prediction_label,
            confidence=confidence,
            is_attack=is_attack,
            severity=ATTACK_SEVERITY.get(prediction_label, "medium"),
            model_agreement=model_agreement,
            top_features=top_features,
            probabilities=prob_dict,
        )
        _publish_detection(detection)
```

- [ ] **Step 6: Commit**

```bash
git add api/app.py
git commit -m "feat(api): add DetectionRecord, ring buffer, and SSE pub/sub"
```

---

### Task 2: /api/recent and /api/stream endpoints in api/app.py

**Files:**
- Modify: `api/app.py` (add two new endpoint functions before the `if __name__` block)

- [ ] **Step 1: Add /api/recent endpoint**

Add before the `# MAIN ENTRY POINT` section (before line 923):

```python
@app.get("/api/recent", tags=["Live Feed"])
async def get_recent_detections(
    limit: int = 50,
    severity: Optional[str] = None,
):
    """Return recent detections from the ring buffer.

    Query params:
        limit: max records to return (default 50, max 200)
        severity: optional filter — 'critical', 'high', 'medium', 'low'
    """
    limit = min(max(limit, 1), 200)
    records = list(_recent_detections)  # already newest-first
    if severity:
        severity = severity.lower()
        records = [r for r in records if r.severity == severity]
    return [r.to_dict() for r in records[:limit]]
```

- [ ] **Step 2: Add /api/stream SSE endpoint**

```python
from fastapi.responses import StreamingResponse

@app.get("/api/stream", tags=["Live Feed"])
async def stream_detections():
    """SSE endpoint. Streams DetectionRecords as they arrive."""
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _sse_subscribers.add(q)

    async def event_generator():
        try:
            while True:
                try:
                    record = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(record.to_dict())}\n\n"
                except asyncio.TimeoutError:
                    # keepalive comment so proxies don't close the connection
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _sse_subscribers.discard(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

Also add `import json` at the top of the file if not already present. Check — `json` is not in the current imports. Add it to the import block.

- [ ] **Step 3: Update the root endpoint's endpoint list**

In the `root()` handler (around line 582), add the two new endpoints to the `"endpoints"` dict:

```python
            "GET /api/recent": "Recent detections from live feed (supports ?limit=&severity=)",
            "GET /api/stream": "SSE stream of live detections",
```

- [ ] **Step 4: Commit**

```bash
git add api/app.py
git commit -m "feat(api): add /api/recent and /api/stream SSE endpoints"
```

---

### Task 3: Update demo_traffic_simulator to send flow_id

**Files:**
- Modify: `demo_traffic_simulator.py:204-217`

The simulator currently sends `{"features": {...}}` but no `flow_id`. Without `flow_id`, every detection card shows `— → —`. We'll synthesize a flow_id from the feature values.

- [ ] **Step 1: Add flow_id to the API payload**

Replace the `send_flow` function (lines 204-217):

```python
def send_flow(features, api_url):
    """Send a single flow to the API and return the prediction."""
    feature_dict = {name: float(val) for name, val in zip(FEATURE_NAMES_84, features)}

    # Synthesize a flow_id from Src/Dst Port so the dashboard can show IPs.
    # The simulator uses CIC-IDS2017 data which doesn't carry raw IPs in the 84 features,
    # so we generate plausible private IPs for display purposes.
    src_ip = f"192.168.10.{random.randint(1, 254)}"
    dst_ip = f"10.0.0.{random.randint(1, 254)}"
    src_port = int(feature_dict.get("Src Port", 0))
    dst_port = int(feature_dict.get("Dst Port", 0))
    flow_id = f"{src_ip}-{dst_ip}-6-{src_port}-{dst_port}"

    payload = {"features": feature_dict, "flow_id": flow_id}
    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('prediction', 'UNKNOWN'), data.get('confidence', 0.0)
        else:
            return None, 0.0
    except requests.exceptions.RequestException:
        return None, 0.0
```

- [ ] **Step 2: Commit**

```bash
git add demo_traffic_simulator.py
git commit -m "feat(simulator): send flow_id so dashboard can display IPs"
```

---

### Task 4: Backend tests for ring buffer, /api/recent, /api/stream

**Files:**
- Create: `tests/test_recent_and_stream.py`
- Create: `tests/__init__.py` (empty, needed for pytest discovery)

- [ ] **Step 1: Create tests directory and init file**

```bash
mkdir -p tests
```

Write empty `tests/__init__.py`.

- [ ] **Step 2: Write test file**

Create `tests/test_recent_and_stream.py`:

```python
"""Tests for the ring buffer, /api/recent, and /api/stream SSE endpoints."""

import asyncio
import json
import sys
import os

import pytest
from httpx import AsyncClient, ASGITransport

# Ensure project root is on sys.path so api.app can resolve its local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.app import app, _recent_detections, _publish_detection, DetectionRecord
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ring buffer unit tests
# ---------------------------------------------------------------------------

class TestRingBuffer:
    def test_empty_buffer(self):
        assert len(_recent_detections) == 0

    def test_append_and_order(self):
        for i in range(5):
            _publish_detection(_make_record(idx=i))
        assert len(_recent_detections) == 5
        # newest first (appendleft)
        assert _recent_detections[0].id == "test-4"
        assert _recent_detections[4].id == "test-0"

    def test_max_capacity_200(self):
        for i in range(250):
            _publish_detection(_make_record(idx=i))
        assert len(_recent_detections) == 200
        # oldest (idx=0..49) should have been evicted
        ids = {r.id for r in _recent_detections}
        assert "test-0" not in ids
        assert "test-249" in ids


# ---------------------------------------------------------------------------
# /api/recent endpoint tests
# ---------------------------------------------------------------------------

class TestRecentEndpoint:
    @pytest.mark.anyio
    async def test_empty(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/recent")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.anyio
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

    @pytest.mark.anyio
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


# ---------------------------------------------------------------------------
# /api/stream SSE endpoint test
# ---------------------------------------------------------------------------

class TestStreamEndpoint:
    @pytest.mark.anyio
    async def test_receives_event(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream("GET", "/api/stream") as resp:
                # publish a detection while connected
                await asyncio.sleep(0.1)  # let the generator start
                _publish_detection(_make_record(prediction="DDoS", severity="critical", idx=0))

                # read the first SSE data line
                lines_read = []
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        lines_read.append(line)
                        break
                    if line.startswith(":"):
                        # keepalive, skip
                        continue

                assert len(lines_read) == 1
                payload = json.loads(lines_read[0].removeprefix("data:").strip())
                assert payload["prediction"] == "DDoS"
                assert payload["severity"] == "critical"


# ---------------------------------------------------------------------------
# DetectionRecord serialization
# ---------------------------------------------------------------------------

class TestDetectionRecord:
    def test_round_trip(self):
        record = _make_record(idx=42)
        d = record.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["id"] == "test-42"
        assert restored["confidence"] == 0.99
        assert restored["is_attack"] is True
```

- [ ] **Step 3: Run tests to verify they fail (models not loaded — we need to handle that)**

Run:
```bash
cd C:\Users\folosiye\FYP-NIDS-ML
python -m pytest tests/test_recent_and_stream.py -v --timeout=30
```

The ring buffer and serialization tests should pass. The endpoint tests may fail if the lifespan (model loading) fires. If so, we need to handle the lifespan gracefully in tests. We'll fix in the next step.

- [ ] **Step 4: Add pytest fixtures for model loading if needed**

If the tests fail because models try to load, add a `conftest.py` in `tests/`:

```python
"""Shared test fixtures."""
import os
import sys

# Ensure imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

And update `test_recent_and_stream.py` to use `@pytest.fixture` that patches model loading if the ring buffer / endpoint tests fail due to missing model files. The key insight: `/api/recent` and `/api/stream` don't touch models at all, so we can test them without loading models — but FastAPI's lifespan hook calls `load_models()`. We can set an env var or mock.

If tests fail, add to `conftest.py`:

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def skip_model_loading():
    """Prevent model loading during tests that don't need it."""
    with patch("api.app.load_models"):
        yield
```

- [ ] **Step 5: Run tests, confirm green**

```bash
python -m pytest tests/test_recent_and_stream.py -v --timeout=30
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: add tests for ring buffer, /api/recent, and /api/stream"
```

---

### Task 5: Dashboard Flask proxy routes

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: Add /api/recent proxy**

Add after the existing `recent_attacks()` route (around line 222):

```python
@app.route("/api/recent_detections")
def recent_detections():
    """Proxy to the API's /api/recent endpoint."""
    limit = request.args.get("limit", "50")
    severity = request.args.get("severity", "")
    params = f"?limit={limit}"
    if severity:
        params += f"&severity={severity}"
    data = fetch_api(f"/api/recent{params}")
    if data is None:
        return jsonify([])
    return jsonify(data)
```

Also add `request` to Flask imports at the top:

```python
from flask import Flask, render_template, jsonify, request, Response
```

- [ ] **Step 2: Add /api/stream SSE proxy**

```python
@app.route("/api/stream")
def stream_proxy():
    """Proxy the SSE stream from the API to the browser."""
    import requests as req

    def generate():
        try:
            with req.get(
                f"{API_URL}/api/stream",
                headers=get_api_headers(),
                stream=True,
                timeout=(5, None),  # 5s connect, no read timeout
            ) as upstream:
                for line in upstream.iter_lines(decode_unicode=True):
                    if line is not None:
                        yield f"{line}\n"
                    else:
                        yield "\n"
        except req.exceptions.ConnectionError:
            yield "data: {\"error\": \"API disconnected\"}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(dashboard): add /api/recent_detections and /api/stream proxy routes"
```

---

### Task 6: Dashboard HTML rewrite (Layout C)

**Files:**
- Rewrite: `dashboard/templates/index.html`

- [ ] **Step 1: Write the new index.html**

Replace the entire file with the Layout C structure. The JS is moved to `app.js` (created in Task 7), and all new component CSS classes go in `style.css` (Task 8).

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIDS Dashboard - Network Intrusion Detection System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-content">
                <h1>NIDS Dashboard</h1>
                <p class="subtitle">Network Intrusion Detection System - Real-Time Monitoring</p>
            </div>
            <div class="header-right">
                <button class="notify-btn hidden" id="notify-btn" title="Enable browser notifications">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                        <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/>
                        <path d="M13.73 21a2 2 0 01-3.46 0"/>
                    </svg>
                    Enable Alerts
                </button>
                <div class="connection-status" id="connection-status">
                    <span class="status-dot"></span>
                    <span class="status-text">Connecting...</span>
                </div>
            </div>
        </header>

        <!-- Section 1: Stats Row -->
        <section class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon predictions">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                    </svg>
                </div>
                <div class="stat-content">
                    <h3>Total Predictions</h3>
                    <p class="stat-value" id="total-predictions">0</p>
                    <p class="stat-subtitle">All time</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon attacks">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                </div>
                <div class="stat-content">
                    <h3>Attacks Detected</h3>
                    <p class="stat-value attack-count" id="total-attacks">0</p>
                    <p class="stat-subtitle" id="attack-rate">0% attack rate</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon latency">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <polyline points="12,6 12,12 16,14"/>
                    </svg>
                </div>
                <div class="stat-content">
                    <h3>Avg Latency</h3>
                    <p class="stat-value" id="avg-latency">0ms</p>
                    <p class="stat-subtitle">Per prediction</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon uptime">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                    </svg>
                </div>
                <div class="stat-content">
                    <h3>Uptime</h3>
                    <p class="stat-value" id="uptime">0s</p>
                    <p class="stat-subtitle">API runtime</p>
                </div>
            </div>
        </section>

        <!-- Section 2: Traffic Timeline -->
        <section class="panel" id="timeline-panel">
            <h3 class="panel-title">Traffic Timeline <span class="panel-subtitle">last 10 minutes</span></h3>
            <div class="timeline-chart-container">
                <canvas id="timeline-chart"></canvas>
            </div>
        </section>

        <!-- Section 3: Live Detection Feed -->
        <section class="panel" id="feed-panel">
            <div class="feed-header">
                <h3 class="panel-title">Live Detection Feed</h3>
                <button class="new-detections-pill hidden" id="new-pill">
                    <span id="new-pill-count">0</span> new detections &#8593;
                </button>
            </div>
            <div class="sse-banner hidden" id="sse-banner">Disconnected, retrying&hellip;</div>
            <div class="feed-list" id="feed-list">
                <p class="no-data" id="feed-empty">Waiting for detections&hellip;</p>
            </div>
        </section>

        <!-- Section 4: Three-across row -->
        <section class="three-across">
            <div class="panel">
                <h3 class="panel-title">Attack Distribution</h3>
                <div class="chart-container">
                    <canvas id="attack-chart"></canvas>
                </div>
            </div>
            <div class="panel">
                <h3 class="panel-title">Severity Breakdown</h3>
                <div class="severity-bars" id="severity-bars">
                    <p class="no-data">No data yet</p>
                </div>
            </div>
            <div class="panel">
                <h3 class="panel-title">Top Talkers</h3>
                <div class="top-talkers" id="top-talkers">
                    <p class="no-data">No data yet</p>
                </div>
            </div>
        </section>

        <!-- Section 5: System Status Bar -->
        <section class="status-bar" id="status-bar">
            <div class="status-bar-item">
                <span class="status-bar-dot" id="sb-api"></span>
                API <span id="sb-api-label">...</span>
            </div>
            <div class="status-bar-item">
                <span class="status-bar-dot" id="sb-models"></span>
                Models <span id="sb-models-label">...</span>
            </div>
            <div class="status-bar-item">
                <span class="status-bar-dot" id="sb-sse"></span>
                SSE <span id="sb-sse-label">...</span>
            </div>
            <div class="status-bar-item">
                Uptime <span id="sb-uptime">—</span>
            </div>
        </section>

        <!-- Footer -->
        <footer class="footer">
            <p>NIDS Dashboard v2.0.0 | FYP-NIDS-ML Project</p>
        </footer>
    </div>

    <script src="{{ url_for('static', filename='app.js') }}"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/templates/index.html
git commit -m "feat(dashboard): rewrite index.html to Layout C with detection feed"
```

---

### Task 7: Dashboard JavaScript (app.js)

**Files:**
- Create: `dashboard/static/app.js`

- [ ] **Step 1: Write app.js — SSE client, card rendering, charts, notifications**

Create `dashboard/static/app.js`:

```javascript
/* =====================================================================
   NIDS Dashboard — app.js
   SSE-driven live feed, Chart.js charts, browser notifications
   ===================================================================== */

// ---------------------------------------------------------------------------
// Severity helpers
// ---------------------------------------------------------------------------
const SEVERITY_ORDER = { critical: 4, high: 3, medium: 2, low: 1 };
const SEVERITY_COLOR = {
  critical: "#f38ba8",
  high: "#fab387",
  medium: "#f9e2af",
  low: "#a6e3a1",
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let attackChart = null;
let timelineChart = null;
const timelineData = { labels: [], benign: [], attack: [] };
const MAX_TIMELINE_POINTS = 60; // 10 min at 10s buckets
let feedScrolledAway = false;
let pendingCount = 0;
let sseConnected = false;
const severityCounts = { critical: 0, high: 0, medium: 0, low: 0 };
const talkerCounts = {}; // src_ip -> count

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------
function fmt(n) {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
function fmtUptime(s) {
  if (s < 60) return Math.floor(s) + "s";
  if (s < 3600) return Math.floor(s / 60) + "m " + Math.floor(s % 60) + "s";
  return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
}
function timeAgo(iso) {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return Math.floor(diff) + "s ago";
  if (diff < 3600) return Math.floor(diff / 60) + "m ago";
  return Math.floor(diff / 3600) + "h ago";
}

// ---------------------------------------------------------------------------
// Connection status
// ---------------------------------------------------------------------------
function setConnected(ok) {
  const dot = document.querySelector("#connection-status .status-dot");
  const txt = document.querySelector("#connection-status .status-text");
  dot.className = "status-dot " + (ok ? "connected" : "disconnected");
  txt.textContent = ok ? "Connected" : "Disconnected";
}

// ---------------------------------------------------------------------------
// Stats row (polled from /api/status every 10s)
// ---------------------------------------------------------------------------
async function pollStats() {
  try {
    const res = await fetch("/api/status");
    const d = await res.json();
    setConnected(d.connected);
    document.getElementById("total-predictions").textContent = fmt(d.metrics.total_predictions);
    document.getElementById("total-attacks").textContent = fmt(d.metrics.total_attacks);
    document.getElementById("attack-rate").textContent = d.attack_rate_percent + "% attack rate";
    document.getElementById("avg-latency").textContent = d.metrics.avg_prediction_time_ms.toFixed(1) + "ms";
    const up = d.metrics.uptime_seconds;
    document.getElementById("uptime").textContent = fmtUptime(up);
    document.getElementById("sb-uptime").textContent = fmtUptime(up);

    // Status bar
    updateStatusDot("sb-api", d.health.status === "healthy");
    updateStatusDot("sb-models", d.health.models_loaded);
    document.getElementById("sb-api-label").textContent = d.health.status === "healthy" ? "OK" : d.health.status;
    document.getElementById("sb-models-label").textContent = d.health.models_loaded ? "OK" : "down";

    // Attack distribution chart
    updateDoughnut(d.metrics.attack_distribution);

    // Timeline bucket
    pushTimelineBucket(d.metrics.total_predictions, d.metrics.total_attacks);
  } catch {
    setConnected(false);
  }
}

function updateStatusDot(id, ok) {
  const el = document.getElementById(id);
  el.className = "status-bar-dot " + (ok ? "ok" : "err");
}

// ---------------------------------------------------------------------------
// Doughnut chart
// ---------------------------------------------------------------------------
const CATPPUCCIN_PALETTE = [
  "#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#89b4fa",
  "#cba6f7", "#94e2d5", "#f5c2e7", "#74c7ec", "#b4befe",
  "#eba0ac", "#f2cdcd", "#89dceb", "#a6e3a1", "#cdd6f4",
];

function updateDoughnut(distribution) {
  const labels = Object.keys(distribution);
  const values = Object.values(distribution);
  if (labels.length === 0) return;

  const colors = labels.map((_, i) => CATPPUCCIN_PALETTE[i % CATPPUCCIN_PALETTE.length]);
  const ctx = document.getElementById("attack-chart").getContext("2d");

  if (attackChart) {
    attackChart.data.labels = labels;
    attackChart.data.datasets[0].data = values;
    attackChart.data.datasets[0].backgroundColor = colors;
    attackChart.update();
  } else {
    attackChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [{ data: values, backgroundColor: colors, borderWidth: 2, borderColor: "#1e1e2e" }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "right", labels: { color: "#cdd6f4", font: { size: 11 } } },
        },
      },
    });
  }
}

// ---------------------------------------------------------------------------
// Traffic timeline chart
// ---------------------------------------------------------------------------
let _lastTotal = null;
let _lastAttacks = null;

function pushTimelineBucket(total, attacks) {
  if (_lastTotal === null) {
    _lastTotal = total;
    _lastAttacks = attacks;
    return;
  }
  const benignDelta = (total - attacks) - (_lastTotal - _lastAttacks);
  const attackDelta = attacks - _lastAttacks;
  _lastTotal = total;
  _lastAttacks = attacks;

  const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  timelineData.labels.push(now);
  timelineData.benign.push(Math.max(0, benignDelta));
  timelineData.attack.push(Math.max(0, attackDelta));

  if (timelineData.labels.length > MAX_TIMELINE_POINTS) {
    timelineData.labels.shift();
    timelineData.benign.shift();
    timelineData.attack.shift();
  }

  const ctx = document.getElementById("timeline-chart").getContext("2d");
  if (timelineChart) {
    timelineChart.data.labels = timelineData.labels;
    timelineChart.data.datasets[0].data = timelineData.benign;
    timelineChart.data.datasets[1].data = timelineData.attack;
    timelineChart.update("none");
  } else {
    timelineChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: timelineData.labels,
        datasets: [
          {
            label: "Benign",
            data: timelineData.benign,
            borderColor: "#a6e3a1",
            backgroundColor: "rgba(166,227,161,0.1)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
          },
          {
            label: "Attack",
            data: timelineData.attack,
            borderColor: "#f38ba8",
            backgroundColor: "rgba(243,139,168,0.15)",
            fill: true,
            tension: 0.3,
            pointRadius: 2,
            pointBackgroundColor: "#f38ba8",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { ticks: { color: "#6c7086", maxTicksLimit: 10 }, grid: { color: "rgba(205,214,244,0.06)" } },
          y: { beginAtZero: true, ticks: { color: "#6c7086" }, grid: { color: "rgba(205,214,244,0.06)" } },
        },
        plugins: {
          legend: { labels: { color: "#cdd6f4", font: { size: 11 } } },
        },
      },
    });
  }
}

// ---------------------------------------------------------------------------
// Detection card rendering
// ---------------------------------------------------------------------------
function renderCard(d) {
  const card = document.createElement("div");
  card.className = "dcard " + d.severity + (d.prediction === "BENIGN" ? " benign" : "");
  card.dataset.id = d.id;

  const confPct = Math.round(d.confidence * 100);

  card.innerHTML = `
    <div class="stripe"></div>
    <div class="body">
      <div class="row1">
        <span class="attack">${esc(d.prediction)}</span>
        <span class="sev">${esc(d.severity)}</span>
      </div>
      <div class="confbar"><div class="conffill" style="width:${confPct}%"></div></div>
      <div class="flow">${esc(d.src_ip)} &rarr; ${esc(d.dst_ip)}</div>
    </div>
    <div class="rightcol">
      <div class="confpct">${confPct}%</div>
      <div class="time-ago">${timeAgo(d.timestamp)}</div>
    </div>
  `;

  card.addEventListener("click", () => toggleExpand(card, d));
  return card;
}

function esc(s) {
  const el = document.createElement("span");
  el.textContent = s;
  return el.innerHTML;
}

function toggleExpand(card, d) {
  if (card.classList.contains("expanded")) {
    card.classList.remove("expanded");
    const detail = card.querySelector(".detail");
    if (detail) detail.remove();
    return;
  }

  card.classList.add("expanded");
  const confPct = Math.round(d.confidence * 100);

  // Move conf + time into the row1 for expanded view
  const detail = document.createElement("div");
  detail.className = "detail";

  // SHAP features
  let shapHtml = "";
  if (d.top_features && d.top_features.length > 0) {
    shapHtml = `<div><div class="detail-title">Top SHAP contributions</div>`;
    d.top_features.forEach((f) => {
      const isPos = f.contribution >= 0;
      const pct = Math.min(Math.abs(f.contribution) * 100, 50);
      const cls = isPos ? "shap-fill-pos" : "shap-fill-neg";
      shapHtml += `
        <div class="shap-row">
          <div class="fname">${esc(f.feature)}</div>
          <div class="shap-bar"><div class="${cls}" style="width:${pct}%"></div></div>
          <div class="shap-value">${isPos ? "+" : ""}${f.contribution.toFixed(2)}</div>
        </div>`;
    });
    shapHtml += `</div>`;
  }

  // Class probabilities — top 3
  let probHtml = "";
  if (d.probabilities) {
    const sorted = Object.entries(d.probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    probHtml = `<div><div class="detail-title">Class probabilities</div><div class="probs">`;
    sorted.forEach(([cls, prob]) => {
      const pct = (prob * 100).toFixed(1);
      probHtml += `
        <div class="prob-row">
          <span class="pname">${esc(cls)}</span>
          <div class="prob-bar"><div class="prob-fill" style="width:${prob * 100}%"></div></div>
          <span class="prob-value">${pct}%</span>
        </div>`;
    });
    probHtml += `</div></div>`;
  }

  // Model agreement
  const agreeHtml = d.model_agreement
    ? `<div class="agree">&#10003; Random Forest and XGBoost agree</div>`
    : `<div class="agree dis">&#9888; Models disagree</div>`;

  detail.innerHTML = shapHtml + probHtml + agreeHtml;
  // Insert after .body
  const body = card.querySelector(".body");
  body.parentNode.insertBefore(detail, body.nextSibling);
}

// ---------------------------------------------------------------------------
// Feed management
// ---------------------------------------------------------------------------
const MAX_FEED = 200;

function addCardToFeed(d) {
  const list = document.getElementById("feed-list");
  const empty = document.getElementById("feed-empty");
  if (empty) empty.remove();

  // Track severity + talkers
  if (d.is_attack && d.severity) {
    severityCounts[d.severity] = (severityCounts[d.severity] || 0) + 1;
    updateSeverityBars();
  }
  if (d.src_ip && d.src_ip !== "\u2014") {
    talkerCounts[d.src_ip] = (talkerCounts[d.src_ip] || 0) + 1;
    updateTopTalkers();
  }

  const card = renderCard(d);

  if (feedScrolledAway) {
    list.prepend(card);
    pendingCount++;
    showNewPill();
  } else {
    list.prepend(card);
    list.scrollTop = 0;
  }

  // Trim old cards
  while (list.children.length > MAX_FEED) {
    list.lastChild.remove();
  }
}

// Scroll detection
function initFeedScroll() {
  const list = document.getElementById("feed-list");
  list.addEventListener("scroll", () => {
    feedScrolledAway = list.scrollTop > 60;
    if (!feedScrolledAway) {
      pendingCount = 0;
      hideNewPill();
    }
  });
}

function showNewPill() {
  const pill = document.getElementById("new-pill");
  document.getElementById("new-pill-count").textContent = pendingCount;
  pill.classList.remove("hidden");
}

function hideNewPill() {
  document.getElementById("new-pill").classList.add("hidden");
}

document.getElementById("new-pill").addEventListener("click", () => {
  const list = document.getElementById("feed-list");
  list.scrollTop = 0;
  pendingCount = 0;
  hideNewPill();
});

// ---------------------------------------------------------------------------
// Severity bars
// ---------------------------------------------------------------------------
function updateSeverityBars() {
  const container = document.getElementById("severity-bars");
  const total = Object.values(severityCounts).reduce((a, b) => a + b, 0);
  if (total === 0) return;

  let html = "";
  for (const sev of ["critical", "high", "medium", "low"]) {
    const count = severityCounts[sev] || 0;
    const pct = ((count / total) * 100).toFixed(1);
    html += `
      <div class="sev-row">
        <span class="sev-label">${sev}</span>
        <div class="sev-track"><div class="sev-fill" style="width:${pct}%;background:${SEVERITY_COLOR[sev]}"></div></div>
        <span class="sev-count">${count}</span>
      </div>`;
  }
  container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Top talkers
// ---------------------------------------------------------------------------
function updateTopTalkers() {
  const container = document.getElementById("top-talkers");
  const sorted = Object.entries(talkerCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  if (sorted.length === 0) return;

  let html = "";
  sorted.forEach(([ip, count]) => {
    html += `<div class="talker-row"><span class="talker-ip">${esc(ip)}</span><span class="talker-count">${count}</span></div>`;
  });
  container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// SSE client
// ---------------------------------------------------------------------------
function connectSSE() {
  const banner = document.getElementById("sse-banner");
  const source = new EventSource("/api/stream");

  source.onopen = () => {
    sseConnected = true;
    banner.classList.add("hidden");
    updateStatusDot("sb-sse", true);
    document.getElementById("sb-sse-label").textContent = "OK";
  };

  source.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      if (d.error) return; // ignore proxy error messages
      addCardToFeed(d);
      maybeNotify(d);
    } catch {
      // ignore parse errors (keepalive comments, etc.)
    }
  };

  source.onerror = () => {
    sseConnected = false;
    banner.classList.remove("hidden");
    updateStatusDot("sb-sse", false);
    document.getElementById("sb-sse-label").textContent = "down";
    // EventSource auto-reconnects, so we just update the UI
  };
}

// ---------------------------------------------------------------------------
// Browser notifications
// ---------------------------------------------------------------------------
const NOTIFY_COOLDOWN_MS = 3000;
let lastNotifyTime = 0;
let coalescedCount = 0;

function initNotifications() {
  if (!("Notification" in window)) return;

  if (Notification.permission === "granted") {
    // ready
  } else if (Notification.permission === "denied") {
    showNotifyButton();
  } else {
    // default — ask once
    Notification.requestPermission().then((perm) => {
      if (perm !== "granted") showNotifyButton();
    });
  }
}

function showNotifyButton() {
  const btn = document.getElementById("notify-btn");
  btn.classList.remove("hidden");
  btn.addEventListener("click", () => {
    Notification.requestPermission().then((perm) => {
      if (perm === "granted") btn.classList.add("hidden");
    });
  });
}

function maybeNotify(d) {
  if (Notification.permission !== "granted") return;
  // Only medium+ severity
  if ((SEVERITY_ORDER[d.severity] || 0) < SEVERITY_ORDER.medium) return;

  const now = Date.now();
  if (now - lastNotifyTime < NOTIFY_COOLDOWN_MS) {
    coalescedCount++;
    return;
  }

  lastNotifyTime = now;
  let body = `${d.src_ip} \u2192 ${d.dst_ip}  \u2022  ${Math.round(d.confidence * 100)}%`;
  if (coalescedCount > 0) {
    body += `\n+${coalescedCount} more detections`;
    coalescedCount = 0;
  }

  const n = new Notification(d.prediction, {
    body,
    icon: "/static/favicon.ico",
    tag: "nids-" + d.severity,
  });
  n.onclick = () => {
    window.focus();
    const card = document.querySelector(`.dcard[data-id="${d.id}"]`);
    if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
    n.close();
  };
}

// ---------------------------------------------------------------------------
// Time-ago updater (refresh relative times every 10s)
// ---------------------------------------------------------------------------
function refreshTimeAgos() {
  // We'd need to store timestamps on each card to refresh — skip for now,
  // since cards arrive with "Xs ago" baked in and the feed moves fast enough
  // that stale times scroll away naturally.
}

// ---------------------------------------------------------------------------
// Initial load: seed feed from /api/recent_detections
// ---------------------------------------------------------------------------
async function seedFeed() {
  try {
    const res = await fetch("/api/recent_detections?limit=50");
    const records = await res.json();
    // records are newest-first; render oldest first so newest ends up at top
    records.reverse().forEach((d) => addCardToFeed(d));
  } catch {
    // API not up yet — SSE will populate when it connects
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initFeedScroll();
  initNotifications();
  seedFeed();
  connectSSE();
  pollStats();
  setInterval(pollStats, 10000);
});
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/static/app.js
git commit -m "feat(dashboard): add app.js with SSE client, card rendering, charts, notifications"
```

---

### Task 8: Dashboard CSS additions

**Files:**
- Modify: `dashboard/static/style.css`

- [ ] **Step 1: Append new component styles to style.css**

Add the following at the end of `dashboard/static/style.css` (after the existing `@keyframes pulse` block, around line 432):

```css
/* ==============================================================================
   Layout C — Narrative flow additions
   ============================================================================== */

/* Header right section */
.header-right {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
}

/* Notify button */
.notify-btn {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: var(--spacing-xs) var(--spacing-sm);
    background: rgba(249, 226, 175, 0.15);
    color: var(--yellow);
    border: 1px solid rgba(249, 226, 175, 0.3);
    border-radius: var(--radius-sm);
    cursor: pointer;
    font-size: 0.75rem;
    font-weight: 600;
    transition: background 0.2s;
}
.notify-btn:hover { background: rgba(249, 226, 175, 0.25); }

/* Panels */
.panel {
    background: var(--base);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
    margin-bottom: var(--spacing-lg);
    box-shadow: var(--shadow-md);
}
.panel-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: var(--spacing-md);
}
.panel-subtitle {
    font-size: 0.75rem;
    color: var(--overlay1);
    font-weight: 400;
}

/* Timeline chart */
.timeline-chart-container {
    height: 200px;
    position: relative;
}

/* Three-across row */
.three-across {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-lg);
}
@media (max-width: 1000px) {
    .three-across { grid-template-columns: 1fr; }
}

/* Chart container (reused) */
.chart-container {
    height: 260px;
    position: relative;
}

/* Status bar */
.status-bar {
    display: flex;
    align-items: center;
    gap: var(--spacing-lg);
    padding: var(--spacing-sm) var(--spacing-lg);
    background: var(--base);
    border-radius: var(--radius-lg);
    margin-bottom: var(--spacing-lg);
    font-size: 0.8rem;
    color: var(--subtext0);
}
.status-bar-item {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
}
.status-bar-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--overlay0);
}
.status-bar-dot.ok { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-bar-dot.err { background: var(--red); box-shadow: 0 0 6px var(--red); }

/* ==============================================================================
   Detection card
   ============================================================================== */
.feed-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-md);
}
.feed-header .panel-title { margin-bottom: 0; }

.feed-list {
    max-height: 600px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.sse-banner {
    background: rgba(249, 226, 175, 0.15);
    color: var(--yellow);
    padding: var(--spacing-xs) var(--spacing-md);
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    text-align: center;
    margin-bottom: var(--spacing-sm);
}

.new-detections-pill {
    background: var(--blue);
    color: var(--crust);
    border: none;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.15s;
}
.new-detections-pill:hover { transform: scale(1.05); }

.hidden { display: none !important; }

.dcard {
    display: grid;
    grid-template-columns: 4px 1fr auto;
    gap: 0;
    background: var(--base);
    border: 1px solid rgba(205, 214, 244, 0.08);
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.dcard:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4); }
.dcard.benign { opacity: 0.55; }

.dcard .stripe { width: 4px; background: var(--overlay0); }
.dcard.critical .stripe { background: var(--red); }
.dcard.high .stripe { background: var(--peach); }
.dcard.medium .stripe { background: var(--yellow); }
.dcard.low .stripe { background: var(--green); }

.dcard .body {
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
}
.dcard .row1 { display: flex; align-items: center; gap: 8px; }
.dcard .attack { font-weight: 700; font-size: 0.85rem; color: var(--text); }
.dcard .sev {
    font-size: 0.625rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 3px;
    background: rgba(249, 226, 175, 0.15);
    color: var(--yellow);
}
.dcard.critical .sev { background: rgba(243, 139, 168, 0.18); color: var(--red); }
.dcard.high .sev { background: rgba(250, 179, 135, 0.18); color: var(--peach); }
.dcard.low .sev { background: rgba(166, 227, 161, 0.18); color: var(--green); }

.dcard .confbar {
    height: 3px;
    background: rgba(205, 214, 244, 0.08);
    border-radius: 2px;
    overflow: hidden;
}
.dcard .conffill {
    height: 100%;
    border-radius: 2px;
    background: var(--yellow);
}
.dcard.critical .conffill { background: var(--red); }
.dcard.high .conffill { background: var(--peach); }
.dcard.low .conffill { background: var(--green); }

.dcard .flow {
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 0.7rem;
    color: var(--subtext0);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.dcard .rightcol {
    padding: 10px 14px 10px 4px;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 2px;
    font-size: 0.7rem;
    color: var(--subtext0);
}
.dcard .confpct { font-weight: 700; color: var(--text); font-size: 0.85rem; }

/* Expanded state */
.dcard.expanded { grid-template-columns: 4px 1fr; }
.dcard.expanded .rightcol { display: none; }

.dcard .detail {
    grid-column: 2;
    padding: 0 14px 14px;
    border-top: 1px solid rgba(205, 214, 244, 0.08);
    padding-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.dcard .detail-title {
    font-size: 0.625rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 6px;
}

.dcard .shap-row {
    display: grid;
    grid-template-columns: 160px 1fr 55px;
    gap: 8px;
    align-items: center;
    font-size: 0.7rem;
}
.dcard .fname {
    font-family: 'Fira Code', monospace;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.dcard .shap-bar {
    height: 12px;
    background: rgba(205, 214, 244, 0.05);
    border-radius: 2px;
    position: relative;
    overflow: hidden;
}
.dcard .shap-fill-pos {
    position: absolute; top: 0; left: 50%; height: 100%;
    background: var(--peach);
    border-radius: 0 2px 2px 0;
}
.dcard .shap-fill-neg {
    position: absolute; top: 0; right: 50%; height: 100%;
    background: rgba(205, 214, 244, 0.15);
    border-radius: 2px 0 0 2px;
}
.dcard .shap-value {
    font-family: 'Fira Code', monospace;
    color: var(--subtext0);
    text-align: right;
}

.dcard .probs { display: flex; flex-direction: column; gap: 3px; }
.dcard .prob-row {
    display: grid;
    grid-template-columns: 160px 1fr 40px;
    gap: 8px;
    align-items: center;
    font-size: 0.7rem;
}
.dcard .pname { color: var(--subtext0); }
.dcard .prob-bar {
    height: 5px;
    background: rgba(205, 214, 244, 0.05);
    border-radius: 2px;
}
.dcard .prob-fill {
    height: 100%;
    background: var(--blue);
    border-radius: 2px;
}
.dcard .prob-value {
    font-family: 'Fira Code', monospace;
    color: var(--text);
    text-align: right;
    font-size: 0.625rem;
}

.dcard .agree { font-size: 0.7rem; color: var(--green); }
.dcard .agree.dis { color: var(--yellow); }

/* ==============================================================================
   Severity bars
   ============================================================================== */
.sev-row {
    display: grid;
    grid-template-columns: 70px 1fr 40px;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;
    font-size: 0.75rem;
}
.sev-label {
    text-transform: capitalize;
    color: var(--subtext0);
}
.sev-track {
    height: 8px;
    background: rgba(205, 214, 244, 0.06);
    border-radius: 4px;
    overflow: hidden;
}
.sev-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}
.sev-count {
    text-align: right;
    font-weight: 600;
    color: var(--text);
}

/* ==============================================================================
   Top talkers
   ============================================================================== */
.talker-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 0.75rem;
    border-bottom: 1px solid rgba(205, 214, 244, 0.06);
}
.talker-row:last-child { border-bottom: none; }
.talker-ip {
    font-family: 'Fira Code', monospace;
    color: var(--text);
}
.talker-count {
    font-weight: 600;
    color: var(--peach);
}
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/static/style.css
git commit -m "feat(dashboard): add detection card, feed, severity, talker, status-bar CSS"
```

---

### Task 9: Wire it all up and smoke test

**Files:**
- No new files; this task is integration verification

- [ ] **Step 1: Run backend tests**

```bash
cd C:\Users\folosiye\FYP-NIDS-ML
python -m pytest tests/test_recent_and_stream.py -v --timeout=30
```

Expected: all tests PASS.

- [ ] **Step 2: Start the API**

```bash
cd C:\Users\folosiye\FYP-NIDS-ML\api
NIDS_RATE_LIMIT_ENABLED=false python app.py
```

Verify it starts with "All models loaded successfully".

- [ ] **Step 3: Start the dashboard**

In a second terminal:
```bash
cd C:\Users\folosiye\FYP-NIDS-ML\dashboard
python app.py
```

- [ ] **Step 4: Open browser and verify cold load**

Open `http://localhost:5001`. Verify:
- Stats row shows zeros
- Timeline chart is empty but renders axes
- Feed says "Waiting for detections..."
- Status bar shows API OK, Models OK, SSE OK
- No console errors

- [ ] **Step 5: Run the demo simulator**

In a third terminal:
```bash
cd C:\Users\folosiye\FYP-NIDS-ML
python demo_traffic_simulator.py
```

Verify in browser:
- Detection cards stream in live via SSE
- Cards show colored severity stripe and IP flow
- Clicking a card expands SHAP + probabilities + agreement
- Stats row updates every 10s
- Doughnut chart populates
- Severity bars populate
- Top talkers populate
- Timeline chart shows benign/attack lines

- [ ] **Step 6: Test scroll behavior**

Scroll the feed down. Verify:
- New cards still arrive but a "N new detections" pill appears at top
- Clicking the pill scrolls back to top and clears counter

- [ ] **Step 7: Test notification permission**

Deny notification permission when prompted (or reset in browser settings). Verify:
- "Enable Alerts" button appears in header
- Clicking it re-requests permission

- [ ] **Step 8: Test disconnect recovery**

Kill the API (Ctrl+C). Verify:
- "Disconnected, retrying..." banner appears in feed
- SSE status dot goes red

Restart the API. Verify:
- Banner clears, SSE reconnects, detections resume

- [ ] **Step 9: Commit final state**

```bash
git add -A
git commit -m "feat(dashboard): complete Layout C dashboard refresh with SSE live feed"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | DetectionRecord + ring buffer + pub/sub | `api/app.py` |
| 2 | `/api/recent` + `/api/stream` endpoints | `api/app.py` |
| 3 | flow_id in demo simulator | `demo_traffic_simulator.py` |
| 4 | Backend tests | `tests/test_recent_and_stream.py` |
| 5 | Flask proxy routes | `dashboard/app.py` |
| 6 | index.html Layout C rewrite | `dashboard/templates/index.html` |
| 7 | app.js (SSE, cards, charts, notifications) | `dashboard/static/app.js` |
| 8 | CSS additions | `dashboard/static/style.css` |
| 9 | Integration smoke test | — |
