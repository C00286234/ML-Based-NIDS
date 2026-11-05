# Dashboard Refresh + Alerting Configuration — Design

**Date:** 2026-04-15
**Status:** Approved, pending implementation plan
**Scope:** Refresh the Flask dashboard at `dashboard/app.py` into a dual-purpose VIVA showpiece + operator console, add small backing changes to `api/app.py` to support real-time detection streaming, and wire up in-dashboard alerting.

---

## 1. Goals and non-goals

**Goals**
- Make the dashboard legible as both a VIVA demo (top-to-bottom narrative for an examiner) and as an operator triage surface (live detections with SHAP explainability).
- Stream detections to the browser in real time instead of 5-second polling.
- Surface alerting in the browser itself — no email/Slack config required for the demo.
- Reuse existing backend primitives (`src/alerting.py` severity map, SHAP explainer, model-agreement field) rather than introducing parallel ones.

**Non-goals**
- Email / Slack / SMS channels. `src/alerting.py` supports them; they stay disabled.
- Alert acknowledgement, assignment, or persistence. No operator state.
- Authentication on the dashboard. Single-operator demo deployment.
- Load testing the SSE endpoint. Single consumer.
- Model or training changes.

---

## 2. Role framing

The dashboard serves two audiences simultaneously:
- **Leaders / examiner** — wants a one-screen story: "the system is working, here is what it sees, here is the evidence." Reads top-to-bottom.
- **Security analyst** — wants the live feed, the ability to click into a detection, and enough explainability (SHAP + class probabilities + model agreement) to decide whether to trust a call.

A single layout satisfies both because the narrative ordering (stats → pulse → evidence → breakdown → health) is also the order an analyst naturally scans.

---

## 3. Architecture

```
┌────────────────────┐         SSE stream          ┌──────────────────────┐
│  api/app.py        │ ──────────────────────────▶ │  dashboard/app.py    │
│  (FastAPI, 8000)   │         /api/stream         │  (Flask, 5001)       │
│                    │                             │                      │
│  - /predict        │         HTTP proxy          │  - /                 │
│  - /api/recent     │ ◀────────────────────────── │  - /api/status       │
│  - /api/stream     │                             │  - /api/recent       │
│  - ring buffer 200 │                             │  - /api/stream       │
│  - asyncio.Queue   │                             │    (proxied)         │
│    pub/sub         │                             │                      │
└────────────────────┘                             └──────────────────────┘
                                                              │
                                                              ▼
                                                   ┌──────────────────────┐
                                                   │  Browser             │
                                                   │  - EventSource       │
                                                   │  - Notification API  │
                                                   │  - Chart.js          │
                                                   └──────────────────────┘
```

**Real-time transport: Server-Sent Events**
Chosen over WebSockets and polling because:
- One-way server→client is exactly what we need.
- Plain HTTP, works through the Flask proxy without special handling.
- Browser `EventSource` handles reconnect automatically.
- No extra dependency beyond what FastAPI already ships.

**Data flow**
1. `demo_traffic_simulator.py` (or a real capture) POSTs a flow to `/predict`.
2. `api/app.py` runs the ensemble, builds a `DetectionRecord`, appends it to a `deque(maxlen=200)`, and publishes to an `asyncio.Queue`-based fan-out.
3. Any client connected to `/api/stream` receives the record as an SSE event.
4. The browser renders a new detection card, updates the doughnut and stats row, and — if severity ≥ medium — fires a browser notification.

---

## 4. Backend changes (`api/app.py`)

### 4.1 DetectionRecord schema

```python
@dataclass
class DetectionRecord:
    id: str              # prediction_id, already generated per predict
    timestamp: str       # ISO 8601
    flow_id: str         # 5-tuple string from request
    src_ip: str          # parsed from flow_id
    dst_ip: str          # parsed from flow_id
    prediction: str      # e.g. "DoS Hulk"
    confidence: float    # 0.0–1.0
    is_attack: bool
    severity: str        # "critical" | "high" | "medium" | "low"
    model_agreement: bool
    top_features: list   # SHAP top 5, existing shape
    probabilities: dict  # class → prob, existing shape
```

Severity is computed from the existing `ATTACK_SEVERITY` dict in `src/alerting.py` so the dashboard and any future alerting channel stay consistent.

### 4.2 Ring buffer + pub/sub

Module-level state in `api/app.py`:

```python
from collections import deque
import asyncio

_recent: deque[DetectionRecord] = deque(maxlen=200)
_subscribers: set[asyncio.Queue] = set()
```

After each successful `/predict`, build the record, `_recent.append(record)`, and `put_nowait` on every subscriber queue. `put_nowait` is intentional — if a subscriber is slow, we'd rather drop its message than block the prediction path. A slow subscriber's queue will fill, and we evict on overflow.

### 4.3 New endpoints

**`GET /api/recent`**
- Query params: `limit` (default 50, max 200), `severity` (optional filter: critical/high/medium/low)
- Returns: JSON array of `DetectionRecord` in reverse-chronological order
- Used on initial dashboard load to seed the feed before SSE takes over

**`GET /api/stream`**
- SSE endpoint. On connect, creates a new `asyncio.Queue`, adds it to `_subscribers`, and streams events as they arrive.
- Content-Type: `text/event-stream`
- Each event: `data: <json>\n\n`
- On disconnect, removes the queue from `_subscribers`.
- Heartbeat comment (`: keepalive\n\n`) every 15s so intermediaries don't close the connection.

### 4.4 What the backend does NOT change

- `/predict`, `/batch_predict`, `/health`, `/stats`, `/security`, `/classes`, `/features`, `/metrics` — untouched.
- SHAP explainer, model loading, rate limiting — untouched.
- No database. The ring buffer lives in memory and resets on restart, which matches the rest of `api/app.py`.

---

## 5. Frontend changes (`dashboard/`)

### 5.1 File layout

```
dashboard/
  app.py                    # add /api/recent and /api/stream proxies
  templates/
    index.html              # rewritten, Layout C structure
  static/
    style.css               # extended with new component classes
    app.js                  # new file — all dashboard JS lives here
```

Moving the JS out of the template into `static/app.js` because the current inline script is already ~150 lines and we're adding SSE handling, notification logic, and card rendering on top of it.

### 5.2 Layout C — narrative flow

Top-to-bottom, full-width panels so the eye has one job at a time:

1. **Stats row** — Total Predictions, Attacks Detected, Avg Latency, Uptime (existing 4 cards, reused)
2. **Traffic Timeline** — 10-minute rolling line chart, predictions/sec with attack events overlaid as dots
3. **Live Detection Feed** — the main event, expandable SHAP cards (see 5.3)
4. **Three-across row** — Distribution (doughnut, existing), Severity breakdown (new), Top Talkers (stretch goal)
5. **System Status bar** — compact one-line strip: API ✓ | Models ✓ | SSE ✓ | Uptime 2h 14m

### 5.3 Detection card

Mockup approved via visual companion. Two states:

**Collapsed (default)**
- 4px left stripe coloured by severity (critical pink, high orange, medium yellow, low green)
- Attack name + severity chip
- Thin confidence bar
- Flow: `src_ip → dst_ip` in monospace
- Right column: confidence %, "N seconds ago"
- BENIGN cards render at 55% opacity so they're visible but recede

**Expanded (on click)**
- Top SHAP contributions: feature name + directional bar (positive pushes toward attack, negative pulls away) + signed value
- Class probabilities: top 3 classes with bar + percentage
- Model agreement line: "✓ Random Forest and XGBoost agree" (green) or "⚠ Models disagree" (yellow)

### 5.4 Feed scroll behaviour

- Auto-scroll to top when a new detection arrives **only if** the user is already at the top.
- If the user has scrolled down to inspect a card, new arrivals surface a "N new detections ↑" pill at the top of the feed instead of stealing their scroll position. Clicking the pill scrolls to top and clears the counter.

### 5.5 Connection state

A small banner at the top of the feed when the SSE connection drops: "Disconnected, retrying…" in yellow. Clears when `EventSource` reconnects.

---

## 6. Alerting (in-browser only)

Two channels with different thresholds so the browser doesn't nag the operator into muting notifications.

### 6.1 In-dashboard feed (channel 1)

- **Threshold:** everything including BENIGN
- **Behaviour:** detections appear as cards in the live feed
- **Rate limiting:** none — the ring buffer itself caps growth at 200

### 6.2 Browser notifications (channel 2)

- **Threshold:** medium severity and above (Web Attack classes, Portscan, DoS family, DDoS, brute force, bot, Heartbleed, Infiltration)
- **Permission:** requested once on first dashboard load. If denied, a small "Enable browser alerts" button appears in the header for later opt-in.
- **Content:** title = attack name, body = `src_ip → dst_ip` + confidence. Click focuses the dashboard tab and scrolls to that card.
- **Rate limit:** max 1 notification per 3 seconds. Overflow within the window is coalesced into a single "N more critical detections" notification.

### 6.3 Severity mapping

Reuses the existing `ATTACK_SEVERITY` dict from `src/alerting.py`:

- **Critical:** DoS Hulk, DDoS, DoS GoldenEye, Heartbleed, Infiltration
- **High:** Portscan, SSH-Patator, FTP-Patator, DoS Slowloris, DoS Slowhttptest, Bot
- **Medium:** Web Attack – Brute Force, Web Attack – XSS, Web Attack – SQL Injection
- **Low:** BENIGN

### 6.4 Explicitly excluded

- Email alerts — `src/alerting.py` supports them, stays disabled in `config/nids_config.yaml`.
- Slack alerts — same.
- Alert acknowledgement / dismissal — no operator state to persist.

---

## 7. Testing

### 7.1 Backend (pytest)

New file `tests/test_recent_and_stream.py`:

- `/api/recent` returns empty list when ring buffer is empty
- `/api/recent` returns last N in reverse-chronological order after seeding
- `/api/recent?severity=critical` filters correctly
- Ring buffer caps at 200 — push 250 records, assert length is 200 and oldest is evicted
- `/api/stream` SSE: connect, POST a prediction, assert the event arrives on the stream within 1s
- `DetectionRecord` serialization round-trips (dict → JSON → dict)

### 7.2 Frontend (manual smoke test)

Checklist run in the browser before declaring done. No Jest/Playwright — not worth the setup for a solo FYP.

- [ ] Load dashboard cold → recent feed populates from `/api/recent`
- [ ] Run `demo_traffic_simulator.py` → new cards stream in via SSE, stats row updates, doughnut updates
- [ ] Click a critical card → expands with SHAP bars and class probabilities
- [ ] Scroll feed down → new arrivals show "N new ↑" pill instead of jumping scroll
- [ ] Deny notification permission → header shows "Enable alerts" button
- [ ] Kill the API → dashboard shows "Disconnected, retrying…" banner
- [ ] Restart the API → banner clears, SSE auto-reconnects

### 7.3 Not tested

- `/predict`, `/stats`, `/health` — unchanged, already covered
- Model accuracy — nothing in this change touches the model
- SSE load testing — single operator, not a concern

---

## 8. Open questions

None. All design decisions locked during brainstorming on 2026-04-15.

---

## 9. Related work from this session (context)

This design follows a cleanup session where the following was also done, relevant because it established the current state of the codebase the dashboard refresh builds on:

- Models retrained via `04_model_training_improved.py` (final test accuracy 99.41%) after two SMOTE bugs were fixed (removed invalid `n_jobs` kwarg, rewrote moderate/conservative `target_counts` to skip classes where the cap would shrink the count).
- Redundant `03_feature_engineering.py` deleted — its responsibilities already live in `04_model_training_improved.py`, and running it was corrupting the scaler/encoder/selector timestamps so they no longer matched the trained models.
- Demo simulator error storm root-caused to Docker-container rate limiting (source IP `172.18.0.1`, the Docker bridge gateway, hitting 60/min). Fixed by running the API natively with `NIDS_RATE_LIMIT_ENABLED=false` and adding that env var to `docker-compose.yml` for future Docker runs.
- 7.1 GB of cached `.npy` files moved out of `data/processed/` into `C:\Users\folosiye\` — they're training intermediates that don't belong in the project tree.
- Stale "next step: `03_feature_engineering.py`" reference in `02_data_preprocessing.py:233` updated to point at `04_model_training_improved.py`. Equivalent stale references in `README.md`, `SYSTEM_EXPLANATION.md`, and `VIVA_PREPARATION.md` are still pending.
