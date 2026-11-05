"""FastAPI prediction service.

Run with `nids serve` (or `uvicorn nids.api.app:app`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from nids import __version__, paths
from nids.api.auth import get_api_key, get_auth_status
from nids.api.rate_limit import (
    BATCH_RATE_LIMIT,
    PREDICT_RATE_LIMIT,
    SLOWAPI_AVAILABLE,
    get_rate_limit_status,
    limiter,
    rate_limit_exceeded_handler,
)
from nids.features import ATTACK_SEVERITY, FEATURE_NAMES_84
from nids.ml.model import NIDSModel

log = logging.getLogger("nids.api")

# ---------------------------------------------------------------------------- state


@dataclass
class ApiState:
    model: NIDSModel | None = None
    started: float = field(default_factory=time.time)
    total_predictions: int = 0
    total_attacks: int = 0
    attack_distribution: dict = field(default_factory=lambda: defaultdict(int))
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, label: str, is_attack: bool) -> None:
        self.total_predictions += 1
        if is_attack:
            self.total_attacks += 1
            self.attack_distribution[label] += 1


state = ApiState()


@dataclass
class DetectionRecord:
    """One detection, as shown in the dashboard's live feed."""

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
        return asdict(self)


_recent_detections: deque[DetectionRecord] = deque(maxlen=200)
_sse_subscribers: set[asyncio.Queue] = set()
_loop: asyncio.AbstractEventLoop | None = None


def _parse_flow_ips(flow_id: str | None) -> tuple[str, str]:
    """'192.168.1.1-10.0.0.5-6-443-80' -> ('192.168.1.1', '10.0.0.5')."""
    parts = (flow_id or "").split("-")
    return (parts[0], parts[1]) if len(parts) >= 2 else ("—", "—")


def _publish_detection(record: DetectionRecord) -> None:
    """Add to the ring buffer and fan out to live-feed subscribers (thread-safe)."""
    _recent_detections.appendleft(record)

    def fan_out():
        for q in list(_sse_subscribers):
            try:
                q.put_nowait(record)
            except asyncio.QueueFull:
                _sse_subscribers.discard(q)

    if _loop is not None and _loop.is_running():
        _loop.call_soon_threadsafe(fan_out)
    else:
        fan_out()


# ------------------------------------------------------------------------- schemas


class FlowRequest(BaseModel):
    features: dict[str, float | None] = Field(
        ..., description="All 84 CICFlowMeter features, by name. null = missing (the model imputes it)"
    )
    flow_id: str | None = Field(None, description="Optional 5-tuple id from the capture engine")

    @field_validator("features")
    @classmethod
    def _all_features(cls, v):
        missing = set(FEATURE_NAMES_84) - set(v)
        if missing:
            raise ValueError(f"missing {len(missing)} features, e.g. {sorted(missing)[:5]}")
        return v


class FlowVector(BaseModel):
    features: list[float | None] = Field(..., min_length=84, max_length=84, description="84 values in /features order")


class BatchRequest(BaseModel):
    flows: list[FlowVector] = Field(..., min_length=1, max_length=1000)


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictionResponse(BaseModel):
    prediction: str
    prediction_id: int
    confidence: float
    probabilities: dict[str, float]
    top_features: list[FeatureContribution]
    is_attack: bool
    severity: str
    model_agreement: bool
    timestamp: str


class BatchResponse(BaseModel):
    predictions: list[dict]
    total_flows: int
    attack_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    uptime_seconds: float
    version: str


class StatsResponse(BaseModel):
    total_predictions: int
    total_attacks: int
    attack_rate: float
    attack_distribution: dict[str, int]
    avg_prediction_time_ms: float
    uptime_seconds: float


# ----------------------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    if state.model is None:  # tests may inject a model before startup
        state.model = NIDSModel.load(paths.MODELS_DIR)
        log.info("Loaded model: %d classes, RF weight %.2f", len(state.model.classes), state.model.rf_weight)
    state.started = time.time()
    if METRICS:
        METRICS["models_loaded"].set(1)
    yield
    if METRICS:
        METRICS["models_loaded"].set(0)


app = FastAPI(
    title="NIDS — Network Intrusion Detection API",
    description=(
        "Classifies network flows (84 CICFlowMeter features) as benign or one of 13 attack types "
        "using a Random Forest + XGBoost ensemble, with SHAP explanations."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("NIDS_CORS_ORIGINS", "http://localhost:5001,http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
if SLOWAPI_AVAILABLE:
    from slowapi.errors import RateLimitExceeded

    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


def _setup_metrics() -> dict:
    try:
        from prometheus_client import Counter, Gauge, Histogram
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        return {}
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    return {
        "predictions": Counter("nids_predictions_total", "Predictions made", ["prediction_class", "is_attack"]),
        "latency": Histogram("nids_prediction_latency_seconds", "Prediction latency",
                             buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        "models_loaded": Gauge("nids_models_loaded", "1 when the model is loaded"),
        "attacks": Counter("nids_attacks_detected_total", "Attacks detected", ["attack_type"]),
    }


METRICS = _setup_metrics()


def _model() -> NIDSModel:
    if state.model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded")
    return state.model


def _observe(label: str, is_attack: bool, seconds: float | None = None) -> None:
    state.record(label, is_attack)
    if METRICS:
        METRICS["predictions"].labels(prediction_class=label, is_attack=str(is_attack).lower()).inc()
        if is_attack:
            METRICS["attacks"].labels(attack_type=label).inc()
        if seconds is not None:
            METRICS["latency"].observe(seconds)


# ----------------------------------------------------------------------- endpoints


@app.get("/", tags=["General"])
def root():
    return {
        "service": "NIDS prediction API",
        "version": __version__,
        "docs": "/docs",
        "classes": state.model.classes if state.model else [],
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    loaded = state.model is not None
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        models_loaded=loaded,
        uptime_seconds=time.time() - state.started,
        version=__version__,
    )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
def stats():
    n = state.total_predictions
    lat = state.latencies_ms
    return StatsResponse(
        total_predictions=n,
        total_attacks=state.total_attacks,
        attack_rate=state.total_attacks / n if n else 0.0,
        attack_distribution=dict(state.attack_distribution),
        avg_prediction_time_ms=sum(lat) / len(lat) if lat else 0.0,
        uptime_seconds=time.time() - state.started,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit(PREDICT_RATE_LIMIT)
def predict(
    request: Request,
    flow: FlowRequest,
    explain: bool = Query(True, description="Include the top-5 SHAP feature contributions"),
    api_key: str | None = Depends(get_api_key),
):
    """Classify one flow. Set `explain=false` to skip SHAP when you only need the label."""
    model = _model()
    t0 = time.perf_counter()
    X = pd.DataFrame([flow.features], columns=FEATURE_NAMES_84, dtype="float64")
    p = model.predict_flows(X)[0]
    top = model.explain(X, p.label) if explain else []
    elapsed = time.perf_counter() - t0
    state.latencies_ms.append(elapsed * 1000)
    _observe(p.label, p.is_attack, elapsed)

    severity = ATTACK_SEVERITY.get(p.label, "medium")
    now = datetime.now().isoformat()
    src_ip, dst_ip = _parse_flow_ips(flow.flow_id)
    _publish_detection(
        DetectionRecord(
            id=str(uuid.uuid4()), timestamp=now, flow_id=flow.flow_id or "", src_ip=src_ip, dst_ip=dst_ip,
            prediction=p.label, confidence=p.confidence, is_attack=p.is_attack, severity=severity,
            model_agreement=p.models_agree, top_features=top, probabilities=p.probabilities,
        )
    )
    if p.is_attack:
        log.warning("ATTACK %s (%.2f) flow=%s", p.label, p.confidence, flow.flow_id)
    return PredictionResponse(
        prediction=p.label, prediction_id=model.classes.index(p.label), confidence=p.confidence,
        probabilities=p.probabilities, top_features=top, is_attack=p.is_attack, severity=severity,
        model_agreement=p.models_agree, timestamp=now,
    )


@app.post("/batch_predict", response_model=BatchResponse, tags=["Prediction"])
@limiter.limit(BATCH_RATE_LIMIT)
def batch_predict(request: Request, batch: BatchRequest, api_key: str | None = Depends(get_api_key)):
    """Classify up to 1000 flows, each as 84 values in `/features` order."""
    model = _model()
    t0 = time.perf_counter()
    X = pd.DataFrame([f.features for f in batch.flows], columns=FEATURE_NAMES_84, dtype="float64")
    preds = model.predict_flows(X)
    elapsed = time.perf_counter() - t0
    state.latencies_ms.append(elapsed * 1000 / len(preds))
    out = []
    for i, p in enumerate(preds):
        _observe(p.label, p.is_attack)
        out.append({
            "flow_index": i, "prediction": p.label, "prediction_id": model.classes.index(p.label),
            "confidence": p.confidence, "is_attack": p.is_attack, "model_agreement": p.models_agree,
        })
    if METRICS:
        METRICS["latency"].observe(elapsed)
    return BatchResponse(
        predictions=out, total_flows=len(out), attack_count=sum(p.is_attack for p in preds),
        processing_time_ms=elapsed * 1000,
    )


@app.get("/features", tags=["Model"])
def features():
    model = state.model
    return {
        "total_features": len(FEATURE_NAMES_84),
        "feature_names": FEATURE_NAMES_84,
        "selected_feature_names": model.selected_features if model else [],
        "description": "Send all 84 features; the model picks the ones it was trained on.",
    }


@app.get("/classes", tags=["Model"])
def classes():
    model = _model()
    return {
        "total_classes": len(model.classes),
        "classes": [
            {"id": i, "name": c, "is_attack": c != "BENIGN", "severity": ATTACK_SEVERITY.get(c, "medium")}
            for i, c in enumerate(model.classes)
        ],
    }


@app.get("/model", tags=["Model"])
def model_card():
    """Training metadata plus the held-out test results, if `nids evaluate` has been run."""
    model = _model()
    card = {"rf_weight": model.rf_weight, "metadata": model.metadata}
    metrics_file = paths.REPORTS_DIR / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        card["test_results"] = {"test_rows": m["test_rows"], "models": m["models"], "latency": m["latency"]}
    return card


@app.get("/security", tags=["General"])
def security():
    return {
        "authentication": get_auth_status(),
        "rate_limiting": get_rate_limit_status(),
        "prometheus_metrics": {"enabled": bool(METRICS), "endpoint": "/metrics" if METRICS else None},
    }


@app.get("/api/recent", tags=["Live feed"])
def recent(limit: int = 50, severity: str | None = None):
    limit = min(max(limit, 1), 200)
    records = list(_recent_detections)
    if severity:
        records = [r for r in records if r.severity == severity.lower()]
    return [r.to_dict() for r in records[:limit]]


@app.get("/api/stream", tags=["Live feed"])
async def stream():
    """Server-sent events: one message per detection."""
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _sse_subscribers.add(q)

    async def events():
        try:
            while True:
                try:
                    record = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(record.to_dict())}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sse_subscribers.discard(q)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
