"""
Network Intrusion Detection System (NIDS) - FastAPI Application
================================================================
A production-ready API for classifying network traffic flows into 16 attack types
using Random Forest and XGBoost models with SHAP explainability.

Author: C00286234
"""

# --- Standard library imports ---
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
from contextlib import asynccontextmanager

# --- Third-party imports ---
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --- Local imports for security features ---
from auth import get_api_key, get_auth_status, is_auth_enabled
from rate_limit import (
    limiter, rate_limit_exceeded_handler, is_rate_limit_enabled,
    get_rate_limit_status, SLOWAPI_AVAILABLE
)
if SLOWAPI_AVAILABLE:
    from slowapi.errors import RateLimitExceeded

# --- Path configuration: resolve project root and key directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "api", "logs")

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Configure logging for the application with both file and console handlers."""

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    simple_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # Main app logger
    app_logger = logging.getLogger("nids_api")
    app_logger.setLevel(logging.INFO)

    # File handler for general logs
    general_file_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, "api.log"), encoding="utf-8"
    )
    general_file_handler.setLevel(logging.INFO)
    general_file_handler.setFormatter(detailed_formatter)
    app_logger.addHandler(general_file_handler)

    # Predictions logger
    predictions_logger = logging.getLogger("predictions")
    predictions_logger.setLevel(logging.INFO)
    predictions_file_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, "predictions.log"), encoding="utf-8"
    )
    predictions_file_handler.setFormatter(detailed_formatter)
    predictions_logger.addHandler(predictions_file_handler)

    # Attacks logger
    attacks_logger = logging.getLogger("attacks")
    attacks_logger.setLevel(logging.INFO)
    attacks_file_handler = logging.FileHandler(
        os.path.join(LOGS_DIR, "attacks.log"), encoding="utf-8"
    )
    attacks_file_handler.setFormatter(detailed_formatter)
    attacks_logger.addHandler(attacks_file_handler)

    return app_logger, predictions_logger, attacks_logger

# Initialize loggers
logger, predictions_logger, attacks_logger = setup_logging()

# ============================================================================
# FEATURE DEFINITIONS
# Full 84-feature list from constants; top-50 loaded from training output.
# ============================================================================

from src.constants import FEATURE_NAMES_84

# 50 selected features (loaded dynamically from training output)
def _load_selected_features():
    """Load feature names from selected_features.txt, fall back to hardcoded list."""
    features_file = os.path.join(MODELS_DIR, "..", "data", "processed", "selected_features.txt")
    features_file = os.path.normpath(features_file)
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        if features:
            return features
    # Fallback if file not found
    return [
        "Dst Port", "Protocol", "Flow Duration", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Packets/s",
        "Flow IAT Std", "Flow IAT Max", "Fwd IAT Total", "Fwd IAT Std", "Fwd IAT Max",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Fwd PSH Flags",
        "Fwd RST Flags", "Bwd RST Flags", "Fwd Packets/s", "Bwd Packets/s",
        "Packet Length Min", "Packet Length Max", "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "Down/Up Ratio", "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg",
        "Bwd Bulk Rate Avg", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Bytes",
        "FWD Init Win Bytes", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean",
        "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max"
    ]

SELECTED_FEATURES_50 = _load_selected_features()

# ============================================================================
# GLOBAL STATE
# Singleton container for loaded ML models and runtime metrics.
# ============================================================================

class ModelState:
    """Global state container for ML models and metrics."""

    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.shap_explainer = None
        self.startup_time: Optional[datetime] = None

        # Metrics tracking
        self.total_predictions = 0
        self.total_attacks = 0
        self.attack_distribution: Dict[str, int] = defaultdict(int)
        self.prediction_times: deque = deque(maxlen=1000)

state = ModelState()

# ============================================================================
# PYDANTIC MODELS
# Request/response schemas for FastAPI endpoint validation.
# ============================================================================

class NetworkFlowFeatures(BaseModel):
    """
    Network flow features for prediction.
    Accepts 84 features as a dictionary.
    """
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of 84 network flow features",
        json_schema_extra={
            "example": {
                "Src Port": 443,
                "Dst Port": 80,
                "Protocol": 6,
                "Flow Duration": 1000000,
                "Total Fwd Packet": 10
            }
        }
    )
    flow_id: Optional[str] = Field(
        None,
        description="Optional flow identifier (5-tuple) from the NIDS engine"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        """Validate that we have the required features."""
        if len(v) != 84:
            raise ValueError(f"Expected 84 features, got {len(v)}")

        missing = set(FEATURE_NAMES_84) - set(v.keys())
        if missing:
            raise ValueError(f"Missing features: {list(missing)[:5]}...")

        return v


class NetworkFlowList(BaseModel):
    """List of network flow features for batch prediction."""
    features: List[float] = Field(
        ...,
        description="List of 84 feature values in order",
        min_length=84,
        max_length=84
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request with multiple flows."""
    flows: List[NetworkFlowList] = Field(
        ...,
        description="List of network flows to predict",
        min_length=1,
        max_length=1000
    )


class FeatureContribution(BaseModel):
    """SHAP feature contribution for explainability."""
    feature: str
    value: float
    contribution: float


class PredictionResponse(BaseModel):
    """Response for single prediction endpoint."""
    prediction: str
    prediction_id: int
    confidence: float
    probabilities: Dict[str, float]
    top_features: List[FeatureContribution]
    is_attack: bool
    model_agreement: bool
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction endpoint."""
    predictions: List[Dict[str, Any]]
    total_flows: int
    attack_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    uptime_seconds: float
    version: str


class MetricsResponse(BaseModel):
    """API metrics response."""
    total_predictions: int
    total_attacks: int
    attack_rate: float
    attack_distribution: Dict[str, int]
    avg_prediction_time_ms: float
    uptime_seconds: float


# ============================================================================
# MODEL LOADING / UNLOADING
# Load all ML artefacts at startup; release them on shutdown.
# ============================================================================

def load_models():
    """Load all ML models and preprocessing objects at startup."""

    logger.info("Loading ML models...")

    try:
        # Load Random Forest model
        rf_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
        state.rf_model = joblib.load(rf_path)
        logger.info(f"Loaded Random Forest model from {rf_path}")

        # Load XGBoost model
        xgb_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
        state.xgb_model = joblib.load(xgb_path)
        logger.info(f"Loaded XGBoost model from {xgb_path}")

        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        state.scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")

        # Load feature selector
        selector_path = os.path.join(MODELS_DIR, "feature_selector.pkl")
        state.feature_selector = joblib.load(selector_path)
        logger.info(f"Loaded feature selector from {selector_path}")

        # Load label encoder
        encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
        state.label_encoder = joblib.load(encoder_path)
        logger.info(f"Loaded label encoder from {encoder_path}")
        logger.info(f"Attack classes: {list(state.label_encoder.classes_)}")

        # Initialize SHAP explainer with XGBoost (better performing model)
        try:
            import shap
            state.shap_explainer = shap.TreeExplainer(state.xgb_model)
            logger.info("SHAP TreeExplainer initialized (XGBoost)")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            state.shap_explainer = None

        state.startup_time = datetime.now()
        logger.info("All models loaded successfully")

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def unload_models():
    """Clean up models on shutdown."""
    logger.info("Shutting down API, cleaning up resources...")
    state.rf_model = None
    state.xgb_model = None
    state.scaler = None
    state.feature_selector = None
    state.label_encoder = None
    state.shap_explainer = None


# ============================================================================
# FASTAPI APP INITIALIZATION
# Lifespan manager, app instance, CORS middleware, and rate-limiter wiring.
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load models on startup, cleanup on shutdown."""
    load_models()

    # Update Prometheus gauge
    if PROMETHEUS_AVAILABLE and MODELS_LOADED:
        MODELS_LOADED.set(1 if state.rf_model is not None else 0)

    # Initialize Prometheus instrumentator
    if PROMETHEUS_AVAILABLE and instrumentator:
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
        logger.info("Prometheus metrics exposed at /metrics")

    yield
    unload_models()

    # Update Prometheus gauge on shutdown
    if PROMETHEUS_AVAILABLE and MODELS_LOADED:
        MODELS_LOADED.set(0)


app = FastAPI(
    title="NIDS - Network Intrusion Detection API",
    description="""
## Network Intrusion Detection System API

This API provides real-time network traffic classification using Machine Learning models.

### Features:
- **16 Attack Classes**: Classifies traffic into BENIGN or 15 attack types
- **High Accuracy**: 99.4% accuracy using Random Forest
- **Explainability**: SHAP-based feature contributions for each prediction
- **Batch Processing**: Efficient batch prediction endpoint

### Attack Types Detected:
- DoS (Hulk, GoldenEye, Slowloris, Slowhttptest)
- DDoS
- Portscan
- Brute Force (FTP, SSH, Web)
- Web Attacks (XSS, SQL Injection)
- Botnet, Infiltration, Heartbleed
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
# Note: allow_origins=["*"] with allow_credentials=True violates the CORS spec
# and browsers will reject the response. Use explicit origins instead.
CORS_ORIGINS = os.getenv("NIDS_CORS_ORIGINS", "http://localhost:5001,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
if SLOWAPI_AVAILABLE:
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# ==============================================================================
# PROMETHEUS METRICS (optional)
# Custom counters, histograms, and gauges for observability.
# ==============================================================================

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    from prometheus_client import Counter, Histogram, Gauge

    # Custom metrics
    PREDICTIONS_COUNTER = Counter(
        "nids_predictions_total",
        "Total number of predictions made",
        ["prediction_class", "is_attack"]
    )
    PREDICTION_LATENCY = Histogram(
        "nids_prediction_latency_seconds",
        "Prediction latency in seconds",
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    )
    MODELS_LOADED = Gauge(
        "nids_models_loaded",
        "Whether ML models are loaded (1=yes, 0=no)"
    )
    ACTIVE_ATTACKS = Counter(
        "nids_attacks_detected_total",
        "Total attacks detected",
        ["attack_type"]
    )

    # Initialize instrumentator
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        inprogress_name="nids_requests_inprogress",
        inprogress_labels=True,
    )

    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus metrics enabled")

except ImportError:
    PROMETHEUS_AVAILABLE = False
    PREDICTIONS_COUNTER = None
    PREDICTION_LATENCY = None
    MODELS_LOADED = None
    ACTIVE_ATTACKS = None
    instrumentator = None
    logger.info("Prometheus metrics disabled (install prometheus-fastapi-instrumentator)")


# ============================================================================
# HELPER FUNCTIONS
# Preprocessing, SHAP explainability, and prediction logging utilities.
# ============================================================================

def preprocess_features(features_dict: Dict[str, float]) -> np.ndarray:
    """
    Preprocess input features: scale and select top 50 features.

    Args:
        features_dict: Dictionary of 84 feature names to values

    Returns:
        np.ndarray: Preprocessed features ready for prediction (1, 50)
    """
    # Create DataFrame with correct column order
    df = pd.DataFrame([features_dict], columns=FEATURE_NAMES_84)

    # Scale features
    scaled = state.scaler.transform(df)

    # Select top 50 features
    selected = state.feature_selector.transform(scaled)

    return selected


def preprocess_features_batch(features_list: List[List[float]]) -> np.ndarray:
    """
    Preprocess batch of features.

    Args:
        features_list: List of feature lists (each with 84 values)

    Returns:
        np.ndarray: Preprocessed features (n_samples, 50)
    """
    df = pd.DataFrame(features_list, columns=FEATURE_NAMES_84)
    scaled = state.scaler.transform(df)
    selected = state.feature_selector.transform(scaled)
    return selected


def get_shap_explanation(features: np.ndarray, prediction_idx: int) -> List[FeatureContribution]:
    """
    Get SHAP explanation for a prediction.

    Args:
        features: Preprocessed features (1, 50)
        prediction_idx: Index of predicted class

    Returns:
        List of top 5 feature contributions
    """
    if state.shap_explainer is None:
        return []

    try:
        # Calculate SHAP values
        shap_values = state.shap_explainer.shap_values(features)

        # Handle different SHAP output formats
        # SHAP can return different formats depending on version and model
        if isinstance(shap_values, list):
            # Multi-class: list of arrays, one per class
            # Each element is shape (n_samples, n_features)
            class_shap = np.array(shap_values[int(prediction_idx)]).flatten()
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # Shape: (n_samples, n_features, n_classes)
                class_shap = shap_values[0, :, int(prediction_idx)]
            elif shap_values.ndim == 2:
                # Shape: (n_samples, n_features) - binary or single class
                class_shap = shap_values[0]
            else:
                class_shap = shap_values.flatten()
        else:
            # Unknown format - try to extract values
            class_shap = np.array(shap_values).flatten()

        # Ensure class_shap is 1D with correct length
        n_features = len(SELECTED_FEATURES_50)
        if len(class_shap) != n_features:
            # If size doesn't match, truncate or pad with zeros
            if len(class_shap) > n_features:
                class_shap = class_shap[:n_features]
            else:
                class_shap = np.pad(class_shap, (0, n_features - len(class_shap)))

        # Get top 5 contributing features by absolute value
        feature_indices = np.argsort(np.abs(class_shap))[::-1][:5]

        contributions = []
        for idx in feature_indices:
            idx = int(idx)
            # Guard against out-of-bounds access
            if idx >= n_features or idx >= features.shape[1]:
                continue
            contributions.append(FeatureContribution(
                feature=SELECTED_FEATURES_50[idx],
                value=float(features[0, idx]),
                contribution=float(class_shap[idx])
            ))

        return contributions

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return []


def log_prediction(prediction: str, confidence: float, is_attack: bool, flow_id: str = None):
    """Log prediction to appropriate log files."""

    log_msg = f"flow_id={flow_id or 'N/A'} | prediction={prediction} | confidence={confidence:.4f}"
    predictions_logger.info(log_msg)

    if is_attack:
        attacks_logger.warning(f"ATTACK DETECTED | {log_msg}")


# ============================================================================
# API ENDPOINTS
# Root, health, stats, predict, batch_predict, features, classes, security.
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Welcome endpoint with API information and available endpoints.
    """
    return {
        "message": "Welcome to the Network Intrusion Detection System API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This welcome message",
            "GET /health": "Health check endpoint",
            "GET /stats": "API usage statistics",
            "POST /predict": "Single flow prediction with SHAP explanation",
            "POST /batch_predict": "Batch prediction for multiple flows",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)"
        },
        "attack_classes": list(state.label_encoder.classes_) if state.label_encoder else [],
        "model": "Random Forest (99.4% accuracy)"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint to verify API and model status.
    """
    models_loaded = all([
        state.rf_model is not None,
        state.xgb_model is not None,
        state.scaler is not None,
        state.feature_selector is not None,
        state.label_encoder is not None
    ])

    uptime = 0.0
    if state.startup_time:
        uptime = (datetime.now() - state.startup_time).total_seconds()

    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        uptime_seconds=uptime,
        version="1.0.0"
    )


@app.get("/stats", response_model=MetricsResponse, tags=["General"])
async def get_metrics():
    """
    Get API usage statistics and metrics.
    """
    uptime = 0.0
    if state.startup_time:
        uptime = (datetime.now() - state.startup_time).total_seconds()

    attack_rate = 0.0
    if state.total_predictions > 0:
        attack_rate = state.total_attacks / state.total_predictions

    avg_time = 0.0
    if state.prediction_times:
        avg_time = sum(state.prediction_times) / len(state.prediction_times)

    return MetricsResponse(
        total_predictions=state.total_predictions,
        total_attacks=state.total_attacks,
        attack_rate=attack_rate,
        attack_distribution=dict(state.attack_distribution),
        avg_prediction_time_ms=avg_time,
        uptime_seconds=uptime
    )


def _ensemble_proba(rf_model, xgb_model, X):
    """Compute soft-voting ensemble probabilities from RF and XGBoost."""
    rf_probs = rf_model.predict_proba(X)
    xgb_probs = xgb_model.predict_proba(X)
    return rf_probs, xgb_probs, (rf_probs + xgb_probs) / 2.0


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("60/minute")
async def predict(
    request: Request,
    flow: NetworkFlowFeatures,
    api_key: Optional[str] = Depends(get_api_key)
):
    """
    Predict the class of a single network flow.

    Accepts 84 network flow features and returns:
    - Predicted class (BENIGN or attack type)
    - Confidence score
    - Probability distribution across all classes
    - Top 5 SHAP feature contributions

    **Authentication**: Required if NIDS_REQUIRE_AUTH=true (X-API-Key header)
    **Rate Limit**: 60 requests per minute per IP

    Example request body:
    ```json
    {
        "features": {
            "Src Port": 443,
            "Dst Port": 80,
            "Protocol": 6,
            ... (84 features total)
        }
    }
    ```
    """
    import time
    start_time = time.time()

    try:
        # Preprocess features
        processed = preprocess_features(flow.features)

        # --- Ensemble prediction: soft voting (RF + XGBoost) ---
        rf_probs_2d, xgb_probs_2d, ens_2d = _ensemble_proba(state.rf_model, state.xgb_model, processed)
        rf_probs = rf_probs_2d[0]
        xgb_probs = xgb_probs_2d[0]
        ensemble_probs = ens_2d[0]

        # Ensemble prediction from averaged probabilities
        prediction_idx = int(np.argmax(ensemble_probs))
        prediction_label = state.label_encoder.inverse_transform([prediction_idx])[0]
        confidence = float(np.max(ensemble_probs))

        # Check if both models agree on the class
        rf_pred = int(np.argmax(rf_probs))
        xgb_pred = int(np.argmax(xgb_probs))
        model_agreement = (rf_pred == xgb_pred)

        # Create probability dict
        prob_dict = {
            label: float(prob)
            for label, prob in zip(state.label_encoder.classes_, ensemble_probs)
        }

        # Get SHAP explanation (from XGBoost — the stronger model)
        top_features = get_shap_explanation(processed, prediction_idx)

        # Determine if attack
        is_attack = prediction_label != "BENIGN"

        # Update metrics
        state.total_predictions += 1
        if is_attack:
            state.total_attacks += 1
            state.attack_distribution[prediction_label] += 1

        elapsed_ms = (time.time() - start_time) * 1000
        elapsed_sec = elapsed_ms / 1000
        state.prediction_times.append(elapsed_ms)

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            PREDICTIONS_COUNTER.labels(
                prediction_class=prediction_label,
                is_attack=str(is_attack).lower()
            ).inc()
            PREDICTION_LATENCY.observe(elapsed_sec)
            if is_attack:
                ACTIVE_ATTACKS.labels(attack_type=prediction_label).inc()

        # Log prediction
        log_prediction(prediction_label, confidence, is_attack, flow_id=flow.flow_id)

        return PredictionResponse(
            prediction=prediction_label,
            prediction_id=int(prediction_idx),
            confidence=confidence,
            probabilities=prob_dict,
            top_features=top_features,
            is_attack=is_attack,
            model_agreement=model_agreement,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
@limiter.limit("30/minute")
async def batch_predict(
    request: Request,
    batch_request: BatchPredictionRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """
    Predict classes for multiple network flows efficiently.

    Accepts up to 1000 flows at once. Each flow should have 84 features
    as a list in the correct order (see /features for feature order).

    **Authentication**: Required if NIDS_REQUIRE_AUTH=true (X-API-Key header)
    **Rate Limit**: 30 requests per minute per IP

    Returns predictions for all flows with attack summary.
    """
    import time
    start_time = time.time()

    try:
        # Extract feature lists
        features_list = [flow.features for flow in batch_request.flows]

        # Preprocess all features
        processed = preprocess_features_batch(features_list)

        # --- Ensemble batch prediction: soft voting (RF + XGBoost) ---
        rf_probs, xgb_probs, ensemble_probs = _ensemble_proba(state.rf_model, state.xgb_model, processed)

        # Ensemble predictions from averaged probabilities
        prediction_indices = np.argmax(ensemble_probs, axis=1)
        prediction_labels = state.label_encoder.inverse_transform(prediction_indices)

        # Check per-flow model agreement
        rf_preds = np.argmax(rf_probs, axis=1)
        xgb_preds = np.argmax(xgb_probs, axis=1)

        # Build response
        predictions = []
        attack_count = 0

        for i, (pred_label, pred_idx, probs) in enumerate(
            zip(prediction_labels, prediction_indices, ensemble_probs)
        ):
            is_attack = pred_label != "BENIGN"
            if is_attack:
                attack_count += 1
                state.total_attacks += 1
                state.attack_distribution[pred_label] += 1

            predictions.append({
                "flow_index": i,
                "prediction": pred_label,
                "prediction_id": int(pred_idx),
                "confidence": float(np.max(probs)),
                "is_attack": is_attack,
                "model_agreement": bool(rf_preds[i] == xgb_preds[i])
            })

        # Update metrics
        state.total_predictions += len(predictions)

        elapsed_ms = (time.time() - start_time) * 1000
        elapsed_sec = elapsed_ms / 1000
        state.prediction_times.append(elapsed_ms / len(predictions))

        # Update Prometheus metrics for batch
        if PROMETHEUS_AVAILABLE:
            PREDICTION_LATENCY.observe(elapsed_sec)
            for pred in predictions:
                PREDICTIONS_COUNTER.labels(
                    prediction_class=pred["prediction"],
                    is_attack=str(pred["is_attack"]).lower()
                ).inc()
                if pred["is_attack"]:
                    ACTIVE_ATTACKS.labels(attack_type=pred["prediction"]).inc()

        logger.info(f"Batch prediction: {len(predictions)} flows, {attack_count} attacks")

        return BatchPredictionResponse(
            predictions=predictions,
            total_flows=len(predictions),
            attack_count=attack_count,
            processing_time_ms=elapsed_ms
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/features", tags=["General"])
async def get_feature_info():
    """
    Get information about expected input features.

    Returns the list of 84 features in the expected order.
    """
    return {
        "total_features": 84,
        "selected_features": 50,
        "feature_names": FEATURE_NAMES_84,
        "selected_feature_names": SELECTED_FEATURES_50,
        "description": "Pass all 84 features to /predict. The API will scale and select the top 50."
    }


@app.get("/classes", tags=["General"])
async def get_attack_classes():
    """
    Get information about attack classes the model can predict.
    """
    if state.label_encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    classes = list(state.label_encoder.classes_)
    return {
        "total_classes": len(classes),
        "classes": [
            {"id": i, "name": name, "is_attack": name != "BENIGN"}
            for i, name in enumerate(classes)
        ]
    }


@app.get("/security", tags=["General"])
async def get_security_status():
    """
    Get current security configuration status.

    Returns information about:
    - Authentication status
    - Rate limiting status
    - Prometheus metrics status
    """
    return {
        "authentication": get_auth_status(),
        "rate_limiting": get_rate_limit_status(),
        "prometheus_metrics": {
            "enabled": PROMETHEUS_AVAILABLE,
            "endpoint": "/metrics" if PROMETHEUS_AVAILABLE else None,
        },
    }


# ============================================================================
# MAIN ENTRY POINT
# Launches uvicorn with optional hot-reload for development.
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # uvicorn can either take an import string or the app object itself.
    # Passing the string causes the module to be imported again which
    # registers Prometheus metrics twice (see duplicate timeseries error).
    #
    # When hot-reload is enabled we still need the string form so the
    # reloader can watch the file, otherwise pass the app object directly
    # to avoid the second import.
    reload_flag = os.getenv("NIDS_DEBUG", "false").lower() == "true"
    if reload_flag:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level=os.getenv("NIDS_LOG_LEVEL", "info").lower(),
        )
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level=os.getenv("NIDS_LOG_LEVEL", "info").lower(),
        )
