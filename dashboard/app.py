"""
NIDS Real-Time Dashboard
========================
A Flask-based web dashboard for monitoring the NIDS system.

Features:
- Real-time attack count display
- Last 100 predictions table
- Attack distribution pie chart
- System health status
- Auto-refresh every 5 seconds

Usage:
    cd dashboard
    pip install -r requirements.txt
    python app.py

    Then open http://localhost:5001 in your browser.
"""

import os
import sys
import logging
import secrets
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, render_template, jsonify, request, Response
import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# API Configuration
API_URL = os.getenv("NIDS_API_URL", "http://localhost:8000")
API_KEY = os.getenv("NIDS_API_KEY", "")

# Dashboard Configuration
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5001"))
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "5"))  # seconds

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nids.dashboard")

# ==============================================================================
# FLASK APP
# ==============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# ==============================================================================
# API CLIENT
# ==============================================================================

def get_api_headers() -> Dict[str, str]:
    """Get headers for API requests."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def fetch_api(endpoint: str) -> Optional[Dict[str, Any]]:
    """
    Fetch data from the NIDS API.

    Args:
        endpoint: API endpoint (e.g., "/health", "/metrics")

    Returns:
        JSON response data or None on error
    """
    try:
        url = f"{API_URL}{endpoint}"
        response = requests.get(url, headers=get_api_headers(), timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        logger.warning(f"Cannot connect to API at {API_URL}")
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"API request timed out: {endpoint}")
        return None
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return None


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template(
        "index.html",
        api_url=API_URL,
        refresh_interval=REFRESH_INTERVAL
    )


@app.route("/api/status")
def api_status():
    """
    Get combined status from the NIDS API.

    Returns:
        JSON with health, metrics, and security status
    """
    health = fetch_api("/health")
    metrics = fetch_api("/stats")
    security = fetch_api("/security")
    classes = fetch_api("/classes")

    # Calculate derived metrics
    attack_rate = 0
    if metrics and metrics.get("total_predictions", 0) > 0:
        attack_rate = metrics["total_attacks"] / metrics["total_predictions"] * 100

    return jsonify({
        "connected": health is not None,
        "timestamp": datetime.now().isoformat(),
        "health": health or {"status": "disconnected", "models_loaded": False},
        "metrics": metrics or {
            "total_predictions": 0,
            "total_attacks": 0,
            "attack_rate": 0,
            "attack_distribution": {},
            "avg_prediction_time_ms": 0,
            "uptime_seconds": 0
        },
        "security": security or {"authentication": {}, "rate_limiting": {}},
        "attack_classes": classes.get("classes", []) if classes else [],
        "attack_rate_percent": round(attack_rate, 2)
    })


@app.route("/api/attack_distribution")
def attack_distribution():
    """
    Get attack distribution data for charts.

    Returns:
        JSON with labels and values for pie chart
    """
    metrics = fetch_api("/stats")

    if not metrics or not metrics.get("attack_distribution"):
        return jsonify({
            "labels": ["No Data"],
            "values": [1],
            "colors": ["#cccccc"]
        })

    distribution = metrics["attack_distribution"]

    # Define colors for attack types
    attack_colors = {
        "BENIGN": "#28a745",  # Green
        "DoS Hulk": "#dc3545",  # Red
        "DoS GoldenEye": "#e83e8c",  # Pink
        "DoS Slowloris": "#fd7e14",  # Orange
        "DoS slowhttptest": "#ffc107",  # Yellow
        "DDoS": "#dc3545",  # Red
        "PortScan": "#6f42c1",  # Purple
        "FTP-Patator": "#20c997",  # Teal
        "SSH-Patator": "#17a2b8",  # Cyan
        "Bot": "#6c757d",  # Gray
        "Web Attack - Brute Force": "#007bff",  # Blue
        "Web Attack - XSS": "#e83e8c",  # Pink
        "Web Attack - SQL Injection": "#fd7e14",  # Orange
        "Infiltration": "#343a40",  # Dark gray
        "Heartbleed": "#dc3545",  # Red
    }

    labels = list(distribution.keys())
    values = list(distribution.values())
    colors = [attack_colors.get(label, "#6c757d") for label in labels]

    return jsonify({
        "labels": labels,
        "values": values,
        "colors": colors,
        "total": sum(values)
    })


@app.route("/api/recent_attacks")
def recent_attacks():
    """
    Get recent attack detections.

    Note: This would require the API to store recent predictions.
    For now, returns summary from metrics.
    """
    metrics = fetch_api("/stats")

    if not metrics:
        return jsonify({"attacks": [], "total": 0})

    # Convert attack distribution to list
    attacks = []
    for attack_type, count in metrics.get("attack_distribution", {}).items():
        attacks.append({
            "type": attack_type,
            "count": count,
            "percentage": round(count / max(metrics.get("total_attacks", 1), 1) * 100, 1)
        })

    # Sort by count descending
    attacks.sort(key=lambda x: x["count"], reverse=True)

    return jsonify({
        "attacks": attacks,
        "total": metrics.get("total_attacks", 0)
    })


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


@app.route("/api/stream")
def stream_proxy():
    """Proxy the SSE stream from the API to the browser."""
    def generate():
        try:
            with requests.get(
                f"{API_URL}/api/stream",
                headers=get_api_headers(),
                stream=True,
                timeout=(5, None),
            ) as upstream:
                for line in upstream.iter_lines(decode_unicode=True):
                    if line is not None:
                        yield f"{line}\n"
                    else:
                        yield "\n"
        except requests.exceptions.ConnectionError:
            yield "data: {\"error\": \"API disconnected\"}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/health")
def health():
    """Dashboard health check endpoint."""
    api_health = fetch_api("/health")
    return jsonify({
        "dashboard": "healthy",
        "api_connected": api_health is not None,
        "api_url": API_URL,
        "timestamp": datetime.now().isoformat()
    })


# ==============================================================================
# ERROR HANDLERS
# ==============================================================================

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NIDS Real-Time Dashboard")
    print("=" * 60)
    print(f"Dashboard URL: http://localhost:{DASHBOARD_PORT}")
    print(f"API URL: {API_URL}")
    print(f"Refresh interval: {REFRESH_INTERVAL} seconds")
    print("=" * 60)

    app.run(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )
