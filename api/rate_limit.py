"""
Rate Limiting Module
====================
Provides rate limiting for the NIDS API using slowapi.

Features:
- Per-IP rate limiting
- Configurable limits via environment variables
- Separate limits for different endpoint types
- Custom error responses

Configuration:
    Set these environment variables:
    - NIDS_RATE_LIMIT: Requests per minute (default: 100)
    - NIDS_RATE_LIMIT_BURST: Burst limit (default: 10)
    - NIDS_RATE_LIMIT_ENABLED: Set to 'false' to disable (default: true)

Usage:
    from rate_limit import limiter, rate_limit_exceeded_handler

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    @app.get("/endpoint")
    @limiter.limit("10/minute")
    async def limited_endpoint(request: Request):
        return {"message": "Rate limited endpoint"}
"""

import os
import logging
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    Limiter = None
    RateLimitExceeded = Exception

logger = logging.getLogger("nids_api.rate_limit")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Rate limiting enabled?
RATE_LIMIT_ENABLED = os.getenv("NIDS_RATE_LIMIT_ENABLED", "true").lower() == "true"

# Default rate limits
DEFAULT_RATE_LIMIT = os.getenv("NIDS_RATE_LIMIT", "100/minute")
PREDICT_RATE_LIMIT = os.getenv("NIDS_PREDICT_RATE_LIMIT", "60/minute")
BATCH_RATE_LIMIT = os.getenv("NIDS_BATCH_RATE_LIMIT", "30/minute")


# ==============================================================================
# KEY FUNCTION
# ==============================================================================

def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.

    Handles X-Forwarded-For header for requests behind proxies/load balancers.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address string
    """
    # Check for forwarded header (common with reverse proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(",")[0].strip()

    # Check for real IP header (nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return "unknown"


# ==============================================================================
# LIMITER SETUP
# ==============================================================================

if SLOWAPI_AVAILABLE and RATE_LIMIT_ENABLED:
    # Create limiter with IP-based key function
    limiter = Limiter(
        key_func=get_client_ip,
        default_limits=[DEFAULT_RATE_LIMIT],
        storage_uri="memory://",  # In-memory storage (use Redis for production clusters)
        strategy="fixed-window",  # Reset at fixed intervals
    )
    logger.info(f"Rate limiting enabled: {DEFAULT_RATE_LIMIT}")
else:
    # Create a no-op limiter when disabled or slowapi not available
    class NoOpLimiter:
        """Dummy limiter that does nothing."""

        def limit(self, limit_value: str) -> Callable:
            """Return a no-op decorator."""
            def decorator(func: Callable) -> Callable:
                return func
            return decorator

        def shared_limit(self, limit_value: str, scope: str) -> Callable:
            """Return a no-op decorator."""
            def decorator(func: Callable) -> Callable:
                return func
            return decorator

    limiter = NoOpLimiter()

    if not SLOWAPI_AVAILABLE:
        logger.warning("slowapi not installed - rate limiting disabled")
        logger.warning("Install with: pip install slowapi")
    else:
        logger.info("Rate limiting disabled (set NIDS_RATE_LIMIT_ENABLED=true to enable)")


# ==============================================================================
# ERROR HANDLER
# ==============================================================================

async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors.

    Returns a JSON response with details about the rate limit.

    Args:
        request: FastAPI request object
        exc: Rate limit exceeded exception

    Returns:
        JSON response with 429 status code
    """
    client_ip = get_client_ip(request)
    logger.warning(f"Rate limit exceeded for {client_ip}: {request.url.path}")

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please slow down.",
            "detail": str(exc.detail) if hasattr(exc, 'detail') else "Rate limit exceeded",
            "retry_after": getattr(exc, 'retry_after', 60),
        },
        headers={
            "Retry-After": str(getattr(exc, 'retry_after', 60)),
            "X-RateLimit-Limit": DEFAULT_RATE_LIMIT,
        }
    )


# ==============================================================================
# RATE LIMIT DECORATORS
# ==============================================================================

def limit_predict():
    """Rate limit decorator for prediction endpoints."""
    return limiter.limit(PREDICT_RATE_LIMIT)


def limit_batch():
    """Rate limit decorator for batch prediction endpoints."""
    return limiter.limit(BATCH_RATE_LIMIT)


def limit_general():
    """Rate limit decorator for general endpoints."""
    return limiter.limit(DEFAULT_RATE_LIMIT)


# ==============================================================================
# STATUS FUNCTIONS
# ==============================================================================

def is_rate_limit_enabled() -> bool:
    """Check if rate limiting is currently enabled."""
    return RATE_LIMIT_ENABLED and SLOWAPI_AVAILABLE


def get_rate_limit_status() -> dict:
    """
    Get current rate limiting status.

    Returns:
        Dictionary with rate limit configuration details
    """
    return {
        "enabled": is_rate_limit_enabled(),
        "slowapi_installed": SLOWAPI_AVAILABLE,
        "default_limit": DEFAULT_RATE_LIMIT,
        "predict_limit": PREDICT_RATE_LIMIT,
        "batch_limit": BATCH_RATE_LIMIT,
    }
