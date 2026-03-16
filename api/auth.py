"""
API Authentication Module
=========================
Provides API key authentication for the NIDS API.

Features:
- X-API-Key header validation
- Environment variable configuration
- Optional authentication (can be disabled for development)
- Multiple API keys support

Configuration:
    Set these environment variables:
    - NIDS_API_KEY: Primary API key (required if auth enabled)
    - NIDS_API_KEYS: Comma-separated list of valid keys (optional)
    - NIDS_REQUIRE_AUTH: Set to 'true' to enable authentication (default: false)

Usage:
    from auth import get_api_key

    # In endpoint that requires authentication
    @app.get("/protected")
    async def protected_endpoint(api_key: str = Depends(get_api_key)):
        return {"message": "Authenticated!"}
"""

import os
import secrets
import logging
from typing import Optional, Set

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("nids_api.auth")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# API Key header name
API_KEY_HEADER = "X-API-Key"

# Whether authentication is required
REQUIRE_AUTH = os.getenv("NIDS_REQUIRE_AUTH", "false").lower() == "true"

# Load API keys from environment
def _load_api_keys() -> Set[str]:
    """Load valid API keys from environment variables."""
    keys = set()

    # Primary key
    primary_key = os.getenv("NIDS_API_KEY")
    if primary_key:
        keys.add(primary_key)

    # Additional keys (comma-separated)
    additional_keys = os.getenv("NIDS_API_KEYS", "")
    if additional_keys:
        for key in additional_keys.split(","):
            key = key.strip()
            if key:
                keys.add(key)

    return keys


# Valid API keys
VALID_API_KEYS: Set[str] = _load_api_keys()

# ==============================================================================
# API KEY SECURITY
# ==============================================================================

# API Key header dependency
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Validate API key from request header.

    If NIDS_REQUIRE_AUTH is false, returns None (no authentication required).
    If NIDS_REQUIRE_AUTH is true, validates the API key and raises 401/403 on failure.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        The validated API key, or None if auth not required

    Raises:
        HTTPException: 401 if key missing (when required), 403 if key invalid
    """
    # If auth not required, allow all requests
    if not REQUIRE_AUTH:
        return None

    # Auth required but no key provided
    if not api_key:
        logger.warning("Authentication failed: No API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Set X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate key using constant-time comparison (prevents timing attacks)
    key_valid = False
    for valid_key in VALID_API_KEYS:
        if secrets.compare_digest(api_key, valid_key):
            key_valid = True
            break

    if not key_valid:
        logger.warning(f"Authentication failed: Invalid API key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    logger.debug("API key validated successfully")
    return api_key


def is_auth_enabled() -> bool:
    """Check if authentication is currently enabled."""
    return REQUIRE_AUTH


def get_auth_status() -> dict:
    """
    Get current authentication status.

    Returns:
        Dictionary with auth configuration details
    """
    return {
        "auth_enabled": REQUIRE_AUTH,
        "keys_configured": len(VALID_API_KEYS),
        "header_name": API_KEY_HEADER,
    }


# ==============================================================================
# INITIALIZATION
# ==============================================================================

# Log authentication status on module load
if REQUIRE_AUTH:
    if VALID_API_KEYS:
        logger.info(f"API authentication enabled with {len(VALID_API_KEYS)} key(s)")
    else:
        logger.warning("API authentication enabled but no keys configured!")
        logger.warning("Set NIDS_API_KEY environment variable")
else:
    logger.info("API authentication disabled (set NIDS_REQUIRE_AUTH=true to enable)")
