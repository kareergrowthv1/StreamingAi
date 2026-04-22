"""
Verify admin Bearer token via AdminBackend internal API.
Used by /ai/* routes (skills, JD) so only authenticated admins can call AI.
"""
import logging
import os
import time
import hashlib
from urllib.parse import urlparse

import httpx
from fastapi import Depends, HTTPException, Request

import config

logger = logging.getLogger(config.APP_NAME)

_TOKEN_CACHE = {}
_TOKEN_CACHE_TTL_SECONDS = 120


async def verify_admin_token(request: Request):
    """Dependency: require valid Bearer token (validated by AdminBackend)."""
    auth = request.headers.get("Authorization") or request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization Bearer token required")
    token = auth[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authorization Bearer token required")

    token_key = hashlib.sha256(token.encode("utf-8")).hexdigest()
    cached_at = _TOKEN_CACHE.get(token_key)
    now = time.time()
    if cached_at and (now - cached_at) < _TOKEN_CACHE_TTL_SECONDS:
        return True

    configured_admin_url = getattr(config, "ADMIN_BACKEND_URL", None) or os.getenv("ADMIN_BACKEND_URL", "http://localhost:8002")
    configured_admin_url = configured_admin_url.rstrip("/")
    service_token = getattr(config, "ADMIN_SERVICE_TOKEN", None) or os.getenv("ADMIN_SERVICE_TOKEN", "")
    if not service_token:
        logger.warning("ADMIN_SERVICE_TOKEN not set; cannot verify admin token")
        raise HTTPException(status_code=503, detail="Service misconfiguration: ADMIN_SERVICE_TOKEN not set")

    # Primary configured URL + safe local fallbacks for local/dev runs.
    admin_urls = [configured_admin_url]
    try:
        parsed = urlparse(configured_admin_url)
        scheme = parsed.scheme or "https"
        port = parsed.port or (443 if scheme == "https" else 80)
        for host in ["localhost", "127.0.0.1"]:
            candidate = f"{scheme}://{host}:{port}"
            if candidate not in admin_urls:
                admin_urls.append(candidate)
    except Exception:
        pass

    last_request_error = None
    for admin_url in admin_urls:
        try:
            async with httpx.AsyncClient(timeout=6) as client:
                resp = await client.post(
                    f"{admin_url}/internal/verify-token",
                    json={"token": token},
                    headers={"Content-Type": "application/json", "X-Service-Token": service_token},
                )
                if resp.status_code == 200:
                    _TOKEN_CACHE[token_key] = now
                    return True
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                raise HTTPException(status_code=401, detail=data.get("message", "Invalid or expired token"))
        except httpx.RequestError as e:
            last_request_error = e
            continue

    logger.exception("Verify token request failed: %s", last_request_error)
    raise HTTPException(status_code=502, detail="Failed to verify token with AdminBackend")
