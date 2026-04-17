"""
Verify admin Bearer token via AdminBackend internal API.
Used by /ai/* routes (skills, JD) so only authenticated admins can call AI.
"""
import logging
import os

import httpx
from fastapi import Depends, HTTPException, Request

import config

logger = logging.getLogger(config.APP_NAME)


async def verify_admin_token(request: Request):
    """Dependency: require valid Bearer token (validated by AdminBackend)."""
    auth = request.headers.get("Authorization") or request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization Bearer token required")
    token = auth[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authorization Bearer token required")

    admin_url = getattr(config, "ADMIN_BACKEND_URL", None) or os.getenv("ADMIN_BACKEND_URL", "http://localhost:8002")
    admin_url = admin_url.rstrip("/")
    service_token = getattr(config, "ADMIN_SERVICE_TOKEN", None) or os.getenv("ADMIN_SERVICE_TOKEN", "")
    if not service_token:
        logger.warning("ADMIN_SERVICE_TOKEN not set; cannot verify admin token")
        raise HTTPException(status_code=503, detail="Service misconfiguration: ADMIN_SERVICE_TOKEN not set")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{admin_url}/internal/verify-token",
                json={"token": token},
                headers={"Content-Type": "application/json", "X-Service-Token": service_token},
            )
            if resp.status_code != 200:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                raise HTTPException(status_code=401, detail=data.get("message", "Invalid or expired token"))
            return True
    except httpx.RequestError as e:
        logger.exception("Verify token request failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to verify token with AdminBackend")
