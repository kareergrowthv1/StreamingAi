"""
Fetch AI config from Superadmin backend (dynamic API).
GET {SUPERADMIN_BACKEND_URL}/superadmin/settings/ai-config.
No static/env fallback for provider settings.
"""
import os
import logging
import time

import config

logger = logging.getLogger(config.APP_NAME)

_CACHE = {}        # keyed by tenant_id (or "" for global)
_CACHE_TS = {}     # keyed by tenant_id
_CACHE_TTL = 60    # seconds


def _normalize_dynamic_config(cfg: dict) -> dict:
    provider = str(cfg.get("provider") or "").strip().upper()
    api_key = str(cfg.get("apiKey") or "").strip()
    base_url = str(cfg.get("baseUrl") or "").strip()
    model = str(cfg.get("model") or "").strip()

    if not provider or not api_key or not base_url or not model:
        raise RuntimeError("Dynamic AI config incomplete: provider/apiKey/baseUrl/model are required")

    return {
        "provider": provider,
        "apiKey": api_key,
        "baseUrl": base_url,
        "model": model,
        "temperature": float(cfg.get("temperature", 0.7)),
        "maxTokens": int(cfg.get("maxTokens", 1024)),
        "topP": float(cfg.get("topP", 1.0)),
        "frequencyPenalty": float(cfg.get("frequencyPenalty", 0)),
        "presencePenalty": float(cfg.get("presencePenalty", 0)),
        "stream": bool(cfg.get("stream", True)),
        "timeout": int(cfg.get("timeout", 60)),
        "chunkSize": int(cfg.get("chunkSize", 1024)),
        "retryOnTimeout": bool(cfg.get("retryOnTimeout", True)),
        "maxRetries": int(cfg.get("maxRetries", 1)),
    }


async def get_ai_config(tenant_id: str = "") -> dict:
    """
    Fetch AI config from Superadmin backend. Caches per tenant for _CACHE_TTL seconds.
    No static/env fallback for provider settings.
    tenant_id is passed as X-Tenant-Id header so the backend can return tenant-specific config.
    """
    global _CACHE, _CACHE_TS

    cache_key = tenant_id or ""
    now = time.time()
    if cache_key in _CACHE and (now - _CACHE_TS.get(cache_key, 0)) < _CACHE_TTL:
        return _CACHE[cache_key]

    base_url = getattr(config, "SUPERADMIN_BACKEND_URL", None) or os.getenv("SUPERADMIN_BACKEND_URL", "http://localhost:8001")
    base_url = base_url.rstrip("/")
    url = f"{base_url}/superadmin/settings/ai-config"

    service_token = (
        getattr(config, "SUPERADMIN_SERVICE_TOKEN", None)
        or os.getenv("SUPERADMIN_SERVICE_TOKEN")
        or os.getenv("INTERNAL_SERVICE_TOKEN")
    )
    headers = {}
    if service_token:
        headers["X-Service-Token"] = service_token
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id

    try:
        import httpx
        async with httpx.AsyncClient(timeout=4) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") and data.get("data"):
                    cfg = data["data"]
                    result = _normalize_dynamic_config(cfg)
                    _CACHE[cache_key] = result
                    _CACHE_TS[cache_key] = now
                    return result
            else:
                logger.warning("get_ai_config: Superadmin returned %s for tenant=%s", resp.status_code, tenant_id or "(global)")
    except Exception as e:
        logger.warning(
            "Failed to fetch AI config from Superadmin (dynamic API): %s. "
            "Set SUPERADMIN_BACKEND_URL and SUPERADMIN_SERVICE_TOKEN correctly.",
            e,
        )

    # If fetch failed but we have a previously cached config, keep using it to avoid outages/timeouts.
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    raise RuntimeError("Dynamic AI config unavailable and no cached configuration present")


def get_ai_config_sync(tenant_id: str = "") -> dict:
    """Sync version for use in sync contexts."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(get_ai_config(tenant_id))
