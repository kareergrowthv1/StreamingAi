"""
Fetch AI config from Superadmin backend (dynamic API). No hardcoded API keys.
GET {SUPERADMIN_BACKEND_URL}/superadmin/settings/ai-config.
Falls back to env vars if fetch fails.
"""
import os
import logging
import time

import config

logger = logging.getLogger(config.APP_NAME)

_CACHE = {}        # keyed by tenant_id (or "" for global)
_CACHE_TS = {}     # keyed by tenant_id
_CACHE_TTL = 60    # seconds


def _default_config():
    """Fallback when Superadmin backend fetch fails. Prefer Superadmin Settings > AI Config (DB)."""
    return {
        "provider": "OPENAI",
        "apiKey": os.getenv("OPENAI_API_KEY", ""),
        "baseUrl": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        "maxTokens": int(os.getenv("OPENAI_MAX_TOKENS", "1024")),
        "topP": float(os.getenv("OPENAI_TOP_P", "1.0")),
        "frequencyPenalty": float(os.getenv("OPENAI_FREQUENCY_PENALTY", "0")),
        "presencePenalty": float(os.getenv("OPENAI_PRESENCE_PENALTY", "0")),
        "stream": os.getenv("OPENAI_STREAM", "true").lower() == "true",
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "300")),
        "chunkSize": int(os.getenv("OPENAI_CHUNK_SIZE", "1024")),
        "retryOnTimeout": os.getenv("OPENAI_RETRY_ON_TIMEOUT", "true").lower() == "true",
        "maxRetries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
    }


async def get_ai_config(tenant_id: str = "") -> dict:
    """
    Fetch AI config from Superadmin backend. Caches per tenant for _CACHE_TTL seconds.
    Falls back to env vars if fetch fails.
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
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") and data.get("data"):
                    cfg = data["data"]
                    result = {
                        "provider": cfg.get("provider", "OPENAI"),
                        "apiKey": cfg.get("apiKey", ""),
                        "baseUrl": cfg.get("baseUrl", "https://api.openai.com/v1"),
                        "model": cfg.get("model", "gpt-3.5-turbo"),
                        "temperature": float(cfg.get("temperature", 0.7)),
                        "maxTokens": int(cfg.get("maxTokens", 1024)),
                        "topP": float(cfg.get("topP", 1.0)),
                        "frequencyPenalty": float(cfg.get("frequencyPenalty", 0)),
                        "presencePenalty": float(cfg.get("presencePenalty", 0)),
                        "stream": cfg.get("stream", True),
                        "timeout": int(cfg.get("timeout", 300)),
                        "chunkSize": int(cfg.get("chunkSize", 1024)),
                        "retryOnTimeout": cfg.get("retryOnTimeout", True),
                        "maxRetries": int(cfg.get("maxRetries", 3)),
                    }
                    _CACHE[cache_key] = result
                    _CACHE_TS[cache_key] = now
                    return result
            else:
                logger.warning("get_ai_config: Superadmin returned %s for tenant=%s", resp.status_code, tenant_id or "(global)")
    except Exception as e:
        logger.warning(
            "Failed to fetch AI config from Superadmin (dynamic API), using env fallback: %s. "
            "Set SUPERADMIN_BACKEND_URL and SUPERADMIN_SERVICE_TOKEN if using DB config.",
            e,
        )

    return _default_config()


def get_ai_config_sync(tenant_id: str = "") -> dict:
    """Sync version for use in sync contexts."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(get_ai_config(tenant_id))
