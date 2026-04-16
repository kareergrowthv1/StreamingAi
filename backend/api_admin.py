"""
AdminBackend API calls. Used by main.py WebSocket handler only.
All HTTP calls to AdminBackend (assessment-summaries, question-sections, round-timing, private-links).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

try:
    import config as _config  # type: ignore
except ImportError:
    _config = None  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0


def _internal_headers() -> dict:
    """Return headers with X-Service-Token for calling /internal endpoints."""
    token = (
        getattr(_config, "INTERNAL_SERVICE_TOKEN", "")
        or getattr(_config, "ADMIN_SERVICE_TOKEN", "")
        if _config else ""
    )
    h = {}
    if token:
        h["X-Service-Token"] = token
    return h


def _auth_headers() -> dict:
    """Optional bearer auth header for admin routes that require JWT/session auth."""
    bearer = (
        getattr(_config, "ADMIN_BEARER_TOKEN", "")
        or getattr(_config, "ADMIN_AUTH_TOKEN", "")
        if _config else ""
    )
    return {"Authorization": f"Bearer {bearer}"} if bearer else {}


def _candidate_admin_bases(admin_url: str) -> list[str]:
    """Return base URL candidates (normalized) for different route mount styles."""
    base = admin_url.rstrip("/")
    bases = [base]
    if base.endswith("/admins"):
        bases.append(base[:-7])
    if base.endswith("/api"):
        bases.append(base[:-4])
    # Preserve order but dedupe.
    out = []
    seen = set()
    for b in bases:
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return out


async def get_assessment_summaries(
    admin_url: str,
    candidate_id: str,
    position_id: str,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """GET /candidates/assessment-summaries?candidateId=&positionId= . Returns (status_code, body)."""
    url = f"{admin_url.rstrip('/')}/candidates/assessment-summaries"
    params = {"candidateId": candidate_id, "positionId": position_id}
    headers = {}
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("get_assessment_summaries failed: %s", e)
        return 0, {"error": str(e)}


async def put_round_timing(
    admin_url: str,
    tenant_id: Optional[str],
    payload: dict,
) -> tuple[int, Any]:
    """PUT /candidates/assessment-summaries/round-timing . Returns (status_code, body)."""
    url = f"{admin_url.rstrip('/')}/candidates/assessment-summaries/round-timing"
    headers = {}
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    try:
        async with httpx.AsyncClient() as client:
            r = await client.put(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("put_round_timing failed: %s", e)
        return 0, {"error": str(e)}


async def get_question_sections(
    admin_url: str,
    question_set_id: str,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """Fetch question-sections from AdminBackend across route variants.

    Order:
    1) /internal/question-sections/... with X-Service-Token (preferred)
    2) /api/question-sections/... (older stacks)
    3) /admins/question-sections/... (legacy proxy route)
    """
    bases = _candidate_admin_bases(admin_url)
    paths = [
        f"/internal/question-sections/question-set/{question_set_id}",
        f"/api/question-sections/question-set/{question_set_id}",
        f"/admins/question-sections/question-set/{question_set_id}",
    ]
    last_status = 0
    last_body: Any = {"error": "No response"}

    for base in bases:
        for path in paths:
            url = f"{base}{path}"
            headers = _internal_headers() if path.startswith("/internal/") else _auth_headers()
            if tenant_id:
                headers["X-Tenant-Id"] = tenant_id
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
                try:
                    body = r.json()
                except Exception:
                    body = {"raw": r.text}
                if r.status_code == 200:
                    return 200, body
                last_status, last_body = r.status_code, body
                logger.info("get_question_sections fallback: %s -> %s", url, r.status_code)
            except Exception as e:
                last_status, last_body = 0, {"error": str(e)}
                logger.info("get_question_sections fallback exception for %s: %s", url, e)

    logger.warning("get_question_sections failed after all fallbacks: status=%s body=%s", last_status, last_body)
    return last_status, last_body


async def get_cross_question_settings(
    admin_url: str,
    client_id: str,
    tenant_id: Optional[str] = None,
) -> dict:
    """Fetch cross-question count settings from AdminBackend internal API.

    Returns dict with crossQuestionCountGeneral and crossQuestionCountPosition (defaults: 2).
    Always returns a usable dict — never raises.
    """
    url = f"{admin_url.rstrip('/')}/internal/cross-question-settings"
    headers = _internal_headers()
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    params = {"clientId": client_id}
    _default = {"crossQuestionCountGeneral": 2, "crossQuestionCountPosition": 2}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
        body = {}
        try:
            body = r.json()
        except Exception:
            pass
        data = body.get("data") or {}
        return {
            "crossQuestionCountGeneral": max(1, min(4, int(data.get("crossQuestionCountGeneral") or 2))),
            "crossQuestionCountPosition": max(1, min(4, int(data.get("crossQuestionCountPosition") or 2))),
        }
    except Exception as e:
        logger.warning("get_cross_question_settings failed: %s", e)
        return _default


async def put_update_interview_status(
    admin_url: str,
    position_id: str,
    candidate_id: str,
) -> tuple[int, Any]:
    """PUT /private-links/update-interview-status?positionId=&candidateId= . Returns (status_code, body)."""
    url = f"{admin_url.rstrip('/')}/private-links/update-interview-status"
    params = {"positionId": position_id, "candidateId": candidate_id}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.put(url, params=params, json={}, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("put_update_interview_status failed: %s", e)
        return 0, {"error": str(e)}


async def patch_complete_interview(
    admin_url: str,
    tenant_id: Optional[str],
    position_id: str,
    candidate_id: str,
) -> tuple[int, Any]:
    """PATCH /internal/complete-interview — sets interview_completed_at = NOW() in candidate_positions.
    Called by WebSocket test_complete handler when all assigned rounds finish.
    Returns (status_code, body).
    """
    url = f"{admin_url.rstrip('/')}/internal/complete-interview"
    payload = {"positionId": position_id, "candidateId": candidate_id, "tenantId": tenant_id or ""}
    headers = _internal_headers()
    try:
        async with httpx.AsyncClient() as client:
            r = await client.patch(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("patch_complete_interview failed: %s", e)
        return 0, {"error": str(e)}


async def patch_recording_link(
    admin_url: str,
    tenant_id: Optional[str],
    position_id: str,
    candidate_id: str,
    recording_link: str,
    recording_type: str = "screen",
) -> tuple[int, Any]:
    """PATCH /internal/recording-link — persists merged recording URL to tenant tables.
    Returns (status_code, body).
    """
    url = f"{admin_url.rstrip('/')}/internal/recording-link"
    payload = {
        "positionId": position_id,
        "candidateId": candidate_id,
        "tenantId": tenant_id or "",
        "recordingLink": (recording_link or "").strip(),
        "recordingType": (recording_type or "screen").strip().lower(),
    }
    headers = _internal_headers()
    try:
        async with httpx.AsyncClient() as client:
            r = await client.patch(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("patch_recording_link failed: %s", e)
        return 0, {"error": str(e)}
