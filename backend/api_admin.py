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
    """GET /admins/question-sections/question-set/{question_set_id} . Returns (status_code, body)."""
    url = f"{admin_url.rstrip('/')}/admins/question-sections/question-set/{question_set_id}"
    headers = {}
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("get_question_sections failed: %s", e)
        return 0, {"error": str(e)}


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
