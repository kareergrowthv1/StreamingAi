"""
CandidateBackend API calls. Used by main.py WebSocket handler only.
All HTTP calls to CandidateBackend (candidate-interviews, question-answers).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0


def _extract_data(body: Any) -> Any:
    """Normalize API envelope `{ success, data }` into raw payload when available."""
    if isinstance(body, dict) and "data" in body and isinstance(body.get("data"), (dict, list)):
        return body.get("data")
    return body


async def get_candidate_interview(
    cand_url: str,
    candidate_id: str,
    position_id: str,
    question_set_id: str,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """GET interview summary, supporting current and legacy routes.

    Primary: /candidate/assessment-summary
    Legacy fallback: /candidate-interviews/candidate/{candidateId}/position/{positionId}
    """
    url = f"{cand_url.rstrip('/')}/candidate/assessment-summary"
    params = {
        "candidateId": candidate_id,
        "positionId": position_id,
        "questionSetId": question_set_id
    }
    headers = {}
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    last_status = 0
    last_body: Any = {}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        if r.status_code == 200:
            return 200, _extract_data(body)
        last_status, last_body = r.status_code, body
    except Exception as e:
        last_status, last_body = 0, {"error": str(e)}

    # Legacy fallback for older CandidateBackend builds.
    legacy_url = f"{cand_url.rstrip('/')}/candidate-interviews/candidate/{candidate_id}/position/{position_id}"
    legacy_params = {"questionSetId": question_set_id}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(legacy_url, params=legacy_params, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        if r.status_code == 200:
            return 200, _extract_data(body)
        return r.status_code, body
    except Exception as e:
        logger.warning("get_candidate_interview failed (primary=%s, fallback error=%s)", last_status, e)
        return last_status, (last_body or {"error": str(e)})


async def post_candidate_interview(
    cand_url: str,
    payload: dict,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """POST interview summary, supporting current and legacy routes."""
    url = f"{cand_url.rstrip('/')}/candidate/assessment-summary"
    headers = {}
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    last_status = 0
    last_body: Any = {}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        if 200 <= r.status_code < 300:
            return r.status_code, _extract_data(body)
        last_status, last_body = r.status_code, body
    except Exception as e:
        last_status, last_body = 0, {"error": str(e)}

    legacy_url = f"{cand_url.rstrip('/')}/candidate-interviews"
    legacy_payload = dict(payload)
    # Legacy API generally expects questionSetId (not questionId); keep both when present.
    if "questionId" in legacy_payload and "questionSetId" not in legacy_payload:
        legacy_payload["questionSetId"] = legacy_payload.get("questionId")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(legacy_url, json=legacy_payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        if 200 <= r.status_code < 300:
            return r.status_code, _extract_data(body)
        return r.status_code, body
    except Exception as e:
        logger.warning("post_candidate_interview failed (primary=%s, fallback error=%s)", last_status, e)
        return last_status, (last_body or {"error": str(e)})


async def put_candidate_interview(
    cand_url: str,
    screening_id: str,
    payload: dict,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """PATCH interview summary, with legacy fallback for candidate-interviews routes."""
    # Note: screening_id is no longer used in current route URL (uses candidateId/positionId in payload)
    url = f"{cand_url.rstrip('/')}/candidate/assessment-summary"
    headers = {}
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    last_status = 0
    last_body: Any = {}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.patch(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        if 200 <= r.status_code < 300:
            return r.status_code, _extract_data(body)
        last_status, last_body = r.status_code, body
    except Exception as e:
        last_status, last_body = 0, {"error": str(e)}

    # Legacy fallback expects resource id in URL; try PATCH first, then PUT.
    if not screening_id:
        return last_status, last_body
    legacy_url = f"{cand_url.rstrip('/')}/candidate-interviews/{screening_id}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.patch(legacy_url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        if 200 <= r.status_code < 300:
            return r.status_code, _extract_data(body)
        # Some legacy stacks only expose PUT.
        async with httpx.AsyncClient() as client:
            r2 = await client.put(legacy_url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body2 = r2.json()
        except Exception:
            body2 = {"raw": r2.text}
        if 200 <= r2.status_code < 300:
            return r2.status_code, _extract_data(body2)
        return r2.status_code, body2
    except Exception as e:
        logger.warning("put_candidate_interview failed (primary=%s, fallback error=%s)", last_status, e)
        return last_status, (last_body or {"error": str(e)})


async def get_question_answers(
    cand_url: str,
    params: dict,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """GET /public/question-answers?clientId=&candidateId=&positionId=&round= . Returns (status_code, body)."""
    url = f"{cand_url.rstrip('/')}/public/question-answers"
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
        logger.warning("get_question_answers failed: %s", e)
        return 0, {"error": str(e)}


async def post_question_answer(
    cand_url: str,
    payload: dict,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """POST /public/question-answers . Returns (status_code, body)."""
    url = f"{cand_url.rstrip('/')}/public/question-answers"
    headers = {}
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("post_question_answer failed: %s", e)
        return 0, {"error": str(e)}


# ─── MongoDB Interview Responses (Round 1 & 2 answers) ───────────────────────
# Endpoints: /candidate/interview-responses  (CandidateBackend → MongoDB)

async def get_interview_responses(
    cand_url: str,
    candidate_id: str,
    position_id: str,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """GET /candidate/interview-responses?candidateId=&positionId= — returns full doc with saved answers."""
    url = f"{cand_url.rstrip('/')}/candidate/interview-responses"
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
        if r.status_code == 200:
            return 200, _extract_data(body)
        return r.status_code, body
    except Exception as e:
        logger.warning("get_interview_responses failed: %s", e)
        return 0, {"error": str(e)}


async def post_interview_responses(
    cand_url: str,
    payload: dict,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """POST /candidate/interview-responses — create doc (idempotent: returns existing if already exists)."""
    url = f"{cand_url.rstrip('/')}/candidate/interview-responses"
    headers = {}
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        if 200 <= r.status_code < 300:
            return r.status_code, _extract_data(body)
        return r.status_code, body
    except Exception as e:
        logger.warning("post_interview_responses failed: %s", e)
        return 0, {"error": str(e)}


async def patch_interview_responses(
    cand_url: str,
    payload: dict,
    tenant_id: Optional[str] = None,
) -> tuple[int, Any]:
    """PATCH /candidate/interview-responses — targeted answer update or cross-question append."""
    url = f"{cand_url.rstrip('/')}/candidate/interview-responses"
    headers = {}
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    try:
        async with httpx.AsyncClient() as client:
            r = await client.patch(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        if 200 <= r.status_code < 300:
            return r.status_code, _extract_data(body)
        return r.status_code, body
    except Exception as e:
        logger.warning("patch_interview_responses failed: %s", e)
        return 0, {"error": str(e)}
        return 0, {"error": str(e)}
