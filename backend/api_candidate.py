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


async def get_candidate_interview(
    cand_url: str,
    candidate_id: str,
    position_id: str,
    question_set_id: str,
) -> tuple[int, Any]:
    """GET /candidate-interviews/candidate/{candidate_id}/position/{position_id}?questionSetId= . Returns (status_code, body)."""
    url = f"{cand_url.rstrip('/')}/candidate-interviews/candidate/{candidate_id}/position/{position_id}"
    params = {"questionSetId": question_set_id}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("get_candidate_interview failed: %s", e)
        return 0, {"error": str(e)}


async def post_candidate_interview(
    cand_url: str,
    payload: dict,
) -> tuple[int, Any]:
    """POST /candidate-interviews . Returns (status_code, body)."""
    url = f"{cand_url.rstrip('/')}/candidate-interviews"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("post_candidate_interview failed: %s", e)
        return 0, {"error": str(e)}


async def put_candidate_interview(
    cand_url: str,
    screening_id: str,
    payload: dict,
) -> tuple[int, Any]:
    """PUT /candidate-interviews/{screening_id} . Returns (status_code, body)."""
    url = f"{cand_url.rstrip('/')}/candidate-interviews/{screening_id}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.put(url, json=payload, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("put_candidate_interview failed: %s", e)
        return 0, {"error": str(e)}


async def get_question_answers(
    cand_url: str,
    params: dict,
) -> tuple[int, Any]:
    """GET /public/question-answers?clientId=&candidateId=&positionId=&round= . Returns (status_code, body)."""
    url = f"{cand_url.rstrip('/')}/public/question-answers"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params, timeout=DEFAULT_TIMEOUT)
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
) -> tuple[int, Any]:
    """POST /public/question-answers . Returns (status_code, body)."""
    url = f"{cand_url.rstrip('/')}/public/question-answers"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return r.status_code, body
    except Exception as e:
        logger.warning("post_question_answer failed: %s", e)
        return 0, {"error": str(e)}
