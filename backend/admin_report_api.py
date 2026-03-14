"""
Admin Report Payload API — Streaming AI
--------------------------------------
Single API for Admin/Java backend to fetch complete report input data in one call,
and optionally trigger report generation queue.

Endpoint:
  GET /admin-report/payload
    query params:
      - positionId (required)
      - candidateId (required)
      - clientId (required by caller contract; used as fallback tenantId)
      - tenantId (optional)
      - questionSetId (optional; used for round-4 answer fetch)
      - triggerReportGeneration (optional, default=true)
"""

import asyncio
import json
import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin-report", tags=["Admin Report API"])


def _service_headers() -> Dict[str, str]:
    svc_token = getattr(config, "INTERNAL_SERVICE_TOKEN", "") or getattr(config, "ADMIN_SERVICE_TOKEN", "")
    if svc_token:
        return {"X-Service-Token": svc_token, "Content-Type": "application/json"}
    return {"Content-Type": "application/json"}


async def _queue_report_generation(
    position_id: str,
    candidate_id: str,
    client_id: str,
    tenant_id: str,
    question_set_id: str,
) -> Dict[str, Any]:
    """Queue report generation in existing report worker (no duplicate enqueue).

    Checks in order:
      1. MySQL assessment_report_generation.is_generated — if true, skip (already done).
      2. MongoDB report exists and isGenerated — skip.
      3. Worker in-progress / queue set — skip (concurrent request already running).
      4. Otherwise enqueue.
    """
    import report_generator as rg

    await rg._start_worker()

    key = rg._queue_key(position_id, candidate_id)

    # ── 1. MySQL flag check (authoritative "already done" signal) ────────────
    generation_flag = await _get_report_generation_flag(position_id, candidate_id)
    if generation_flag.get("isGenerated"):
        logger.info("[AdminReportAPI] Queue skipped — MySQL is_generated=1 for %s/%s", position_id, candidate_id)
        return {
            "status": "already_generated",
            "message": "Report already generated (MySQL flag).",
            "queued": False,
        }

    # ── 2. MongoDB check ──────────────────────────────────────────────────────
    existing = await rg._get_existing_report(position_id, candidate_id)
    if existing and existing.get("isGenerated"):
        logger.info("[AdminReportAPI] Queue skipped — MongoDB report exists for %s/%s", position_id, candidate_id)
        return {
            "status": "already_generated",
            "message": "Report already generated (MongoDB).",
            "queued": False,
        }

    # ── 3. Already queued or in-progress in this worker ──────────────────────
    if key in rg._in_progress or key in rg._queue_keys:
        return {
            "status": "in_progress",
            "message": "Report generation already in progress.",
            "queued": False,
        }

    # ── 4. Enqueue ────────────────────────────────────────────────────────────
    rg._queue_keys.add(key)
    await rg._report_queue.put(
        {
            "key": key,
            "positionId": position_id,
            "candidateId": candidate_id,
            "clientId": client_id,
            "tenantId": tenant_id,
            "questionSetId": question_set_id or "",
        }
    )

    return {
        "status": "queued",
        "message": "Report generation queued.",
        "queued": True,
    }


def _json_from_response(resp: Any) -> Dict[str, Any]:
    try:
        body = getattr(resp, "body", b"")
        if isinstance(body, (bytes, bytearray)):
            return json.loads(body.decode("utf-8") or "{}")
        if isinstance(body, str):
            return json.loads(body or "{}")
    except Exception:
        pass
    return {}


async def _get_report_generation_flag(position_id: str, candidate_id: str) -> Dict[str, Any]:
    """Read candidates_db.assessment_report_generation via AdminBackend internal API."""
    admin_url = getattr(config, "ADMIN_BACKEND_URL", "http://localhost:8002").rstrip("/")
    headers = _service_headers()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{admin_url}/internal/report-generation-status",
                headers=headers,
                json={"positionId": position_id, "candidateId": candidate_id},
            )
            if resp.status_code == 200:
                body = resp.json() or {}
                data = body.get("data") or {}
                return {
                    "exists": bool(data.get("exists")),
                    "isGenerated": bool(data.get("isGenerated")),
                }
    except Exception as e:
        logger.warning("[AdminReportAPI] report-generation-status check failed: %s", e)
    return {"exists": False, "isGenerated": False}


async def _fetch_base_data(
    client: httpx.AsyncClient,
    admin_url: str,
    headers: Dict[str, str],
    position_id: str,
    candidate_id: str,
    tenant_id: str,
) -> dict:
    try:
        resp = await client.post(
            f"{admin_url}/internal/report-data",
            headers=headers,
            json={"positionId": position_id, "candidateId": candidate_id, "tenantId": tenant_id},
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("[AdminReportAPI] fetch base data failed: %s", e)
    return {}


async def _fetch_assessment_summary(
    client: httpx.AsyncClient,
    admin_url: str,
    headers: Dict[str, str],
    position_id: str,
    candidate_id: str,
    tenant_id: str,
) -> dict:
    try:
        params = {"positionId": position_id, "candidateId": candidate_id}
        if tenant_id:
            params["tenantId"] = tenant_id
        hdrs = {**headers, **({"X-Tenant-Id": tenant_id} if tenant_id else {})}
        resp = await client.get(f"{admin_url}/candidates/assessment-summaries", headers=hdrs, params=params)
        if resp.status_code == 200:
            body = resp.json()
            return body.get("data") or body or {}
    except Exception as e:
        logger.warning("[AdminReportAPI] fetch assessment summary failed: %s", e)
    return {}


async def _fetch_conversational_data(
    client: httpx.AsyncClient,
    cand_url: str,
    candidate_id: str,
    position_id: str,
    question_set_id: str = "",
) -> dict:
    try:
        params = {}
        if question_set_id:
            params["questionSetId"] = question_set_id
        resp = await client.get(
            f"{cand_url}/candidate-interviews/candidate/{candidate_id}/position/{position_id}",
            params=params,
        )
        if resp.status_code == 200:
            return resp.json()
        # If questionSetId narrowing returned nothing, retry without it to catch any saved document
        if resp.status_code == 404 and question_set_id:
            logger.info(
                "[AdminReportAPI] conversational data 404 with questionSetId=%s, retrying without filter",
                question_set_id,
            )
            resp2 = await client.get(
                f"{cand_url}/candidate-interviews/candidate/{candidate_id}/position/{position_id}",
            )
            if resp2.status_code == 200:
                return resp2.json()
            logger.warning(
                "[AdminReportAPI] conversational data (no qset fallback) returned %d: %s",
                resp2.status_code, resp2.text[:200],
            )
        else:
            logger.warning(
                "[AdminReportAPI] conversational data returned %d: %s",
                resp.status_code, resp.text[:200],
            )
    except Exception as e:
        logger.warning("[AdminReportAPI] fetch conversational data failed: %s", e)
    return {}


async def _fetch_coding_data(
    client: httpx.AsyncClient,
    cand_url: str,
    candidate_id: str,
    position_id: str,
) -> dict:
    try:
        resp = await client.get(f"{cand_url}/candidate-coding-responses/candidate/{candidate_id}/position/{position_id}")
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("[AdminReportAPI] fetch coding data failed: %s", e)
    return {}


async def _fetch_aptitude_data(
    client: httpx.AsyncClient,
    cand_url: str,
    candidate_id: str,
    position_id: str,
    question_set_id: str = "",
    client_id: str = "",
) -> list:
    """
    Fetch complete aptitude data by merging:
      1. MongoDB generatedQuestions (question text, options, correctAnswer)
      2. MySQL candidate answers (answerText per questionId)
    """
    logger.info(
        "[AdminReportAPI][Aptitude] START fetch — candidate=%s position=%s questionSetId=%s clientId=%s",
        candidate_id, position_id, question_set_id or "(none)", client_id or "(none)",
    )

    # ── Step 1: Fetch full question objects from MongoDB ─────────────────────
    mongo_questions: list = []
    try:
        if question_set_id:
            r = await client.get(
                f"{cand_url}/candidate-aptitude-responses/generated-questions",
                params={"candidateId": candidate_id, "positionId": position_id, "questionSetId": question_set_id},
            )
            logger.info(
                "[AdminReportAPI][Aptitude] MongoDB /generated-questions status=%d",
                r.status_code,
            )
            if r.status_code == 200:
                mongo_questions = r.json().get("questions") or []
                logger.info("[AdminReportAPI][Aptitude] MongoDB questions (by qset): %d", len(mongo_questions))
            else:
                logger.warning(
                    "[AdminReportAPI][Aptitude] /generated-questions %d: %s",
                    r.status_code, r.text[:300],
                )
        else:
            logger.info("[AdminReportAPI][Aptitude] No questionSetId — skipping /generated-questions, using fallback")

        if not mongo_questions:
            # Fallback: fetch full doc by candidate/position (no questionSetId filter)
            r2 = await client.get(
                f"{cand_url}/candidate-aptitude-responses/candidate/{candidate_id}/position/{position_id}",
            )
            logger.info(
                "[AdminReportAPI][Aptitude] MongoDB fallback /candidate/.../position/... status=%d",
                r2.status_code,
            )
            if r2.status_code == 200:
                mongo_questions = r2.json().get("generatedQuestions") or []
                logger.info("[AdminReportAPI][Aptitude] MongoDB fallback questions: %d", len(mongo_questions))
            else:
                logger.warning(
                    "[AdminReportAPI][Aptitude] fallback %d: %s",
                    r2.status_code, r2.text[:300],
                )
    except Exception as e:
        logger.warning("[AdminReportAPI][Aptitude] MongoDB fetch exception: %s", e)

    # ── Step 2: Fetch candidate answers from MySQL ───────────────────────────
    mysql_answers: Dict[str, Dict] = {}
    try:
        params: Dict[str, str] = {
            "clientId": client_id or candidate_id,
            "candidateId": candidate_id,
            "positionId": position_id,
            "round": "4",
        }
        if question_set_id:
            params["questionSetId"] = question_set_id
        logger.info("[AdminReportAPI][Aptitude] MySQL fetch params: %s", params)
        r3 = await client.get(f"{cand_url}/public/question-answers", params=params)
        logger.info("[AdminReportAPI][Aptitude] MySQL /question-answers status=%d", r3.status_code)
        if r3.status_code == 200:
            rows = r3.json().get("data") or []
            for row in rows:
                qid = str(row.get("questionId") or "")
                if qid:
                    mysql_answers[qid] = {
                        "answerText": row.get("answerText") or "",
                        "correctAnswer": row.get("correctAnswer") or "",
                    }
            logger.info("[AdminReportAPI][Aptitude] MySQL answers fetched: %d rows, %d unique questionIds", len(rows), len(mysql_answers))
        else:
            logger.warning("[AdminReportAPI][Aptitude] MySQL %d: %s", r3.status_code, r3.text[:300])
    except Exception as e:
        logger.warning("[AdminReportAPI][Aptitude] MySQL fetch exception: %s", e)

    # ── Step 3: Merge question objects with candidate answers ────────────────
    if mongo_questions:
        merged = []
        for q in mongo_questions:
            qid = str(q.get("id") or "")
            ans_row = mysql_answers.get(qid) or {}
            merged.append({
                **q,
                "questionId": qid,
                "answerText": ans_row.get("answerText") or q.get("answerText") or "",
                "correctAnswer": ans_row.get("correctAnswer") or q.get("correctAnswer") or "",
            })
        logger.info("[AdminReportAPI][Aptitude] MERGED result: %d questions (%d with answers)", len(merged), sum(1 for m in merged if m.get("answerText")))
        return merged

    # Fallback: if no MongoDB questions, return MySQL rows only (no question text)
    if mysql_answers:
        logger.info("[AdminReportAPI][Aptitude] MongoDB empty — returning %d MySQL-only rows (no question text)", len(mysql_answers))
        return [
            {"questionId": qid, "answerText": v["answerText"], "correctAnswer": v["correctAnswer"]}
            for qid, v in mysql_answers.items()
        ]

    logger.warning("[AdminReportAPI][Aptitude] BOTH MongoDB and MySQL returned no aptitude data — r4_data will be []")
    return []


async def fetch_report_payload_data(
    position_id: str,
    candidate_id: str,
    client_id: str,
    tenant_id: str = "",
    question_set_id: str = "",
) -> Dict[str, Any]:
    """Reusable data aggregator used by both this API and report_generator."""
    resolved_tenant_id = (tenant_id or client_id or "").strip()
    admin_url = getattr(config, "ADMIN_BACKEND_URL", "http://localhost:8002").rstrip("/")
    cand_url = getattr(config, "CANDIDATE_BACKEND_URL", "http://localhost:8003").rstrip("/")
    headers = _service_headers()

    async with httpx.AsyncClient(timeout=30.0) as client:
        base_data, r1r2_data, r3_data, r4_data, assessment_summary = await asyncio.gather(
            _fetch_base_data(client, admin_url, headers, position_id, candidate_id, resolved_tenant_id),
            _fetch_conversational_data(client, cand_url, candidate_id, position_id, question_set_id or ""),
            _fetch_coding_data(client, cand_url, candidate_id, position_id),
            _fetch_aptitude_data(client, cand_url, candidate_id, position_id, question_set_id or "", client_id or ""),
            _fetch_assessment_summary(client, admin_url, headers, position_id, candidate_id, resolved_tenant_id),
            return_exceptions=False,
        )

    return {
        "positionId": position_id,
        "candidateId": candidate_id,
        "clientId": client_id,
        "tenantId": resolved_tenant_id,
        "questionSetId": question_set_id or "",
        "candidateProfile": base_data.get("candidateProfile") or {},
        "positionDetails": base_data.get("positionDetails", {}),
        "jobDescriptionText": base_data.get("jobDescriptionText", ""),
        "resumeText": base_data.get("resumeText", ""),
        "assessmentSummary": assessment_summary or {},
        "candidateAnswers": {
            "round1Round2": r1r2_data or {},
            "round3Coding": r3_data or {},
            "round4Aptitude": r4_data or [],
        },
    }


@router.get("/payload")
async def get_admin_report_payload(
    positionId: str = Query(..., description="Position ID"),
    candidateId: str = Query(..., description="Candidate ID"),
    clientId: str = Query(..., description="Client ID"),
    tenantId: str = Query("", description="Tenant DB name (optional; falls back to clientId)"),
    questionSetId: str = Query("", description="Question set ID (optional; used for round-4 answers)"),
    triggerReportGeneration: bool = Query(True, description="If true, queues report generation"),
):
    """
    Fetch complete report-input payload in one API call:
      - Admin position/JD/resume data
      - Assessment summary
      - Candidate round answers (R1/R2, R3 coding, R4 aptitude)

    Optionally queues report generation, which then updates:
      - MongoDB assessment_reports
      - MySQL assessment_report_generation + interview_evaluations (via AdminBackend internal API)
    """
    try:
        payload = await fetch_report_payload_data(
            position_id=positionId,
            candidate_id=candidateId,
            client_id=clientId,
            tenant_id=tenantId,
            question_set_id=questionSetId,
        )
        resolved_tenant_id = payload.get("tenantId") or (tenantId or clientId or "").strip()

        generation = {
            "status": "not_requested",
            "message": "Report generation not requested.",
            "queued": False,
        }
        if triggerReportGeneration:
            generation = await _queue_report_generation(
                position_id=positionId,
                candidate_id=candidateId,
                client_id=clientId,
                tenant_id=resolved_tenant_id,
                question_set_id=questionSetId,
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Admin report payload fetched successfully",
                "data": payload,
                "reportGeneration": generation,
            },
        )
    except Exception as e:
        logger.error("[AdminReportAPI] payload fetch failed: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Failed to fetch admin report payload: {str(e)}",
            },
        )


@router.get("/fetch-or-generate")
async def fetch_or_generate_report(
    positionId: str = Query(..., description="Position ID"),
    candidateId: str = Query(..., description="Candidate ID"),
    clientId: str = Query(..., description="Client ID"),
    tenantId: str = Query("", description="Tenant DB name (optional; falls back to clientId)"),
    questionSetId: str = Query("", description="Question set ID (optional)"),
    forceRegenerate: bool = Query(False, description="Force regeneration even if report already exists"),
):
    """
    Single API orchestration (SYNCHRONOUS):
      1) Check MySQL assessment_report_generation.is_generated
         → true  : fetch from MongoDB and return existing report AS-IS
                   (skip only if forceRegenerate=true was explicitly passed)
      2) Check MongoDB directly (catches MySQL out-of-sync)
         → isGenerated=true  : return it
         → generationStatus in {queued, in_progress} : still generating → 202
      3) Otherwise → generate new report synchronously (200)

    Use forceRegenerate=true to force a fresh regeneration regardless of the is_generated flag.
    """
    try:
        resolved_tenant_id = (tenantId or clientId or "").strip()

        import report_generator as rg

        # ── Step 1: Check MySQL is_generated flag ────────────────────────────
        if not forceRegenerate:
            generation_flag = await _get_report_generation_flag(positionId, candidateId)
            if generation_flag.get("isGenerated"):
                get_resp = await rg.get_report(positionId, candidateId)
                get_body = _json_from_response(get_resp)
                if get_resp.status_code == 200:
                    logger.info("[AdminReportAPI] Returning existing report (is_generated=true) for %s/%s", positionId, candidateId)
                    return JSONResponse(status_code=200, content=get_body)
                # MySQL said generated but MongoDB returned nothing — fall through to regenerate

        # ── Step 2: Check MongoDB directly (catches MySQL out-of-sync) ───────
        existing = await rg._get_existing_report(positionId, candidateId)
        if not forceRegenerate and existing and existing.get("isGenerated"):
            logger.info("[AdminReportAPI] Report found in MongoDB (MySQL flag out-of-sync) for %s/%s", positionId, candidateId)
            return JSONResponse(
                status_code=200,
                content=rg._build_completed_response(existing, is_existing=True),
            )

        # Report is actively being generated in this process — don't duplicate
        if not forceRegenerate and existing and existing.get("generationStatus") in {"queued", "in_progress"}:
            key = rg._queue_key(positionId, candidateId)
            if key in rg._in_progress or key in rg._queue_keys:
                logger.info("[AdminReportAPI] Generation already in progress for %s/%s", positionId, candidateId)
                return JSONResponse(status_code=202, content={
                    "success": True,
                    "status": "in_progress",
                    "message": "Report generation is already in progress. Please wait and retry.",
                })

        # ── Step 3: Generate new report synchronously ────────────────────────
        logger.info("[AdminReportAPI] Generating new report for %s/%s (synchronous)", positionId, candidateId)
        gen_req = rg.GenerateReportRequest(
            positionId=positionId,
            candidateId=candidateId,
            clientId=clientId,
            tenantId=resolved_tenant_id,
            questionSetId=questionSetId,
        )
        gen_resp = await rg.generate_report(gen_req)
        gen_body = _json_from_response(gen_resp)

        if gen_resp.status_code == 200:
            logger.info("[AdminReportAPI] Report generated successfully for %s/%s", positionId, candidateId)
            return JSONResponse(status_code=200, content=gen_body)

        # If concurrent duplicate guard triggered (202) — report will be in MongoDB soon
        if gen_resp.status_code == 202:
            return JSONResponse(status_code=202, content=gen_body)

        logger.warning("[AdminReportAPI] Report generation returned status %d for %s/%s",
                       gen_resp.status_code, positionId, candidateId)
        return JSONResponse(status_code=gen_resp.status_code, content=gen_body)

    except Exception as e:
        logger.error("[AdminReportAPI] fetch-or-generate failed: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Failed to fetch or generate report: {str(e)}",
            },
        )
