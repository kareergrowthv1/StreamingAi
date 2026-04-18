"""
Report Generator — Streaming AI
=================================
Single file for complete AI-driven assessment report generation.

Features:
  - FIFO asyncio queue: only ONE report generates at a time.
  - Duplicate guard: if the same positionId+candidateId is already in queue/generating → return 202.
  - Already-generated guard: if report already exists in CandidateBackend storage → return existing data.
  - Fetches all data: position details, JD, resume, R1/R2 (conversational), R3 (coding), R4 (aptitude/MCQ).
  - Full AI analysis with LLM: communication, technical, coding review, aptitude, soft skills, overall.
  - Persists report via CandidateBackend internal API (Streaming AI has no direct DB connection).
  - Updates MySQL: assessment_report_generation.is_generated = 1, interview_evaluations with scores.

Endpoints:
  POST /report/generate    — queue report generation
  GET  /report/get/{positionId}/{candidateId} — get existing report
"""

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from ai_config_loader import get_ai_config_sync
from gcs_upload import list_report_screenshot_urls

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/report", tags=["Report Generator"])

# ──────────────────────────────────────────────────────────────────────────────
# CandidateBackend persistence API (Streaming AI does not connect to DB directly)
# ──────────────────────────────────────────────────────────────────────────────
def _candidate_backend_url() -> str:
    return (getattr(config, "CANDIDATE_BACKEND_URL", "") or "").rstrip("/")


def _candidate_internal_headers() -> dict:
    token = (
        getattr(config, "INTERNAL_SERVICE_TOKEN", "")
        or getattr(config, "ADMIN_SERVICE_TOKEN", "")
        or ""
    ).strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Service-Token"] = token
    return headers


async def _ensure_report_index():
    """No-op in Streaming AI. Indexes are owned by CandidateBackend Mongo initialization."""
    logger.info("[ReportGen] Report indexes are managed by CandidateBackend (no direct DB connection in Streaming AI)")

# ──────────────────────────────────────────────────────────────────────────────
_report_queue: asyncio.Queue = asyncio.Queue()
_in_progress: Dict[str, bool] = {}   # key = f"{positionId}_{candidateId}"
_queue_keys: set = set()              # keys currently waiting in queue
_worker_started: bool = False


def _queue_key(position_id: str, candidate_id: str) -> str:
    return f"{position_id}_{candidate_id}".lower().replace("-", "")


async def _start_worker():
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    asyncio.ensure_future(_report_worker())
    logger.info("[ReportGen] Background worker started")


async def _report_worker():
    """Consumes items from the FIFO queue one at a time."""
    while True:
        try:
            item = await _report_queue.get()
            key = item["key"]
            _queue_keys.discard(key)
            _in_progress[key] = True
            logger.info("[ReportGen] Processing report: %s", key)
            await _set_generation_state(
                position_id=item.get("positionId", ""),
                candidate_id=item.get("candidateId", ""),
                client_id=item.get("clientId", ""),
                tenant_id=item.get("tenantId", ""),
                status="in_progress",
            )
            try:
                await _process_report(item)
            except Exception as e:
                logger.error("[ReportGen] Error processing %s: %s", key, e, exc_info=True)
                await _set_generation_state(
                    position_id=item.get("positionId", ""),
                    candidate_id=item.get("candidateId", ""),
                    client_id=item.get("clientId", ""),
                    tenant_id=item.get("tenantId", ""),
                    status="failed",
                    error_message=str(e),
                )
            finally:
                _in_progress.pop(key, None)
                _report_queue.task_done()
        except Exception as e:
            logger.error("[ReportGen] Worker error: %s", e, exc_info=True)
            await asyncio.sleep(1)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────────────────────
class GenerateReportRequest(BaseModel):
    positionId: str = Field(..., description="Position UUID")
    candidateId: str = Field(..., description="Candidate UUID")
    clientId: str = Field(default="", description="Client / organization ID")
    tenantId: str = Field(default="", description="Tenant DB name for MySQL")
    questionSetId: str = Field(default="", description="Question set ID for interview data lookup")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/generate")
async def generate_report(req: GenerateReportRequest):
    """
    Generate assessment report SYNCHRONOUSLY.
    - Always regenerates and upserts existing MongoDB doc + MySQL rows (no duplicate records).
    - Returns 200 + completed report on success.
    - Returns 202 if already generating in this process (concurrent duplicate guard).
    NOTE: For the "check is_generated then fetch or generate" flow, use
          GET /admin-report/fetch-or-generate (checks MySQL flag first).
    """
    key = _queue_key(req.positionId, req.candidateId)

    # 1. Prevent duplicate concurrent generation within the same process
    if key in _in_progress:
        logger.info("[ReportGen] Report already being generated for %s", key)
        return JSONResponse(status_code=202, content={
            "success": True,
            "status": "in_progress",
            "message": "Report generation already in progress.",
        })

    # 2. Generate synchronously — always upserts MongoDB doc + MySQL rows, then fetches fresh from DB
    _in_progress[key] = True
    try:
        await _set_generation_state(
            position_id=req.positionId,
            candidate_id=req.candidateId,
            client_id=req.clientId,
            tenant_id=req.tenantId,
            status="in_progress",
        )

        logger.info("[ReportGen] Starting synchronous report generation for %s", key)

        await _process_report({
            "key": key,
            "positionId": req.positionId,
            "candidateId": req.candidateId,
            "clientId": req.clientId,
            "tenantId": req.tenantId,
            "questionSetId": req.questionSetId,
        })

        # Always fetch the saved report from MongoDB — never return from memory
        logger.info("[ReportGen] Report generation completed for %s — fetching from DB", key)
        saved = await _get_existing_report(req.positionId, req.candidateId)
        if saved:
            return JSONResponse(status_code=200, content=_build_completed_response(saved, is_existing=True))

        return JSONResponse(status_code=500, content={
            "success": False,
            "status": "error",
            "message": "Report was saved but could not be retrieved from database.",
        })
        
    except Exception as e:
        logger.error("[ReportGen] Synchronous generation failed for %s: %s", key, e, exc_info=True)
        await _set_generation_state(
            position_id=req.positionId,
            candidate_id=req.candidateId,
            client_id=req.clientId,
            tenant_id=req.tenantId,
            status="failed",
            error_message=str(e),
        )
        return JSONResponse(status_code=500, content={
            "success": False,
            "status": "error",
            "message": f"Report generation failed: {str(e)}",
        })
    finally:
        _in_progress.pop(key, None)


@router.get("/get/{positionId}/{candidateId}")
async def get_report(positionId: str, candidateId: str, clientId: str = Query(default="")):
    """
    Get the existing generated report from MongoDB.
    Returns 404 if not generated yet.
    """
    key = _queue_key(positionId, candidateId)
    in_queue_or_processing = (key in _in_progress or key in _queue_keys)

    existing = await _get_existing_report(positionId, candidateId, clientId)
    if existing and existing.get("isGenerated"):
        enriched = await _inject_dynamic_screenshot_assets(existing, fallback_client_id=clientId)
        return JSONResponse(status_code=200, content=_build_completed_response(enriched, is_existing=True))

    persisted_status = (existing or {}).get("generationStatus")
    if persisted_status in {"queued", "in_progress"}:
        return JSONResponse(status_code=202, content={
            "success": True,
            "status": "in_progress",
            "message": "Report generation is in progress.",
        })

    if in_queue_or_processing:
        return JSONResponse(status_code=202, content={
            "success": True,
            "status": "in_progress",
            "message": "Report generation is in progress.",
        })

    return JSONResponse(status_code=404, content={
        "success": False,
        "status": "not_found",
        "message": "Report not found in MongoDB for the provided keys.",
    })


# ──────────────────────────────────────────────────────────────────────────────
# Core: Fetch all data + AI analysis + Save
# ──────────────────────────────────────────────────────────────────────────────
async def _process_report(item: dict):
    position_id = item["positionId"]
    candidate_id = item["candidateId"]
    client_id = item.get("clientId", "")
    tenant_id = item.get("tenantId", "")
    key = item["key"]

    logger.info("[ReportGen] === STEP 1: Fetching base data for %s ===", key)

    admin_url = getattr(config, "ADMIN_BACKEND_URL", "http://localhost:8002").rstrip("/")
    svc_token = getattr(config, "INTERNAL_SERVICE_TOKEN", "") or getattr(config, "ADMIN_SERVICE_TOKEN", "")

    headers = {"X-Service-Token": svc_token, "Content-Type": "application/json"} if svc_token else {"Content-Type": "application/json"}

    # Fetch all report input from dedicated admin_report_api provider only
    from admin_report_api import fetch_report_payload_data

    payload = await fetch_report_payload_data(
        position_id=position_id,
        candidate_id=candidate_id,
        client_id=client_id,
        tenant_id=tenant_id,
        question_set_id=item.get("questionSetId", ""),
    )

    position_details = payload.get("positionDetails") or {}
    candidate_profile = payload.get("candidateProfile") or {}
    jd_text = payload.get("jobDescriptionText") or ""
    resume_text = payload.get("resumeText") or ""
    assessment_summary = payload.get("assessmentSummary") or {}
    candidate_answers = payload.get("candidateAnswers") or {}
    r1r2_data = candidate_answers.get("round1Round2") or {}
    r3_data = candidate_answers.get("round3Coding") or {}
    r4_data = candidate_answers.get("round4Aptitude") or []

    logger.info("[ReportGen] === STEP 2: Running AI analysis ===")

    screenshot_assets = {
        "profilePicture": "",
        "calibration": [],
        "noFace": [],
        "multipleFaces": [],
        "allDirection": [],
        "all": [],
    }
    if client_id and position_id and candidate_id and config.GCS_BUCKET:
        try:
            screenshot_assets = await asyncio.to_thread(
                list_report_screenshot_urls,
                client_id,
                position_id,
                candidate_id,
                4,
            )
        except Exception as ss_err:
            logger.warning("[ReportGen] screenshot discovery failed: %s", ss_err)

    # Determine domain & round weights
    round_marks = _calc_round_marks(position_details, assessment_summary)

    # AI analysis
    analysis = await _run_full_analysis(
        candidate_id=candidate_id,
        position_id=position_id,
        position_details=position_details,
        jd_text=jd_text,
        resume_text=resume_text,
        r1r2_data=r1r2_data,
        r3_data=r3_data,
        r4_data=r4_data,
        round_marks=round_marks,
        assessment_summary=assessment_summary,
    )

    logger.info("[ReportGen] === STEP 3: Saving report to MongoDB ===")

    report_doc = _build_report_doc(
        position_id=position_id,
        candidate_id=candidate_id,
        client_id=client_id,
        tenant_id=tenant_id,
        candidate_profile=candidate_profile,
        position_details=position_details,
        jd_text=jd_text,
        resume_text=resume_text,
        assessment_summary=assessment_summary,
        round_marks=round_marks,
        analysis=analysis,
        r1r2_data=r1r2_data,
        r3_data=r3_data,
        r4_data=r4_data,
        screenshot_assets=screenshot_assets,
    )

    await _save_report(position_id, candidate_id, report_doc)
    await _set_generation_state(
        position_id=position_id,
        candidate_id=candidate_id,
        client_id=client_id,
        tenant_id=tenant_id,
        status="completed",
    )

    logger.info("[ReportGen] === STEP 4: Updating MySQL tables ===")

    scores = analysis.get("scores", {})
    await _mark_report_generated(admin_url, headers, position_id, candidate_id, tenant_id, scores)

    logger.info("[ReportGen] Report completed and saved to MongoDB for %s", key)
    # No in-memory return — caller fetches from MongoDB


# ──────────────────────────────────────────────────────────────────────────────
# Data Fetching Helpers
# ──────────────────────────────────────────────────────────────────────────────
async def _fetch_base_data(client: httpx.AsyncClient, admin_url: str, headers: dict,
                            position_id: str, candidate_id: str, tenant_id: str) -> dict:
    try:
        resp = await client.post(
            f"{admin_url}/internal/report-data",
            headers=headers,
            json={"positionId": position_id, "candidateId": candidate_id, "tenantId": tenant_id},
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("[ReportGen] fetch base data failed: %s", e)
    return {}


async def _fetch_assessment_summary(client: httpx.AsyncClient, admin_url: str, headers: dict,
                                     position_id: str, candidate_id: str, tenant_id: str) -> dict:
    """Fetch assessment summary from AdminBackend."""
    try:
        params = {"positionId": position_id, "candidateId": candidate_id}
        if tenant_id:
            params["tenantId"] = tenant_id
        resp = await client.get(
            f"{admin_url}/candidates/assessment-summaries",
            headers={**headers, **({"X-Tenant-Id": tenant_id} if tenant_id else {})},
            params=params,
        )
        if resp.status_code == 200:
            body = resp.json()
            return body.get("data") or body or {}
    except Exception as e:
        logger.warning("[ReportGen] fetch assessment summary failed: %s", e)
    return {}


async def _fetch_conversational_data(client: httpx.AsyncClient, cand_url: str,
                                      candidate_id: str, position_id: str) -> dict:
    """Fetch round 1 & 2 interview Q&A from CandidateBackend."""
    try:
        resp = await client.get(
            f"{cand_url}/candidate-interviews/candidate/{candidate_id}/position/{position_id}",
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("[ReportGen] fetch conversational data failed: %s", e)
    return {}


async def _fetch_coding_data(client: httpx.AsyncClient, cand_url: str,
                              candidate_id: str, position_id: str) -> dict:
    """Fetch round 3 (coding) responses from CandidateBackend."""
    try:
        resp = await client.get(
            f"{cand_url}/candidate-coding-responses/candidate/{candidate_id}/position/{position_id}",
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("[ReportGen] fetch coding data failed: %s", e)
    return {}


async def _fetch_aptitude_data(client: httpx.AsyncClient, cand_url: str,
                                candidate_id: str, position_id: str, question_set_id: str = "") -> list:
    """Fetch round 4 (aptitude MCQ) answers from CandidateBackend (public question-answers)."""
    try:
        params = {"candidateId": candidate_id, "positionId": position_id, "round": "4"}
        if question_set_id:
            params["questionSetId"] = question_set_id
        resp = await client.get(f"{cand_url}/public/question-answers", params=params)
        if resp.status_code == 200:
            body = resp.json()
            return body.get("data") or []
    except Exception as e:
        logger.warning("[ReportGen] fetch aptitude data failed: %s", e)
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Score / Round Mark Calculation
# ──────────────────────────────────────────────────────────────────────────────
def _calc_round_marks(position_details: dict, assessment_summary: dict) -> dict:
    """
    Calculate max marks per round based on domain type and which rounds are assigned.
    TECH:     General(10) + Position(40) + Coding(40) + Aptitude(10) = 100
    NON_TECH: General(50) + Position(40) + Aptitude(10) = 100  (or 50+50 if no aptitude)
    """
    domain = (position_details.get("domainType") or "TECH").upper()
    if domain in ("NON_TECH", "NON-TECH", "NONTECH"):
        domain = "NON_TECH"
    else:
        domain = "TECH"

    r1a = bool(assessment_summary.get("round1Assigned") or assessment_summary.get("round1_assigned", True))
    r2a = bool(assessment_summary.get("round2Assigned") or assessment_summary.get("round2_assigned", True))
    r3a = bool(assessment_summary.get("round3Assigned") or assessment_summary.get("round3_assigned", False))
    r4a = bool(assessment_summary.get("round4Assigned") or assessment_summary.get("round4_assigned", False))

    marks = {"general": 0, "position": 0, "coding": 0, "aptitude": 0}

    if domain == "TECH":
        if r1a: marks["general"] = 10
        if r4a: marks["aptitude"] = 10
        if r2a and r3a:
            marks["position"] = 40; marks["coding"] = 40
        elif r2a and not r3a:
            marks["position"] = 90 - marks["aptitude"]
        elif r3a and not r2a:
            marks["coding"] = 90 - marks["aptitude"]
        elif r2a and r3a and not r4a:
            marks["position"] = 45; marks["coding"] = 45
    else:
        if r1a: marks["general"] = 50
        if r4a: marks["aptitude"] = 10
        if r2a and r4a:
            marks["position"] = 40
        elif r2a:
            marks["position"] = 50

    return {
        "general": marks["general"],
        "position": marks["position"],
        "coding": marks["coding"],
        "aptitude": marks["aptitude"],
        "total": marks["general"] + marks["position"] + marks["coding"] + marks["aptitude"],
        "domain": domain,
        "assignedRounds": {"r1": r1a, "r2": r2a, "r3": r3a, "r4": r4a},
    }


def _score_aptitude(r4_answers: list) -> dict:
    """Score aptitude by comparing answer_text vs correct_answer."""
    total = len([q for q in r4_answers if q.get("correctAnswer")])
    if total == 0:
        return {"correct": 0, "total": 0, "percentage": 0.0}
    correct = sum(
        1 for q in r4_answers
        if q.get("correctAnswer") and (q.get("answerText") or "").strip() == str(q.get("correctAnswer") or "").strip()
    )
    return {"correct": correct, "total": total, "percentage": round((correct / total) * 100, 1)}


def _score_coding(r3_data: dict) -> dict:
    """Score coding by test cases passed."""
    total_cases = 0
    passed_cases = 0
    question_count = 0
    sets = r3_data.get("codingQuestionSets") or []
    for qs in sets:
        for q in (qs.get("questions") or []):
            question_count += 1
            tc = q.get("totalTestCases") or 0
            pc = q.get("testCasesPassed") or 0
            total_cases += tc
            passed_cases += pc
    pct = round((passed_cases / total_cases) * 100, 1) if total_cases > 0 else 0.0
    return {"passed": passed_cases, "total": total_cases, "questions": question_count, "percentage": pct}


def _extract_qa_pairs(interview_data: dict, round_key: str) -> list:
    """Extract Q&A pairs from candidate_interviews categories."""
    category = (interview_data.get("categories") or {}).get(round_key) or {}
    conv_sets = category.get("conversationSets") or {}
    pairs = []
    for _ck, conv_list in conv_sets.items():
        if isinstance(conv_list, list):
            for item in conv_list:
                if isinstance(item, dict):
                    q = (item.get("question") or "").strip()
                    a = (item.get("answer") or "").strip()
                    if q or a:
                        pairs.append({"question": q, "answer": a})
    return pairs


def _format_qa_for_prompt(pairs: list, label: str) -> str:
    if not pairs:
        return f"{label}: No answers recorded."
    lines = [f"{label}:"]
    for i, p in enumerate(pairs, 1):
        lines.append(f"  Q{i}: {p['question']}")
        lines.append(f"  A{i}: {p['answer'] or '[No answer]'}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# LLM Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_openai_client():
    """Get OpenAI client using dynamic config from Superadmin."""
    try:
        from openai import OpenAI
        ai_cfg = get_ai_config_sync()
        api_key = ai_cfg.get("apiKey") or ai_cfg.get("openai_api_key") or ""
        base_url = ai_cfg.get("baseUrl") or ai_cfg.get("openai_base_url") or "https://api.openai.com/v1"
        model = ai_cfg.get("model") or ai_cfg.get("openai_model") or "gpt-4o-mini"
        if not api_key:
            logger.warning("[ReportGen] No OpenAI API key — fallback analysis will be used")
            return None, model
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, model
    except Exception as e:
        logger.warning("[ReportGen] OpenAI init failed: %s", e)
        return None, "gpt-4o-mini"


def _llm_call(client, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 800, temp: float = 0.4) -> str:
    """Synchronous LLM call. Returns empty string on failure."""
    if client is None:
        return ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("[ReportGen] LLM call failed: %s", e)
        return ""


def _parse_float(text: str, default: float = 0.0, max_val: float = 100.0) -> float:
    """Extract first float from a string."""
    m = re.search(r"\d+(?:\.\d+)?", (text or ""))
    if m:
        v = float(m.group())
        return min(max_val, max(0.0, v))
    return default


def _parse_json_from_llm(text: str) -> dict:
    """Try to parse JSON from LLM response, handling markdown fences."""
    if not text:
        return {}
    # strip markdown fences
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        return {}


def _generate_resume_summary(client, model: str, resume_text: str, pos_title: str, skills_text: str) -> list:
    """Generate 6-7 concise bullet point resume summary via LLM (resume only — no JD comparison)."""
    if not resume_text or not resume_text.strip():
        return [f"No resume data was submitted for the {pos_title} candidate."]
    result = _llm_call(
        client, model,
        "You are a professional resume analyst. Produce a structured, concise candidate summary based solely on the resume provided.",
        f"Candidate Resume:\n{resume_text[:2500]}\n\n"
        "Generate exactly 6-7 bullet points that objectively summarise this candidate's resume. Cover:\n"
        "1) Current/most recent role and total years of experience\n"
        "2) Primary technical skills, languages, and tools mentioned in the resume\n"
        "3) Educational qualifications\n"
        "4) Key achievements, projects, or measurable impact metrics\n"
        "5) Domain/industry background and experience areas\n"
        "6) Any certifications, notable open-source work, or publications (if present)\n"
        "7) Any standout quality, unique differentiator, or additional strength (if evident)\n"
        "Do NOT compare with any job description or role requirements. Summarise only what the resume says.\n"
        "Return a JSON array of exactly 6-7 strings (each a single-sentence bullet, no bullet symbols). "
        "Return ONLY the JSON array, no extra text.",
        700, 0.3,
    )
    # Try JSON array parse
    points = []
    try:
        clean = re.sub(r"```(?:json)?", "", result).replace("```", "").strip()
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            points = [str(p).strip() for p in parsed if str(p).strip()]
    except Exception:
        pass
    # Fallback: split lines
    if not points and result:
        for line in result.splitlines():
            line = line.strip().lstrip("•-*0123456789.) ")
            if len(line) > 10:
                points.append(line)
    return points[:7] if points else [
        f"Candidate has applied for the {pos_title} position.",
        f"Technical background reviewed against required skills: {skills_text[:100]}.",
        "Resume submitted and processed for comprehensive assessment.",
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Full AI Analysis
# ──────────────────────────────────────────────────────────────────────────────
async def _run_full_analysis(
    candidate_id: str,
    position_id: str,
    position_details: dict,
    jd_text: str,
    resume_text: str,
    r1r2_data: dict,
    r3_data: dict,
    r4_data: list,
    round_marks: dict,
    assessment_summary: dict,
) -> dict:
    """
    Run all AI analysis for all 4 rounds. Returns a comprehensive analysis dict.
    """
    loop = asyncio.get_event_loop()
    client, model = await loop.run_in_executor(None, _get_openai_client)

    pos_title = position_details.get("title") or "the role"
    domain = round_marks.get("domain", "TECH")
    mandatory_skills = position_details.get("mandatorySkills") or []
    optional_skills = position_details.get("optionalSkills") or []
    all_skills = mandatory_skills + optional_skills

    # Extract Q&A pairs
    general_pairs = _extract_qa_pairs(r1r2_data, "generalQuestion")
    position_pairs = _extract_qa_pairs(r1r2_data, "positionSpecificQuestion")

    skills_text = ", ".join(all_skills[:15]) if all_skills else "various technical skills"
    resume_snippet = resume_text[:1200] if resume_text else "No resume data available."
    jd_snippet = jd_text[:600] if jd_text else "No job description available."

    # ── AI Resume Summary (6-7 bullet points) ─────────────────────────────────
    resume_summary_points = await loop.run_in_executor(
        None, _generate_resume_summary, client, model, resume_text, pos_title, skills_text
    )

    # ── Round 1: General Communication Analysis ───────────────────────────────
    general_analysis = {}
    general_score_raw = 0.0

    if general_pairs:
        # Have actual Q&A — evaluate communication quality
        general_qa_text = _format_qa_for_prompt(general_pairs, "General Interview Q&A")
        gen_resp = await loop.run_in_executor(None, _llm_call, client, model,
            "You are a senior HR communication evaluator. Evaluate the candidate's answers strictly for "
            "communication quality (fluency, grammar, confidence, clarity, structure). "
            "Return JSON ONLY with exactly these keys: "
            "fluency (0-10), grammar (0-10), confidence (0-10), clarity (0-10), "
            "overall_score (0-100), strengths (list of 3-5 strings), improvements (list of 2-4 strings), summary (string 2-3 sentences).",
            f"Position: {pos_title}\n\n{general_qa_text}\n\nResume: {resume_snippet[:400]}\n\n"
            "Rate communication quality rigorously. Return valid JSON ONLY.",
            900, 0.4)
        general_analysis = _parse_json_from_llm(gen_resp) or {}
        general_score_raw = float(general_analysis.get("overall_score") or 0)

        # Per-question AI comments
        for pair in general_pairs:
            if pair.get("answer") and len(pair["answer"]) > 5:
                comment = await loop.run_in_executor(None, _llm_call, client, model,
                    "You evaluate interview answers. Be concise (2-3 sentences max).",
                    f"Q: {pair['question']}\nA: {pair['answer']}\n"
                    "Give a brief communication quality comment on fluency, clarity, and grammar.",
                    160, 0.4)
                pair["aiComment"] = comment
            else:
                pair["aiComment"] = "No answer recorded for this question."

    else:
        # No Q&A data — evaluate purely from resume text against communication profile
        gen_resp = await loop.run_in_executor(None, _llm_call, client, model,
            "You are a senior HR evaluator. No interview recording was available for this candidate. "
            "Evaluate the candidate's likely communication profile based on their resume content, "
            "writing style, clarity of expression, and professional tone. "
            "Return JSON ONLY with exactly these keys: "
            "fluency (0-10), grammar (0-10), confidence (0-10), clarity (0-10), "
            "overall_score (0-100), strengths (list of 3-5 strings), improvements (list of 2-4 strings), "
            "summary (string 2-3 sentences describing communication profile from resume).",
            f"Position Applied: {pos_title}\n\nResume:\n{resume_snippet}\n\n"
            "Evaluate written communication quality from resume. Note: No live interview data. Return valid JSON ONLY.",
            900, 0.4)
        general_analysis = _parse_json_from_llm(gen_resp) or {}
        general_score_raw = float(general_analysis.get("overall_score") or 0)

    # Fallback if LLM returned no/bad JSON for general analysis
    if not general_analysis or not general_analysis.get("summary"):
        general_analysis = {
            "fluency": 5, "grammar": 5, "confidence": 5, "clarity": 5,
            "overall_score": 50,
            "strengths": ["Resume demonstrates technical writing ability", "Structured presentation of experience"],
            "improvements": ["Live interview data not available for deeper assessment"],
            "summary": f"Communication assessment based on resume review for {pos_title}. "
                       "No live interview session recorded; evaluation derived from written profile."
        }
        general_score_raw = 50.0

    # ── Round 2: Position-Specific Technical Analysis ────────────────────────
    position_analysis = {}
    position_score_raw = 0.0

    if position_pairs:
        # Full Q&A-based technical evaluation
        pos_qa_text = _format_qa_for_prompt(position_pairs, "Position-Specific Q&A")
        pos_resp = await loop.run_in_executor(None, _llm_call, client, model,
            f"You are a technical evaluator for {pos_title}. Evaluate the candidate's answers for technical depth, "
            f"accuracy, and relevance. Required skills: {skills_text}. "
            "Return JSON ONLY with exactly these keys: "
            "technical_depth (0-10), accuracy (0-10), relevance (0-10), "
            "overall_score (0-100), strengths (list of 3-5 strings), improvements (list of 2-4 strings), "
            "summary (string, 2-3 sentences on technical fit and knowledge gaps).",
            f"JD: {jd_snippet}\n\n{pos_qa_text}\n\n"
            "Evaluate technical knowledge and role fit strictly. Return valid JSON ONLY.",
            1000, 0.4)
        position_analysis = _parse_json_from_llm(pos_resp) or {}
        position_score_raw = float(position_analysis.get("overall_score") or 0)

        for pair in position_pairs:
            if pair.get("answer") and len(pair["answer"]) > 5:
                comment = await loop.run_in_executor(None, _llm_call, client, model,
                    f"You evaluate technical interview answers for {pos_title}. Be concise (2-3 sentences).",
                    f"Q: {pair['question']}\nA: {pair['answer']}\nRequired skills: {skills_text}\n"
                    "Give a brief technical evaluation: correctness, depth, and relevance.",
                    160, 0.4)
                pair["aiComment"] = comment
            else:
                pair["aiComment"] = "No answer recorded for this question."

    else:
        # No Q&A — do JD-resume match analysis
        pos_resp = await loop.run_in_executor(None, _llm_call, client, model,
            f"You are a technical recruiter evaluating fit for {pos_title}. "
            "No technical interview Q&A data is available. Assess only based on resume vs job description. "
            "Analyze skills match, experience relevance, and technical readiness. "
            "Return JSON ONLY with exactly these keys: "
            "technical_depth (0-10), accuracy (0-10), relevance (0-10), "
            "overall_score (0-100), strengths (list of 3-5 strings based on resume skills match), "
            "improvements (list of 2-4 strings listing skill gaps vs JD), "
            "summary (string, 2-3 sentences on resume-JD fit and technical readiness).",
            f"Job Description:\n{jd_snippet}\n\nRequired Skills: {skills_text}\n\nResume:\n{resume_snippet}\n\n"
            "Rate resume-JD technical fit rigorously. Return valid JSON ONLY.",
            1000, 0.4)
        position_analysis = _parse_json_from_llm(pos_resp) or {}
        position_score_raw = float(position_analysis.get("overall_score") or 0)

    # Fallback if LLM returned no/bad JSON for position analysis
    if not position_analysis or not position_analysis.get("summary"):
        position_analysis = {
            "technical_depth": 5, "accuracy": 5, "relevance": 5,
            "overall_score": 50,
            "strengths": [f"Resume indicates experience relevant to {pos_title}", "Technical background aligns with required domain"],
            "improvements": ["Deeper technical assessment required through live interview", f"Verify hands-on proficiency in {skills_text[:100]}"],
            "summary": f"Technical evaluation for {pos_title} based on resume-JD comparison. "
                       "Interview session data was not available; score reflects resume match analysis."
        }
        position_score_raw = 50.0

    # ── Round 3: Coding Analysis ─────────────────────────────────────────────
    coding_score_data = _score_coding(r3_data)
    coding_analysis = {}
    coding_sets_analyzed = []
    if r3_data and r3_data.get("codingQuestionSets"):
        for qs in (r3_data.get("codingQuestionSets") or []):
            for q in (qs.get("questions") or []):
                submissions = q.get("submissions") or []
                latest = submissions[-1] if isinstance(submissions, list) and submissions else {}
                source_code = latest.get("sourceCode", "") if isinstance(latest, dict) else ""
                lang = latest.get("programmingLanguage", "") if isinstance(latest, dict) else ""
                q_title = q.get("questionTitle", "")
                q_desc = q.get("questionDescription", "")
                tc_total = q.get("totalTestCases", 0) or 0
                tc_passed = q.get("testCasesPassed", 0) or 0

                # Extract error/stderr from submission if available
                exec_error = ""
                for sub in (submissions if isinstance(submissions, list) else []):
                    if isinstance(sub, dict):
                        err = sub.get("error") or sub.get("stderr") or sub.get("compilationError") or ""
                        if err:
                            exec_error = str(err)[:400]
                            break

                # ── 5-point AI analysis + code quality score (0-100) ──────────
                code_quality_score = 0.0
                ai_analysis_points = []
                if source_code and source_code.strip() and client:
                    tc_pct_q = round((tc_passed / tc_total * 100), 1) if tc_total > 0 else 0.0
                    review_resp = await loop.run_in_executor(None, _llm_call, client, model,
                        f"You are a senior software engineer doing a rigorous code review for a {pos_title} candidate assessment.",
                        f"Problem Title: {q_title}\n"
                        f"Problem Description: {q_desc[:500] if q_desc else 'N/A'}\n"
                        f"Language: {lang or 'Unknown'}\n"
                        f"Test Cases: {tc_passed}/{tc_total} passed\n"
                        + (f"Execution Error:\n{exec_error}\n" if exec_error else "")
                        + f"\nCandidate Code:\n{source_code[:2000]}\n\n"
                        "Return JSON ONLY with exactly these keys:\n"
                        "  codeQualityScore (integer 0-100 — evaluate logic correctness, approach, completeness),\n"
                        "  analysis (array of EXACTLY 5 strings covering in order:\n"
                        "    1. Code Logic & Correctness: does the approach solve the problem correctly?\n"
                        "    2. Time & Space Complexity: efficiency analysis\n"
                        "    3. Edge Case & Error Handling: how well edge cases are handled\n"
                        "    4. Code Quality & Readability: naming, structure, style\n"
                        "    5. Test Case Performance: explain the {tc_passed}/{tc_total} result and what failed)\n"
                        "Each analysis string must be 1-2 sentences, specific to THIS candidate's code.\n"
                        "Return valid JSON ONLY — no extra text.",
                        700, 0.4)
                    review_json = _parse_json_from_llm(review_resp)
                    code_quality_score = float(review_json.get("codeQualityScore") or 0)
                    code_quality_score = max(0.0, min(100.0, code_quality_score))
                    ai_analysis_points = review_json.get("analysis") or []
                    if isinstance(ai_analysis_points, list):
                        ai_analysis_points = [str(p).strip() for p in ai_analysis_points if str(p).strip()]
                    # Ensure exactly 5 points
                    labels = ["Code Logic & Correctness", "Time & Space Complexity",
                              "Edge Case & Error Handling", "Code Quality & Readability",
                              "Test Case Performance"]
                    while len(ai_analysis_points) < 5:
                        ai_analysis_points.append(
                            f"{labels[len(ai_analysis_points)]}: Insufficient data to evaluate."
                        )
                    ai_analysis_points = ai_analysis_points[:5]
                elif not source_code or not source_code.strip():
                    ai_analysis_points = [
                        "Code Logic & Correctness: No code was submitted for this problem.",
                        "Time & Space Complexity: Cannot evaluate — no submission available.",
                        "Edge Case & Error Handling: Cannot evaluate — no submission available.",
                        "Code Quality & Readability: Cannot evaluate — no submission available.",
                        f"Test Case Performance: {tc_passed}/{tc_total} test cases — no code submitted.",
                    ]

                # Composite score: 65% code quality + 35% test cases
                tc_pct_for_score = round((tc_passed / tc_total * 100), 1) if tc_total > 0 else 0.0
                composite_q_score = round(0.65 * code_quality_score + 0.35 * tc_pct_for_score, 1)

                coding_sets_analyzed.append({
                    "questionTitle": q_title,
                    "questionDescription": q_desc,
                    "language": lang,
                    "testCasesPassed": tc_passed,
                    "totalTestCases": tc_total,
                    "testCasePercentage": tc_pct_for_score,
                    "codeQualityScore": round(code_quality_score, 1),
                    "compositeScore": composite_q_score,
                    "executionStatus": q.get("executionStatus", ""),
                    "executionError": exec_error,
                    "sourceCode": source_code[:2000] if source_code else "",
                    "aiAnalysisPoints": ai_analysis_points,
                    "tags": q.get("tags", []),
                })

        if coding_sets_analyzed:
            # Overall composite coding score (average of all question composite scores)
            composite_overall = round(
                sum(q["compositeScore"] for q in coding_sets_analyzed) / len(coding_sets_analyzed), 1
            )
            if client:
                tc_total_all = coding_score_data["total"]
                tc_passed_all = coding_score_data["passed"]
                per_q_summary = "\n".join(
                    f"- {q['questionTitle']}: score {q['compositeScore']}%, "
                    f"{q['testCasesPassed']}/{q['totalTestCases']} test cases, "
                    f"analysis: {'; '.join(q['aiAnalysisPoints'][:3])}"
                    for q in coding_sets_analyzed
                )
                coding_json_resp = await loop.run_in_executor(None, _llm_call, client, model,
                    f"You evaluate overall coding round performance for {pos_title} candidates.",
                    f"Candidate attempted {len(coding_sets_analyzed)} coding problem(s).\n"
                    f"Test cases: {tc_passed_all}/{tc_total_all} passed ({coding_score_data['percentage']}%).\n"
                    f"Composite coding score (65% code quality + 35% test cases): {composite_overall}%.\n"
                    f"Per-problem breakdown:\n{per_q_summary}\n\n"
                    "Return ONLY a JSON object with exactly these three keys:\n"
                    "  overallReview: a 2-3 sentence overall coding assessment (string)\n"
                    "  strengths: list of 2-4 strings describing what the candidate did well\n"
                    "  concerns: list of 2-3 strings describing areas needing improvement\n"
                    "Example: {\"overallReview\": \"...\", \"strengths\": [\"...\", \"...\"], \"concerns\": [\"...\"]}\n"
                    "No extra text outside the JSON.",
                    400, 0.4)
                coding_sc = _parse_json_from_llm(coding_json_resp)
                coding_analysis["overallReview"] = (
                    coding_sc.get("overallReview") or coding_json_resp or ""
                )
                coding_analysis["strengths"] = coding_sc.get("strengths") or []
                coding_analysis["concerns"] = coding_sc.get("concerns") or []
            coding_analysis["score"] = composite_overall
            coding_analysis["testCaseScore"] = coding_score_data["percentage"]
            coding_analysis["codeQualityAvg"] = round(
                sum(q["codeQualityScore"] for q in coding_sets_analyzed) / len(coding_sets_analyzed), 1
            )
        else:
            coding_analysis["score"] = coding_score_data["percentage"]
        coding_analysis["questions"] = coding_sets_analyzed

    # ── Round 4: Aptitude Analysis ───────────────────────────────────────────
    aptitude_score_data = _score_aptitude(r4_data)
    aptitude_analysis = {"questions": [], "score": aptitude_score_data["percentage"]}
    for qa in r4_data:
        cand_ans = str(qa.get("answerText") or qa.get("answer") or "").strip()
        correct_ans = str(qa.get("correctAnswer") or "").strip()
        is_correct = bool(correct_ans) and (cand_ans == correct_ans)
        aptitude_analysis["questions"].append({
            "questionId": qa.get("questionId") or "",
            "question": qa.get("question") or qa.get("questionText") or qa.get("text") or "",
            "options": qa.get("options") or [],
            "candidateAnswer": cand_ans,
            "correctAnswer": correct_ans,
            "isCorrect": is_correct,
            "explanation": ("Correct" if is_correct else (
                f"Incorrect — candidate answered '{cand_ans}', correct answer is '{correct_ans}'."
                if cand_ans else "Not attempted"
            )),
        })
    if r4_data and client:
        apt_resp = await loop.run_in_executor(None, _llm_call, client, model,
            "Evaluate aptitude test results briefly.",
            f"Candidate scored {aptitude_score_data['correct']}/{aptitude_score_data['total']} "
            f"({aptitude_score_data['percentage']}%) on aptitude MCQs. "
            f"Provide a 1-2 sentence assessment of aptitude performance.",
            150, 0.4)
        aptitude_analysis["overallReview"] = apt_resp

    # ── Score Calculation (weighted by round_marks) ───────────────────────────
    total_max = round_marks.get("total", 100) or 100
    gen_max = round_marks.get("general", 0)
    pos_max = round_marks.get("position", 0)
    cod_max = round_marks.get("coding", 0)
    apt_max = round_marks.get("aptitude", 0)

    # Convert % scores to marks
    gen_marks = round((general_score_raw / 100) * gen_max, 1) if gen_max else 0
    pos_marks = round((position_score_raw / 100) * pos_max, 1) if pos_max else 0
    cod_marks = round((coding_score_data["percentage"] / 100) * cod_max, 1) if cod_max else 0
    apt_marks = round((aptitude_score_data["percentage"] / 100) * apt_max, 1) if apt_max else 0

    total_marks_obtained = gen_marks + pos_marks + cod_marks + apt_marks
    overall_pct = round((total_marks_obtained / total_max) * 100, 1) if total_max else 0

    if overall_pct >= 70:
        recommendation = "RECOMMENDED"
    elif overall_pct >= 50:
        recommendation = "CAUTIOUSLY_RECOMMENDED"
    else:
        recommendation = "NOT_RECOMMENDED"

    logger.info(
        "[ReportGen] AI Recommendation determined: %s (overall_pct=%s, candidate=%s, position=%s)",
        recommendation, overall_pct, candidate_id[:8], position_id[:8]
    )

    # Soft skills from general analysis
    soft_skills = {
        "fluency": general_analysis.get("fluency") or 0,
        "grammar": general_analysis.get("grammar") or 0,
        "confidence": general_analysis.get("confidence") or 0,
        "clarity": general_analysis.get("clarity") or 0,
    }

    # ── Overall Summary ───────────────────────────────────────────────────────
    overall_summary = {}
    summary_parts = []
    has_interview_data = bool(general_pairs or position_pairs)

    if general_pairs:
        summary_parts.append(f"Communication score: {general_score_raw:.0f}/100")
    if position_pairs:
        summary_parts.append(f"Technical/position score: {position_score_raw:.0f}/100")
    if r3_data and r3_data.get("codingQuestionSets"):
        summary_parts.append(f"Coding: {coding_score_data['passed']}/{coding_score_data['total']} test cases passed")
    if r4_data:
        summary_parts.append(f"Aptitude: {aptitude_score_data['correct']}/{aptitude_score_data['total']} correct")

    data_source_note = (
        "Live interview session data was available for analysis."
        if has_interview_data else
        "Note: This assessment is based on resume-to-JD matching; no live interview session data was captured."
    )

    ov_resp = await loop.run_in_executor(None, _llm_call, client, model,
        f"You write comprehensive candidate assessment reports for {pos_title}. "
        "Be thorough, professional, and specific. Never leave any section empty. "
        "Return JSON ONLY with exactly these keys: "
        "executiveSummary (string, 3-4 sentences summarizing overall assessment), "
        "strengths (list of 4-6 specific bullet strings, each starting with a verb or skill noun), "
        "areasForImprovement (list of 3-5 specific bullet strings), "
        "resumeMatchAnalysis (string, 2-3 sentences on resume fit for the role), "
        "skillsGapAnalysis (string, 2-3 sentences on skill gaps identified), "
        "finalRecommendation (string, 1-2 sentences with clear recommendation and reasoning).",
        f"Position: {pos_title}\n"
        f"Domain: {domain}\n"
        f"Required Skills: {skills_text}\n"
        f"Resume Summary:\n{resume_snippet}\n\n"
        f"JD Summary:\n{jd_snippet}\n\n"
        f"Performance Summary: {'; '.join(summary_parts) if summary_parts else 'Evaluated via resume-JD analysis'}\n"
        f"Overall Score: {overall_pct}% → {recommendation}\n"
        f"Communication Strengths: {', '.join(general_analysis.get('strengths') or [])}\n"
        f"Technical Strengths: {', '.join(position_analysis.get('strengths') or [])}\n"
        f"{data_source_note}\n\n"
        "Write a complete, specific assessment report. Return valid JSON ONLY.",
        1100, 0.5)
    overall_summary = _parse_json_from_llm(ov_resp) or {}

    # Robust fallback — always ensure all summary fields are populated
    if not overall_summary.get("executiveSummary"):
        overall_summary["executiveSummary"] = (
            f"Candidate assessed for the {pos_title} role with an overall score of {overall_pct}% ({recommendation}). "
            f"Resume review indicates {'strong' if overall_pct >= 60 else 'partial'} alignment with the required technical profile. "
            f"{data_source_note}"
        )
    if not overall_summary.get("strengths"):
        overall_summary["strengths"] = (
            general_analysis.get("strengths") or [] + position_analysis.get("strengths") or []
        ) or [
            f"Demonstrates relevant educational background for {pos_title}",
            f"Resume reflects familiarity with {skills_text[:80]}",
            "Structured and clear professional profile presentation"
        ]
    if not overall_summary.get("areasForImprovement"):
        overall_summary["areasForImprovement"] = (
            general_analysis.get("improvements") or [] + position_analysis.get("improvements") or []
        ) or [
            "Live interview performance data not available for complete evaluation",
            f"In-depth assessment of hands-on {pos_title} skills required",
            "Communication and problem-solving skills to be verified in person"
        ]
    if not overall_summary.get("resumeMatchAnalysis"):
        overall_summary["resumeMatchAnalysis"] = (
            f"Resume demonstrates {('strong' if overall_pct >= 60 else 'moderate')} alignment with {pos_title} requirements. "
            f"Key skills from resume include: {skills_text[:120]}."
        )
    if not overall_summary.get("skillsGapAnalysis"):
        overall_summary["skillsGapAnalysis"] = (
            f"Areas requiring further verification include advanced proficiency in {skills_text[:100]}. "
            "A live technical assessment is recommended for complete skills validation."
        )
    if not overall_summary.get("finalRecommendation"):
        overall_summary["finalRecommendation"] = (
            f"{recommendation.replace('_', ' ')} — "
            f"Candidate scored {overall_pct}% and "
            f"{'meets the minimum threshold for progression' if overall_pct >= 50 else 'does not meet the minimum threshold at this time'}."
        )

    return {
        "resumeSummaryPoints": resume_summary_points,
        "generalAnalysis": {
            "qaPairs": general_pairs,
            "fluency": general_analysis.get("fluency"),
            "grammar": general_analysis.get("grammar"),
            "confidence": general_analysis.get("confidence"),
            "clarity": general_analysis.get("clarity"),
            "strengths": general_analysis.get("strengths") or [],
            "improvements": general_analysis.get("improvements") or [],
            "summary": general_analysis.get("summary") or "",
            "scorePercentage": general_score_raw,
        },
        "positionAnalysis": {
            "qaPairs": position_pairs,
            "technicalDepth": position_analysis.get("technical_depth"),
            "accuracy": position_analysis.get("accuracy"),
            "relevance": position_analysis.get("relevance"),
            "strengths": position_analysis.get("strengths") or [],
            "improvements": position_analysis.get("improvements") or [],
            "summary": position_analysis.get("summary") or "",
            "scorePercentage": position_score_raw,
        },
        "codingAnalysis": coding_analysis,
        "aptitudeAnalysis": aptitude_analysis,
        "overallSummary": overall_summary,
        "softSkills": soft_skills,
        "scores": {
            "generalScore": round(general_score_raw, 1),
            "positionScore": round(position_score_raw, 1),
            "codingScore": round(coding_score_data["percentage"], 1),
            "aptitudeScore": round(aptitude_score_data["percentage"], 1),
            "generalMarks": gen_marks,
            "positionMarks": pos_marks,
            "codingMarks": cod_marks,
            "aptitudeMarks": apt_marks,
            "totalMarksObtained": round(total_marks_obtained, 1),
            "totalMarksMax": total_max,
            "overallPercentage": overall_pct,
            "recommendation": recommendation,
            "softSkillsFluency": general_analysis.get("fluency"),
            "softSkillsGrammar": general_analysis.get("grammar"),
            "softSkillsConfidence": general_analysis.get("confidence"),
            "softSkillsClarity": general_analysis.get("clarity"),
            "recommendationStatus": recommendation,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Build and Save Report Document
# ──────────────────────────────────────────────────────────────────────────────
def _build_report_doc(
    position_id: str, candidate_id: str, client_id: str, tenant_id: str,
    candidate_profile: dict,
    position_details: dict, jd_text: str, resume_text: str,
    assessment_summary: dict, round_marks: dict, analysis: dict,
    r1r2_data: dict, r3_data: dict, r4_data: list,
    screenshot_assets: dict | None = None,
) -> dict:
    scores = analysis.get("scores", {})
    general_round = analysis.get("generalAnalysis", {})
    position_round = analysis.get("positionAnalysis", {})
    coding_round = analysis.get("codingAnalysis", {})
    aptitude_round = analysis.get("aptitudeAnalysis", {})
    overall_summary = analysis.get("overallSummary", {})
    resume_summary_points = analysis.get("resumeSummaryPoints") or []
    pos_title = position_details.get("title") or "the role"

    def _to_list(val):
        return val if isinstance(val, list) else ([] if val is None else [val])

    strengths = _to_list(overall_summary.get("strengths")) + _to_list(general_round.get("strengths")) + _to_list(position_round.get("strengths"))
    areas = _to_list(overall_summary.get("areasForImprovement")) + _to_list(general_round.get("improvements")) + _to_list(position_round.get("improvements"))
    # Deduplicate while preserving order
    strengths = [s for i, s in enumerate(strengths) if s and s not in strengths[:i]]
    areas = [s for i, s in enumerate(areas) if s and s not in areas[:i]]

    # Build general remarks - must never be empty
    general_remarks_parts = []
    if overall_summary.get("executiveSummary"):
        general_remarks_parts.append(overall_summary["executiveSummary"])
    if overall_summary.get("resumeMatchAnalysis"):
        general_remarks_parts.append(overall_summary["resumeMatchAnalysis"])
    if overall_summary.get("skillsGapAnalysis"):
        general_remarks_parts.append(overall_summary["skillsGapAnalysis"])
    if general_round.get("summary"):
        general_remarks_parts.append(general_round["summary"])
    if position_round.get("summary"):
        general_remarks_parts.append(position_round["summary"])
    # Deduplicate
    general_remarks = list(dict.fromkeys(r for r in general_remarks_parts if r))

    # Build conclusion - must never be empty
    conclusion_text = overall_summary.get("finalRecommendation") or ""
    if not conclusion_text:
        rec = scores.get("recommendationStatus") or scores.get("recommendation", "NOT_RECOMMENDED")
        overall_pct_val = scores.get("overallPercentage", 0)
        conclusion_text = f"{rec.replace('_', ' ')} — Overall score: {overall_pct_val}%."
    conclusion = [conclusion_text]

    general_pairs = _to_list(general_round.get("qaPairs"))
    position_pairs = _to_list(position_round.get("qaPairs"))

    screening_questions = {
        "generalQuestions": [
            {
                "questionNumber": idx + 1,
                "question": q.get("question") or "",
                "candidateAnswer": q.get("answer") or "",
                "aiComments": q.get("aiComment") or "",
                "aiRatings": None,
                "contentType": "general",
            }
            for idx, q in enumerate(general_pairs)
        ],
        "specificQuestions": [
            {
                "questionNumber": idx + 1,
                "question": q.get("question") or "",
                "candidateAnswer": q.get("answer") or "",
                "aiComments": q.get("aiComment") or "",
                "aiRatings": None,
                "contentType": "technical",
            }
            for idx, q in enumerate(position_pairs)
        ],
    }

    # Merge AI analysis results into raw coding question sets (by questionTitle)
    coding_analyzed_map = {}
    for cq in (coding_round.get("questions") or []):
        title = cq.get("questionTitle", "")
        if title:
            coding_analyzed_map[title] = cq

    coding_question_sets = r3_data.get("codingQuestionSets") if isinstance(r3_data, dict) else []
    if coding_question_sets and coding_analyzed_map:
        for qs in (coding_question_sets or []):
            for q in (qs.get("questions") or []):
                ai_q = coding_analyzed_map.get(q.get("questionTitle", ""))
                if ai_q:
                    q["aiAnalysisPoints"] = ai_q.get("aiAnalysisPoints") or []
                    q["compositeScore"] = ai_q.get("compositeScore")
                    q["codeQualityScore"] = ai_q.get("codeQualityScore")
                    q["testCasePercentage"] = ai_q.get("testCasePercentage")
                    q["executionError"] = ai_q.get("executionError") or ""

    aptitude_questions = []
    for q in (r4_data or []):
        aptitude_questions.append({
            "question": q.get("question") or q.get("questionText") or q.get("text") or "",
            "options": q.get("options") or [],
            "candidateAnswer": q.get("answerText") or q.get("answer") or "",
            "correctAnswer": q.get("correctAnswer") or "",
            "isCorrect": bool(q.get("correctAnswer")) and (str(q.get("answerText") or q.get("answer") or "").strip() == str(q.get("correctAnswer") or "").strip()),
        })

    screenshot_assets = screenshot_assets or {}
    profile_picture = (
        candidate_profile.get("profilePicture")
        or screenshot_assets.get("profilePicture")
        or ""
    )

    unified_report = {
        "positionId": position_id,
        "candidateId": candidate_id,
        "candidateCode": candidate_profile.get("candidateCode"),
        "candidateName": candidate_profile.get("candidateName"),
        "email": candidate_profile.get("email"),
        "phone": candidate_profile.get("phone"),
        "profilePicture": profile_picture,
        "companyName": candidate_profile.get("companyName"),
        "positionName": candidate_profile.get("positionName") or position_details.get("title"),
        "jobTitle": position_details.get("title"),
        "positionCode": candidate_profile.get("positionCode"),
        "domainType": position_details.get("domainType"),
        "minimumExperience": position_details.get("minExperience"),
        "maximumExperience": position_details.get("maxExperience"),
        "questionSetCode": candidate_profile.get("questionSetCode"),
        "questionSetDuration": candidate_profile.get("questionSetDuration"),
        "interviewDate": candidate_profile.get("interviewDate") or assessment_summary.get("assessmentStartTime"),
        "interviewDuration": assessment_summary.get("assessmentTimeTaken") or "",
        "recommendationStatus": scores.get("recommendationStatus") or scores.get("recommendation", "NOT_RECOMMENDED"),
        "resumeSummary": "\n".join(f"• {p}" for p in resume_summary_points) if resume_summary_points else (resume_text[:1500] if resume_text else ""),
        "resumeSummaryPoints": resume_summary_points,
        "resumeScore": candidate_profile.get("resumeScore"),
        "overallMarks": scores.get("overallPercentage", 0),
        "rank": 1,
        "suspiciousActivity": "None detected",
        "status": True,
        "generalScreeningScore": scores.get("generalMarks", 0),
        "generalScreeningStatus": bool(round_marks.get("general", 0)),
        "codingScreeningScore": scores.get("codingMarks", 0),
        "codingScreeningStatus": bool(round_marks.get("coding", 0)),
        "aptitudeScreeningScore": scores.get("aptitudeMarks", 0),
        "aptitudeScreeningStatus": bool(round_marks.get("aptitude", 0)),
        "positionSpecificScreeningScore": scores.get("positionMarks", 0),
        "positionSpecificScreeningStatus": bool(round_marks.get("position", 0)),
        "aiGeneratedPercentage": 40,
        "aiGeneratedRefinedPercentage": None,
        "humanWrittenRefinedPercentage": None,
        "humanWrittenPercentage": 60,
        "strengths": strengths,
        "areasForImprovement": areas,
        "generalRemarks": general_remarks,
        "conclusion": conclusion,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "reportVersion": "1.0",
        "platform": "KareerGrowth AI",
        "assessmentType": "Comprehensive Assessment",
        "isActive": True,
        "createdBy": candidate_profile.get("createdBy"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "softSkills": {
            "skills": [
                {"skill": "Fluency", "score": analysis.get("softSkills", {}).get("fluency") or general_round.get("fluency") or 0, "maxMarks": 10},
                {"skill": "Grammar", "score": analysis.get("softSkills", {}).get("grammar") or general_round.get("grammar") or 0, "maxMarks": 10},
                {"skill": "Confidence", "score": analysis.get("softSkills", {}).get("confidence") or general_round.get("confidence") or 0, "maxMarks": 10},
                {"skill": "Clarity", "score": analysis.get("softSkills", {}).get("clarity") or general_round.get("clarity") or 0, "maxMarks": 10},
            ],
            "roundScores": {
                "general": {"assigned": bool(round_marks.get("general", 0)), "totalMarks": round_marks.get("general", 0)},
                "position": {"assigned": bool(round_marks.get("position", 0)), "totalMarks": round_marks.get("position", 0)},
                "aptitude": {"assigned": bool(round_marks.get("aptitude", 0)), "totalMarks": round_marks.get("aptitude", 0)},
                "coding": {"assigned": bool(round_marks.get("coding", 0)), "totalMarks": round_marks.get("coding", 0)},
            },
        },
        "mandatorySkills": [
            {
                "skill": skill,
                "aiRating": round(float(scores.get("positionScore") or 0) / 10, 1),
                "total": 10,
                "aiComment": (
                    f"Based on resume and JD analysis, candidate demonstrates "
                    f"{'strong' if (scores.get('positionScore') or 0) >= 70 else 'moderate' if (scores.get('positionScore') or 0) >= 40 else 'limited'} "
                    f"proficiency in {skill}. "
                    f"{(position_round.get('summary') or '')[:120]}"
                ).strip(),
            }
            for skill in (position_details.get("mandatorySkills") or [])
        ],
        "aiRatings": {
            "communication": round(float(scores.get("generalScore") or 0), 1),
            "technicalKnowledge": round(float(scores.get("positionScore") or 0), 1),
            "problemSolving": round(float(scores.get("codingScore") or 0), 1),
            "aptitude": round(float(scores.get("aptitudeScore") or 0), 1),
            "overallFit": round(float(scores.get("overallPercentage") or 0), 1),
        },
        "sectionWiseAnalysis": {
            "general": {
                "strengths": _to_list(general_round.get("strengths")) or [
                    "Communication profile assessed from available data",
                    "Written expression reviewed from resume"
                ],
                "concerns": _to_list(general_round.get("improvements")) or [
                    "Live interview session data not captured",
                    "Verbal communication skills to be verified in person"
                ],
                "overallAssessment": general_round.get("summary") or (
                    f"General communication screening completed for {pos_title} candidate. "
                    "Assessment derived from available profile data."
                ),
                "examples": [q.get("question") for q in general_pairs[:3] if q.get("question")],
            },
            "position": {
                "strengths": _to_list(position_round.get("strengths")) or [
                    f"Resume demonstrates relevant skills for {pos_title}",
                    "Educational background aligns with role requirements"
                ],
                "concerns": _to_list(position_round.get("improvements")) or [
                    "Live technical Q&A data not available",
                    "Hands-on skill proficiency to be verified"
                ],
                "overallAssessment": position_round.get("summary") or (
                    f"Technical fit assessment for {pos_title} based on resume-JD analysis. "
                    "Live interview responses were not available."
                ),
                "examples": [q.get("question") for q in position_pairs[:3] if q.get("question")],
            },
            "coding": {
                "strengths": coding_round.get("strengths") or (
                    ["Coding assessment not assigned for this position"] if not round_marks.get("coding", 0) else []
                ),
                "concerns": coding_round.get("concerns") or [],
                "overallAssessment": coding_round.get("overallReview") or (
                    "Coding round was not assigned for this position."
                    if not round_marks.get("coding", 0) else
                    "No coding submissions available for review."
                ),
                "examples": [q.get("questionTitle") for s in (coding_question_sets or []) for q in (s.get("questions") or [])][:3],
            },
        },
        "roundTimings": [
            {
                "round": "General Screening",
                "roundKey": "round1",
                "assignedTime": assessment_summary.get("round1GivenTime"),
                "timeTaken": assessment_summary.get("round1TimeTaken"),
                "assigned": bool(round_marks.get("general", 0)),
            },
            {
                "round": "Position-Specific Screening",
                "roundKey": "round2",
                "assignedTime": assessment_summary.get("round2GivenTime"),
                "timeTaken": assessment_summary.get("round2TimeTaken"),
                "assigned": bool(round_marks.get("position", 0)),
            },
            {
                "round": "Coding Challenge",
                "roundKey": "round3",
                "assignedTime": assessment_summary.get("round3GivenTime"),
                "timeTaken": assessment_summary.get("round3TimeTaken"),
                "assigned": bool(round_marks.get("coding", 0)),
            },
            {
                "round": "Aptitude Assessment",
                "roundKey": "round4",
                "assignedTime": assessment_summary.get("round4GivenTime"),
                "timeTaken": assessment_summary.get("round4TimeTaken"),
                "assigned": bool(round_marks.get("aptitude", 0)),
            },
        ],
        "screeningQuestions": screening_questions,
        "codingQuestionSets": coding_question_sets or [],
        "aptitudeAssessment": {
            "questions": aptitude_questions,
            "overallAiReview": aptitude_round.get("overallReview") or "",
        },
        "proctoringScreenshots": {
            "calibration": screenshot_assets.get("calibration") or [],
            "noFace": screenshot_assets.get("noFace") or [],
            "multipleFaces": screenshot_assets.get("multipleFaces") or [],
            "allDirection": screenshot_assets.get("allDirection") or [],
        },
        "screenshots": screenshot_assets.get("all") or [],
    }

    return {
        "positionId": position_id,
        "candidateId": candidate_id,
        "clientId": client_id,
        "tenantId": tenant_id,
        "isGenerated": True,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "positionDetails": position_details,
        "jobDescriptionSummary": jd_text[:1000] if jd_text else "",
        "resumeSummary": "\n".join(f"• {p}" for p in resume_summary_points) if resume_summary_points else (resume_text[:1000] if resume_text else ""),
        "resumeSummaryPoints": resume_summary_points,
        "assessmentSummary": {
            "round1Assigned": assessment_summary.get("round1Assigned"),
            "round2Assigned": assessment_summary.get("round2Assigned"),
            "round3Assigned": assessment_summary.get("round3Assigned"),
            "round4Assigned": assessment_summary.get("round4Assigned"),
            "round1Completed": assessment_summary.get("round1Completed"),
            "round2Completed": assessment_summary.get("round2Completed"),
            "round3Completed": assessment_summary.get("round3Completed"),
            "round4Completed": assessment_summary.get("round4Completed"),
            "assessmentStartTime": assessment_summary.get("assessmentStartTime"),
            "assessmentEndTime": assessment_summary.get("assessmentEndTime"),
        },
        "roundMarks": round_marks,
        "scores": scores,
        "overallPercentage": scores.get("overallPercentage", 0),
        "recommendation": scores.get("recommendationStatus") or scores.get("recommendation", "NOT_RECOMMENDED"),
        "generalRound": analysis.get("generalAnalysis", {}),
        "positionRound": analysis.get("positionAnalysis", {}),
        "codingRound": analysis.get("codingAnalysis", {}),
        "aptitudeRound": analysis.get("aptitudeAnalysis", {}),
        "softSkills": analysis.get("softSkills", {}),
        "overallSummary": analysis.get("overallSummary", {}),
        "rawInputs": {
            "candidateProfile": candidate_profile or {},
            "round1Round2": r1r2_data or {},
            "round3Coding": r3_data or {},
            "round4Aptitude": r4_data or [],
        },
        "unifiedReport": unified_report,
    }


async def _inject_dynamic_screenshot_assets(report_doc: dict, fallback_client_id: str = "") -> dict:
    """Fetch latest screenshot URLs from GCS and merge into report response payload."""
    if not isinstance(report_doc, dict):
        return report_doc

    client_id = str(report_doc.get("clientId") or fallback_client_id or "").strip()
    position_id = str(report_doc.get("positionId") or "").strip()
    candidate_id = str(report_doc.get("candidateId") or "").strip()

    if not (client_id and position_id and candidate_id and config.GCS_BUCKET):
        return report_doc

    try:
        assets = await asyncio.to_thread(
            list_report_screenshot_urls,
            client_id,
            position_id,
            candidate_id,
            4,
        )
    except Exception as e:
        logger.warning("[ReportGen] dynamic screenshot enrichment failed: %s", e)
        return report_doc

    if not isinstance(assets, dict):
        return report_doc

    updated = dict(report_doc)
    unified = dict((updated.get("unifiedReport") or {}) if isinstance(updated.get("unifiedReport"), dict) else {})
    if not unified:
        unified = dict(updated)

    if assets.get("profilePicture"):
        unified["profilePicture"] = assets.get("profilePicture")

    unified["proctoringScreenshots"] = {
        "calibration": assets.get("calibration") or [],
        "noFace": assets.get("noFace") or [],
        "multipleFaces": assets.get("multipleFaces") or [],
        "allDirection": assets.get("allDirection") or [],
    }
    unified["screenshots"] = assets.get("all") or []

    updated["unifiedReport"] = unified
    return updated


def _build_completed_response(report_doc: dict, is_existing: bool) -> dict:
    unified = (report_doc or {}).get("unifiedReport") or (report_doc or {})
    summary = (report_doc or {}).get("assessmentSummary")
    if summary is None:
        summary = unified.get("assessmentSummary") if isinstance(unified, dict) else {}
    data = {
        "requestId": f"{(report_doc or {}).get('positionId', '')}_{(report_doc or {}).get('candidateId', '')}",
        "status": "completed",
        "isExisting": bool(is_existing),
        **unified,
        "assessmentSummary": summary or {},
    }
    return {
        "success": True,
        "message": "Assessment report generation completed successfully",
        "data": data,
    }


async def _save_report(position_id: str, candidate_id: str, doc: dict):
    """Upsert report document through CandidateBackend internal API."""
    base = _candidate_backend_url()
    if not base:
        logger.warning("[ReportGen] CANDIDATE_BACKEND_URL not configured — cannot save report")
        return

    payload = dict(doc or {})
    payload["positionId"] = position_id
    payload["candidateId"] = candidate_id

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{base}/internal/streaming/report-document",
                headers=_candidate_internal_headers(),
                json=payload,
            )
        if resp.status_code >= 300:
            logger.warning("[ReportGen] CandidateBackend report save failed: %s %s", resp.status_code, resp.text[:300])
        else:
            logger.info("[ReportGen] Saved report via CandidateBackend for %s/%s", position_id, candidate_id)
    except Exception as e:
        logger.error("[ReportGen] CandidateBackend report save error: %s", e)


async def _set_generation_state(
    position_id: str,
    candidate_id: str,
    client_id: str,
    tenant_id: str,
    status: str,
    error_message: str = "",
):
    """Persist generation status through CandidateBackend internal API."""
    if not position_id or not candidate_id:
        return

    base = _candidate_backend_url()
    if not base:
        return

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "positionId": position_id,
        "candidateId": candidate_id,
        "clientId": client_id or "",
        "tenantId": tenant_id or "",
        "generationStatus": status,
        "updatedAt": now,
    }

    if status == "completed":
        payload["isGenerated"] = True
        payload["generatedAt"] = now
        payload["generationError"] = ""
    elif status in {"queued", "in_progress"}:
        payload["isGenerated"] = False
    elif status == "failed":
        payload["isGenerated"] = False
        payload["generationError"] = (error_message or "")[:1000]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.patch(
                f"{base}/internal/streaming/report-document",
                headers=_candidate_internal_headers(),
                json=payload,
            )
    except Exception as e:
        logger.warning("[ReportGen] Failed to persist generation status (%s): %s", status, e)


async def _get_existing_report(position_id: str, candidate_id: str, client_id: str = "") -> Optional[dict]:
    """Fetch existing report through CandidateBackend internal API."""
    base = _candidate_backend_url()
    if not base:
        return None

    params = {
        "positionId": position_id,
        "candidateId": candidate_id,
    }
    if client_id:
        params["clientId"] = client_id

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{base}/internal/streaming/report-document",
                headers=_candidate_internal_headers(),
                params=params,
            )
        if resp.status_code == 200:
            body = resp.json() or {}
            return body.get("data") if isinstance(body, dict) else None
        if resp.status_code != 404:
            logger.warning("[ReportGen] CandidateBackend report fetch returned %s", resp.status_code)
        return None
    except Exception as e:
        logger.warning("[ReportGen] CandidateBackend report fetch failed: %s", e)
        return None


async def _mark_report_generated(
    admin_url: str, headers: dict,
    position_id: str, candidate_id: str, tenant_id: str, scores: dict,
):
    """Call AdminBackend to update MySQL assessment_report_generation + interview_evaluations."""
    payload = {
        "positionId": position_id,
        "candidateId": candidate_id,
        "tenantId": tenant_id,
        "scores": {
            "totalScore": scores.get("overallPercentage", 0),
            "generalScore": scores.get("generalScore"),
            "positionScore": scores.get("positionScore"),
            "codingScore": scores.get("codingScore"),
            "aptitudeScore": scores.get("aptitudeScore"),
            "recommendationStatus": scores.get("recommendationStatus") or scores.get("recommendation", "NOT_RECOMMENDED"),
            "softSkillsFluency": scores.get("softSkillsFluency"),
            "softSkillsGrammar": scores.get("softSkillsGrammar"),
            "softSkillsConfidence": scores.get("softSkillsConfidence"),
            "softSkillsClarity": scores.get("softSkillsClarity"),
        },
    }
    
    recommendation_status = payload["scores"].get("recommendationStatus", "NOT_RECOMMENDED")
    logger.info(
        "[ReportGen] Sending mark-report-generated to AdminBackend: "
        "recommendation=%s, totalScore=%s, tenantId=%s",
        recommendation_status, payload["scores"].get("totalScore"), tenant_id
    )
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{admin_url}/internal/mark-report-generated", json=payload, headers=headers)
            if resp.status_code == 200:
                logger.info(
                    "[ReportGen] ✓ MySQL tables updated successfully (recommendation=%s for candidate=%s)",
                    recommendation_status, candidate_id[:8]
                )
            else:
                logger.warning(
                    "[ReportGen] mark-report-generated returned %s (recommendation=%s, candidate=%s)",
                    resp.status_code, recommendation_status, candidate_id[:8]
                )
    except Exception as e:
        logger.warning(
            "[ReportGen] mark-report-generated call failed: %s (recommendation=%s, candidate=%s)",
            e, recommendation_status, candidate_id[:8]
        )
