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
from copy import deepcopy
from typing import Any, Dict

import httpx
import websockets
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

import config
from screening_utils import build_flat_qa_list
from screening_utils import questions_for_round
from coding_question import GenerateQuestionRequest, generate_coding_question
from aptitude_generator import generate_aptitude_questions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin-report", tags=["Admin Report API"])


def _unwrap_route_payload(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("data"), (dict, list)):
        return value.get("data")
    return value


def _summary_flag(assessment_summary: Dict[str, Any], key: str, fallback: bool = False) -> bool:
    if not isinstance(assessment_summary, dict):
        return fallback
    snake_key = []
    for ch in key:
        if ch.isupper():
            snake_key.append("_")
            snake_key.append(ch.lower())
        else:
            snake_key.append(ch)
    normalized_key = "".join(snake_key)
    if normalized_key.startswith("_"):
        normalized_key = normalized_key[1:]
    value = assessment_summary.get(key)
    if value is None:
        value = assessment_summary.get(normalized_key)
    if value is None:
        return fallback
    return bool(value)


def _is_empty_round_list(value: Any) -> bool:
    if not isinstance(value, list):
        return True
    return len(value) == 0


def _extract_screening_questions(conversational_payload: Dict[str, Any], analysis_pairs: list, category_key: str, content_type: str) -> list:
    source = _unwrap_route_payload(conversational_payload) or {}
    round_num = 1 if category_key == "generalQuestion" else 2 if category_key == "positionSpecificQuestion" else None
    categories = source.get("categories") or {}
    ai_map = {}
    for idx, pair in enumerate(analysis_pairs or []):
        question_text = (pair or {}).get("question") or ""
        if question_text:
            ai_map[question_text.strip()] = (pair or {}).get("aiComment") or ""
        else:
            ai_map[f"__index__:{idx}"] = (pair or {}).get("aiComment") or ""

    if round_num and isinstance(categories, dict):
        flattened = []
        for idx, (_conv_key, _pair_idx, question_text, answer_text) in enumerate(build_flat_qa_list(source, round_num)):
            if not question_text and not answer_text:
                continue
            flattened.append(
                {
                    "questionNumber": len(flattened) + 1,
                    "question": question_text,
                    "candidateAnswer": answer_text,
                    "aiComments": ai_map.get(question_text.strip()) or ai_map.get(f"__index__:{idx}") or "",
                    "aiRatings": None,
                    "contentType": content_type,
                }
            )
        if flattened:
            return flattened

    conversation_sets = ((categories.get(category_key) or {}).get("conversationSets") or {})

    flattened = []
    for pairs in conversation_sets.values():
        for pair in pairs or []:
            question_text = (pair or {}).get("question") or ""
            answer_text = (pair or {}).get("answer") or ""
            if not question_text and not answer_text:
                continue
            flattened.append(
                {
                    "questionNumber": len(flattened) + 1,
                    "question": question_text,
                    "candidateAnswer": answer_text,
                    "aiComments": ai_map.get(question_text.strip()) or ai_map.get(f"__index__:{len(flattened)}") or "",
                    "aiRatings": None,
                    "contentType": content_type,
                }
            )
    return flattened


def _pairs_to_screening_questions(analysis_pairs: list, content_type: str) -> list:
    result = []
    for idx, pair in enumerate(analysis_pairs or []):
        question_text = (pair or {}).get("question") or ""
        answer_text = (pair or {}).get("answer") or ""
        if not question_text and not answer_text:
            continue
        result.append(
            {
                "questionNumber": len(result) + 1,
                "question": question_text,
                "candidateAnswer": answer_text,
                "aiComments": (pair or {}).get("aiComment") or "",
                "aiRatings": None,
                "contentType": content_type,
            }
        )
    return result


def _merge_coding_analysis(coding_sets: list, coding_analysis: Dict[str, Any]) -> list:
    if not coding_sets:
        return []
    analyzed_map = {}
    for question in (coding_analysis or {}).get("questions") or []:
        title = question.get("questionTitle") or ""
        if title:
            analyzed_map[title] = question

    merged_sets = deepcopy(coding_sets)
    for question_set in merged_sets:
        for question in question_set.get("questions") or []:
            analyzed = analyzed_map.get(question.get("questionTitle") or "")
            if not analyzed:
                continue
            question["aiAnalysisPoints"] = analyzed.get("aiAnalysisPoints") or question.get("aiAnalysisPoints") or []
            question["compositeScore"] = analyzed.get("compositeScore")
            question["codeQualityScore"] = analyzed.get("codeQualityScore")
            question["testCasePercentage"] = analyzed.get("testCasePercentage")
            question["executionError"] = analyzed.get("executionError") or question.get("executionError") or ""
    return merged_sets


def _extract_coding_sets(coding_payload: Any) -> list:
    """Normalize coding payload to codingQuestionSets list across route response shapes."""
    payload = _unwrap_route_payload(coding_payload)

    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    direct_sets = payload.get("codingQuestionSets")
    if isinstance(direct_sets, list):
        return direct_sets

    for key in ("record", "result", "response", "document"):
        nested = payload.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("codingQuestionSets"), list):
            return nested.get("codingQuestionSets") or []

    return []


def _build_coding_sets_from_analysis(coding_analysis: Dict[str, Any]) -> list:
    questions = (coding_analysis or {}).get("questions") or []
    if not questions:
        return []
    normalized_questions = []
    for question in questions:
        normalized_questions.append(
            {
                "questionTitle": question.get("questionTitle") or question.get("title") or "Coding Question",
                "questionDescription": question.get("questionDescription") or question.get("description") or "",
                "submissions": [
                    {
                        "sourceCode": question.get("sourceCode") or "",
                        "programmingLanguage": question.get("programmingLanguage") or question.get("language") or "N/A",
                    }
                ],
                "testCases": question.get("testCases") or [],
                "testCasesPassed": question.get("testCasesPassed") or 0,
                "totalTestCases": question.get("totalTestCases") or 0,
                "executionStatus": question.get("executionStatus") or "SUBMITTED",
                "aiAnalysisPoints": question.get("aiAnalysisPoints") or [],
                "compositeScore": question.get("compositeScore"),
                "codeQualityScore": question.get("codeQualityScore"),
                "testCasePercentage": question.get("testCasePercentage"),
                "executionError": question.get("executionError") or "",
            }
        )
    return [{"id": "fallback-analysis", "questions": normalized_questions, "overallAiReview": (coding_analysis or {}).get("overallReview") or ""}]


def _build_aptitude_questions(aptitude_rows: list) -> list:
    result = []
    for row in aptitude_rows or []:
        candidate_answer = row.get("answerText") or row.get("answer") or ""
        correct_answer = row.get("correctAnswer") or ""
        result.append(
            {
                "question": row.get("question") or row.get("questionText") or row.get("text") or "",
                "options": row.get("options") or [],
                "candidateAnswer": candidate_answer,
                "correctAnswer": correct_answer,
                "isCorrect": bool(correct_answer) and str(candidate_answer).strip() == str(correct_answer).strip(),
            }
        )
    return result


def _build_aptitude_questions_from_analysis(aptitude_analysis: Dict[str, Any]) -> list:
    result = []
    for row in (aptitude_analysis or {}).get("questions") or []:
        result.append(
            {
                "question": row.get("question") or row.get("questionText") or row.get("text") or "",
                "options": row.get("options") or [],
                "candidateAnswer": row.get("candidateAnswer") or row.get("answerText") or "",
                "correctAnswer": row.get("correctAnswer") or "",
                "isCorrect": bool(row.get("isCorrect")),
            }
        )
    return result


def _coalesce_assigned_from_unified(unified: Dict[str, Any], round_key: str, status_key: str, soft_skill_key: str) -> bool:
    if not isinstance(unified, dict):
        return False

    if status_key in unified and unified.get(status_key) is not None:
        return bool(unified.get(status_key))

    timings = unified.get("roundTimings") or []
    for row in timings:
        if (row or {}).get("roundKey") == round_key and (row or {}).get("assigned") is not None:
            return bool((row or {}).get("assigned"))

    round_scores = ((unified.get("softSkills") or {}).get("roundScores") or {})
    if (round_scores.get(soft_skill_key) or {}).get("assigned") is not None:
        return bool((round_scores.get(soft_skill_key) or {}).get("assigned"))

    if round_key == "round1":
        return len(((unified.get("screeningQuestions") or {}).get("generalQuestions") or [])) > 0
    if round_key == "round2":
        return len(((unified.get("screeningQuestions") or {}).get("specificQuestions") or [])) > 0
    if round_key == "round3":
        return len((unified.get("codingQuestionSets") or [])) > 0
    if round_key == "round4":
        return len(((unified.get("aptitudeAssessment") or {}).get("questions") or [])) > 0
    return False


def _hydrate_existing_report_doc(existing: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    if not existing:
        return existing

    hydrated = deepcopy(existing)
    unified = deepcopy((hydrated.get("unifiedReport") or hydrated) or {})
    candidate_profile = payload.get("candidateProfile") or {}
    position_details = payload.get("positionDetails") or {}
    assessment_summary = payload.get("assessmentSummary") or {}
    candidate_answers = payload.get("candidateAnswers") or {}
    round1_assigned = _summary_flag(
        assessment_summary,
        "round1Assigned",
        _coalesce_assigned_from_unified(unified, "round1", "generalScreeningStatus", "general"),
    )
    round2_assigned = _summary_flag(
        assessment_summary,
        "round2Assigned",
        _coalesce_assigned_from_unified(unified, "round2", "positionSpecificScreeningStatus", "position"),
    )
    round3_assigned = _summary_flag(
        assessment_summary,
        "round3Assigned",
        _coalesce_assigned_from_unified(unified, "round3", "codingScreeningStatus", "coding"),
    )
    round4_assigned = _summary_flag(
        assessment_summary,
        "round4Assigned",
        _coalesce_assigned_from_unified(unified, "round4", "aptitudeScreeningStatus", "aptitude"),
    )

    raw_inputs = hydrated.get("rawInputs") or {}
    conversational_payload = candidate_answers.get("round1Round2") or raw_inputs.get("round1Round2") or {}
    coding_payload = _unwrap_route_payload(candidate_answers.get("round3Coding") or raw_inputs.get("round3Coding") or {}) or {}
    aptitude_payload = candidate_answers.get("round4Aptitude") or raw_inputs.get("round4Aptitude") or []

    general_pairs = ((hydrated.get("generalRound") or {}).get("qaPairs") or [])
    position_pairs = ((hydrated.get("positionRound") or {}).get("qaPairs") or [])

    screening_questions = unified.get("screeningQuestions") or {"generalQuestions": [], "specificQuestions": []}
    general_questions = screening_questions.get("generalQuestions") or []
    specific_questions = screening_questions.get("specificQuestions") or []

    if round1_assigned and not general_questions:
        extracted_general = _extract_screening_questions(conversational_payload, general_pairs, "generalQuestion", "general")
        screening_questions["generalQuestions"] = extracted_general or _pairs_to_screening_questions(general_pairs, "general")
    if round2_assigned and not specific_questions:
        extracted_specific = _extract_screening_questions(conversational_payload, position_pairs, "positionSpecificQuestion", "technical")
        screening_questions["specificQuestions"] = extracted_specific or _pairs_to_screening_questions(position_pairs, "technical")
    unified["screeningQuestions"] = screening_questions

    normalized_coding_sets = _extract_coding_sets(coding_payload)
    if (round3_assigned or len(normalized_coding_sets) > 0) and _is_empty_round_list(unified.get("codingQuestionSets")):
        coding_sets = normalized_coding_sets
        merged_coding_sets = _merge_coding_analysis(coding_sets or [], hydrated.get("codingRound") or {})
        unified["codingQuestionSets"] = merged_coding_sets or _build_coding_sets_from_analysis(hydrated.get("codingRound") or {})

    aptitude_assessment = unified.get("aptitudeAssessment") or {"questions": [], "overallAiReview": ""}
    if round4_assigned and _is_empty_round_list(aptitude_assessment.get("questions")):
        built_questions = _build_aptitude_questions(aptitude_payload or [])
        aptitude_assessment["questions"] = built_questions or _build_aptitude_questions_from_analysis(hydrated.get("aptitudeRound") or {})
    if not aptitude_assessment.get("overallAiReview"):
        aptitude_assessment["overallAiReview"] = (hydrated.get("aptitudeRound") or {}).get("overallReview") or ""
    unified["aptitudeAssessment"] = aptitude_assessment

    if not unified.get("candidateCode"):
        unified["candidateCode"] = candidate_profile.get("candidateCode")
    if not unified.get("candidateName"):
        unified["candidateName"] = candidate_profile.get("candidateName")
    if not unified.get("email"):
        unified["email"] = candidate_profile.get("email")
    if not unified.get("phone"):
        unified["phone"] = candidate_profile.get("phone")
    if not unified.get("companyName"):
        unified["companyName"] = candidate_profile.get("companyName")
    if not unified.get("positionName"):
        unified["positionName"] = candidate_profile.get("positionName") or position_details.get("title")
    if not unified.get("jobTitle"):
        unified["jobTitle"] = position_details.get("title")
    if not unified.get("questionSetCode"):
        unified["questionSetCode"] = candidate_profile.get("questionSetCode")
    if not unified.get("questionSetDuration"):
        unified["questionSetDuration"] = candidate_profile.get("questionSetDuration")
    if not unified.get("interviewDate"):
        unified["interviewDate"] = candidate_profile.get("interviewDate") or assessment_summary.get("assessmentStartTime")
    if not unified.get("interviewDuration"):
        unified["interviewDuration"] = assessment_summary.get("assessmentTimeTaken") or ""

    unified["generalScreeningStatus"] = round1_assigned
    unified["positionSpecificScreeningStatus"] = round2_assigned
    unified["codingScreeningStatus"] = round3_assigned
    unified["aptitudeScreeningStatus"] = round4_assigned

    soft_skills = deepcopy(unified.get("softSkills") or {})
    round_scores = deepcopy(soft_skills.get("roundScores") or {})
    round_scores["general"] = {
        **(round_scores.get("general") or {}),
        "assigned": round1_assigned,
    }
    round_scores["position"] = {
        **(round_scores.get("position") or {}),
        "assigned": round2_assigned,
    }
    round_scores["aptitude"] = {
        **(round_scores.get("aptitude") or {}),
        "assigned": round4_assigned,
    }
    round_scores["coding"] = {
        **(round_scores.get("coding") or {}),
        "assigned": round3_assigned,
    }
    soft_skills["roundScores"] = round_scores
    unified["softSkills"] = soft_skills

    unified["roundTimings"] = [
        {
            "round": "General Screening",
            "roundKey": "round1",
            "assignedTime": assessment_summary.get("round1GivenTime"),
            "timeTaken": assessment_summary.get("round1TimeTaken"),
            "assigned": round1_assigned,
        },
        {
            "round": "Position-Specific Screening",
            "roundKey": "round2",
            "assignedTime": assessment_summary.get("round2GivenTime"),
            "timeTaken": assessment_summary.get("round2TimeTaken"),
            "assigned": round2_assigned,
        },
        {
            "round": "Coding Challenge",
            "roundKey": "round3",
            "assignedTime": assessment_summary.get("round3GivenTime"),
            "timeTaken": assessment_summary.get("round3TimeTaken"),
            "assigned": round3_assigned,
        },
        {
            "round": "Aptitude Assessment",
            "roundKey": "round4",
            "assignedTime": assessment_summary.get("round4GivenTime"),
            "timeTaken": assessment_summary.get("round4TimeTaken"),
            "assigned": round4_assigned,
        },
    ]

    hydrated["rawInputs"] = {
        **raw_inputs,
        "round1Round2": conversational_payload or raw_inputs.get("round1Round2") or {},
        "round3Coding": coding_payload or raw_inputs.get("round3Coding") or {},
        "round4Aptitude": aptitude_payload or raw_inputs.get("round4Aptitude") or [],
    }
    hydrated["assessmentSummary"] = {
        **(hydrated.get("assessmentSummary") or {}),
        **assessment_summary,
    }
    hydrated["unifiedReport"] = unified
    return hydrated


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
    tenant_id: str = "",
    question_set_id: str = "",
) -> dict:
    headers = {"X-Tenant-Id": tenant_id} if tenant_id else {}
    try:
        params = {"candidateId": candidate_id, "positionId": position_id}
        resp = await client.get(
            f"{cand_url}/candidate/interview-responses",
            params=params,
            headers=headers,
        )
        if resp.status_code == 200:
            return _unwrap_route_payload(resp.json()) or {}

        params = {}
        if question_set_id:
            params["questionSetId"] = question_set_id
        resp = await client.get(
            f"{cand_url}/candidate-interviews/candidate/{candidate_id}/position/{position_id}",
            params=params,
            headers=headers,
        )
        if resp.status_code == 200:
            return _unwrap_route_payload(resp.json()) or {}

        # If questionSetId narrowing returned nothing, retry without it to catch any saved document
        if resp.status_code == 404 and question_set_id:
            logger.info(
                "[AdminReportAPI] conversational data 404 with questionSetId=%s, retrying without filter",
                question_set_id,
            )
            resp2 = await client.get(
                f"{cand_url}/candidate-interviews/candidate/{candidate_id}/position/{position_id}",
                headers=headers,
            )
            if resp2.status_code == 200:
                return _unwrap_route_payload(resp2.json()) or {}
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


def _build_conversation_sets_from_ws_questions(questions: list) -> Dict[str, list]:
    conversation_sets: Dict[str, list] = {}
    for idx, q in enumerate(questions or []):
        key = f"conversationQuestion{idx + 1}"
        if isinstance(q, dict):
            conversation_sets[key] = [
                {
                    "question": (q.get("question") or "").strip(),
                    "answer": (q.get("answer") or "").strip(),
                    "answerTime": q.get("timeToAnswer") if q.get("timeToAnswer") is not None else q.get("answerTime"),
                    "prepareTime": q.get("timeToPrepare") if q.get("timeToPrepare") is not None else q.get("prepareTime"),
                }
            ]
        else:
            conversation_sets[key] = [{"question": str(q or ""), "answer": "", "answerTime": None, "prepareTime": None}]
    return conversation_sets


async def _recv_round_questions(ws, round_num: int, timeout_sec: float = 20.0) -> list:
    await ws.send(json.dumps({"type": "get_round_questions", "round": round_num}))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    while loop.time() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - loop.time()))
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if (msg or {}).get("type") == "round_questions" and str((msg or {}).get("round")) == str(round_num):
            return (msg or {}).get("questions") or []
    return []


async def _fetch_conversational_data_via_ws_test(
    candidate_id: str,
    position_id: str,
    client_id: str,
    tenant_id: str,
    question_set_id: str,
    assessment_summary_id: str,
) -> dict:
    try:
        host = "localhost"
        port = getattr(config, "PORT", 9000)
        ws_url = f"ws://{host}:{port}/ws/test"
        init_payload = {
            "type": "init",
            "client_id": client_id,
            "position_id": position_id,
            "candidate_id": candidate_id,
            "question_set_id": question_set_id or "",
            "assessment_summary_id": assessment_summary_id or "",
            "tenant_id": tenant_id or "",
            "is_conversational": True,
            "cross_question_count_general": 2,
            "cross_question_count_position": 2,
        }
        async with websockets.connect(ws_url, open_timeout=10, close_timeout=5) as ws:
            await ws.send(json.dumps(init_payload))
            # Wait for init acknowledgement; ignore unrelated frames.
            for _ in range(5):
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if (msg or {}).get("type") in {"init_ok", "error"}:
                    break

            round1_questions = await _recv_round_questions(ws, 1)
            round2_questions = await _recv_round_questions(ws, 2)

            if not round1_questions and not round2_questions:
                return {}

            return {
                "categories": {
                    "generalQuestion": {
                        "conversationSets": _build_conversation_sets_from_ws_questions(round1_questions)
                    },
                    "positionSpecificQuestion": {
                        "conversationSets": _build_conversation_sets_from_ws_questions(round2_questions)
                    },
                }
            }
    except Exception as e:
        logger.warning("[AdminReportAPI] fetch conversational data via ws/test failed: %s", e)
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


def _normalize_difficulty(value: Any) -> str:
    raw = str(value or "Easy").strip().lower()
    if raw == "hard":
        return "Hard"
    if raw == "medium":
        return "Medium"
    return "Easy"


def _parse_duration_minutes(value: Any, fallback: int = 30) -> int:
    if value is None:
        return fallback
    try:
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else fallback
    except Exception:
        return fallback


async def _fetch_question_sections_data(
    client: httpx.AsyncClient,
    admin_url: str,
    headers: Dict[str, str],
    question_set_id: str,
    tenant_id: str,
) -> Dict[str, Any]:
    if not question_set_id:
        return {}
    try:
        hdrs = {**headers, **({"X-Tenant-Id": tenant_id} if tenant_id else {})}
        resp = await client.get(
            f"{admin_url}/internal/question-sections/question-set/{question_set_id}",
            headers=hdrs,
        )
        if resp.status_code != 200:
            logger.warning("[AdminReportAPI] question-sections fetch failed %d: %s", resp.status_code, resp.text[:200])
            return {}
        body = resp.json() or {}
        data = body.get("data") or {}
        if isinstance(data, list):
            return data[0] if data else {}
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("[AdminReportAPI] question-sections fetch exception: %s", e)
    return {}


async def _seed_coding_if_missing(
    client: httpx.AsyncClient,
    cand_url: str,
    candidate_id: str,
    position_id: str,
    question_set_id: str,
    sections_data: Dict[str, Any],
) -> bool:
    coding_blocks = questions_for_round(sections_data, 3) or []
    if not coding_blocks:
        coding_blocks = [{"language": "JavaScript", "difficulty": "Easy", "duration": 30, "questionSource": "Coding Library"}]

    placeholder_questions = []
    for i, cfg in enumerate(coding_blocks):
        placeholder_questions.append({
            "questionTitle": cfg.get("question") or f"Problem {i + 1}",
            "questionDescription": "",
            "timeLimit": _parse_duration_minutes(cfg.get("duration"), 30),
            "tags": [cfg.get("language") or "JavaScript", _normalize_difficulty(cfg.get("difficulty")).lower()],
            "constraints": "",
            "followUpQuestions": [],
            "sampleInputAndOutput": {"input": "", "output": "", "explanation": ""},
            "testCases": [],
            "totalTestCases": 0,
            "submissions": [],
        })

    response_id = ""
    try:
        upsert_resp = await client.post(
            f"{cand_url}/candidate-coding-responses/upsert",
            json={
                "candidateId": candidate_id,
                "positionId": position_id,
                "questionSetId": question_set_id or "",
                "totalQuestions": len(coding_blocks),
                "codingQuestionSets": [{"questions": placeholder_questions}],
            },
        )
        upsert_body = upsert_resp.json() if upsert_resp.headers.get("content-type", "").startswith("application/json") else {}
        response_id = (
            upsert_body.get("responseId")
            or upsert_body.get("id")
            or (upsert_body.get("record") or {}).get("id")
            or ""
        )
    except Exception as e:
        logger.warning("[AdminReportAPI] coding upsert failed during auto-seed: %s", e)
        return False

    if not response_id:
        logger.warning("[AdminReportAPI] coding auto-seed failed: no responseId from upsert")
        return False

    seeded_count = 0
    for idx, cfg in enumerate(coding_blocks):
        try:
            gen_req = GenerateQuestionRequest(
                programmingLanguage=cfg.get("language") or "JavaScript",
                difficultyLevel=_normalize_difficulty(cfg.get("difficulty")),
                questionSource=cfg.get("questionSource") or "Coding Library",
                topicTags=None,
                questionIndex=idx,
            )
            generated = await generate_coding_question(gen_req)
            if not isinstance(generated, dict):
                continue

            payload = {
                "questionTitle": generated.get("title") or f"Problem {idx + 1}",
                "questionDescription": generated.get("description") or "",
                "timeLimit": _parse_duration_minutes(cfg.get("duration"), 30),
                "tags": [cfg.get("language") or generated.get("language") or "JavaScript", _normalize_difficulty(generated.get("difficulty") or cfg.get("difficulty")).lower()],
                "constraints": "\n".join(generated.get("constraints") or []) if isinstance(generated.get("constraints"), list) else (generated.get("constraints") or ""),
                "testCases": [
                    {
                        "testCaseId": tc.get("testCaseId") or f"tc_{idx}_{ti}",
                        "input": tc.get("input") or "",
                        "expectedOutput": tc.get("output") or tc.get("expectedOutput") or "",
                        "locked": bool(tc.get("locked")),
                    }
                    for ti, tc in enumerate(generated.get("testCases") or [])
                    if isinstance(tc, dict)
                ],
                "totalTestCases": len(generated.get("testCases") or []),
                "followUpQuestions": [],
                "sampleInputAndOutput": (generated.get("examples") or [{}])[0] if isinstance(generated.get("examples"), list) and generated.get("examples") else {"input": "", "output": "", "explanation": ""},
                "language": cfg.get("language") or generated.get("language") or "JavaScript",
                "difficulty": _normalize_difficulty(generated.get("difficulty") or cfg.get("difficulty")),
                "functionSignature": generated.get("functionSignature") or "",
                "inputFormat": generated.get("inputFormat") or "",
                "outputFormat": generated.get("outputFormat") or "",
                "constraintsList": generated.get("constraints") if isinstance(generated.get("constraints"), list) else [],
                "examples": generated.get("examples") if isinstance(generated.get("examples"), list) else [],
                "starterCode": generated.get("starterCode") or "",
            }

            save_resp = await client.put(
                f"{cand_url}/candidate-coding-responses/{response_id}/set/0/question/{idx}/content",
                json=payload,
            )
            if 200 <= save_resp.status_code < 300:
                seeded_count += 1
        except Exception as e:
            logger.warning("[AdminReportAPI] coding question auto-seed failed for index %d: %s", idx, e)

    logger.info("[AdminReportAPI] coding auto-seed completed: %d/%d", seeded_count, len(coding_blocks))
    return seeded_count > 0


async def _seed_aptitude_if_missing(
    client: httpx.AsyncClient,
    cand_url: str,
    candidate_id: str,
    position_id: str,
    question_set_id: str,
    sections_data: Dict[str, Any],
) -> bool:
    topic_blocks = questions_for_round(sections_data, 4) or []
    if not topic_blocks:
        topic_blocks = [{"question": "General Aptitude", "topics": ["General Aptitude"], "count": 5, "difficulty": "MEDIUM"}]

    try:
        generated = await generate_aptitude_questions(topic_blocks)
    except Exception as e:
        logger.warning("[AdminReportAPI] aptitude auto-generate failed: %s", e)
        return False

    if not generated:
        return False

    try:
        resp = await client.post(
            f"{cand_url}/candidate-aptitude-responses/upsert",
            json={
                "candidateId": candidate_id,
                "positionId": position_id,
                "questionSetId": question_set_id or "",
                "totalQuestions": len(generated),
                "generatedQuestions": generated,
            },
        )
        if 200 <= resp.status_code < 300:
            logger.info("[AdminReportAPI] aptitude auto-seed completed: %d questions", len(generated))
            return True
        logger.warning("[AdminReportAPI] aptitude auto-seed save failed %d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("[AdminReportAPI] aptitude auto-seed save exception: %s", e)
    return False


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

    # ── Step 1: Fetch full question objects + selected answers from MongoDB ──
    mongo_questions: list = []
    selected_answers: list = []
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
                parsed = r.json()
                body = parsed if isinstance(parsed, dict) else {}
                mongo_questions = body.get("questions") or []
                selected_answers = (body.get("record") or {}).get("selectedAnswers") or []
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
                parsed = r2.json()
                body = parsed if isinstance(parsed, dict) else {}
                mongo_questions = body.get("generatedQuestions") or []
                selected_answers = body.get("selectedAnswers") or []
                logger.info("[AdminReportAPI][Aptitude] MongoDB fallback questions: %d", len(mongo_questions))
            else:
                logger.warning(
                    "[AdminReportAPI][Aptitude] fallback %d: %s",
                    r2.status_code, r2.text[:300],
                )
    except Exception as e:
        logger.warning("[AdminReportAPI][Aptitude] MongoDB fetch exception: %s", e)

    # ── Step 2: Build answer map from selectedAnswers stored in Mongo ────────
    answer_map: Dict[str, Dict] = {}
    for row in selected_answers or []:
        qid = str(row.get("questionId") or "")
        if not qid:
            continue
        answer_map[qid] = {
            "answerText": row.get("selectedOptionText") or row.get("answerText") or "",
            "selectedOptionKey": row.get("selectedOptionKey") or "",
            "selectedOptionText": row.get("selectedOptionText") or row.get("answerText") or "",
        }
    logger.info(
        "[AdminReportAPI][Aptitude] Mongo selectedAnswers fetched: %d rows, %d unique questionIds",
        len(selected_answers or []),
        len(answer_map),
    )

    # ── Step 3: Merge question objects with candidate answers ────────────────
    if mongo_questions:
        merged = []
        for q in mongo_questions:
            qid = str(q.get("id") or "")
            ans_row = answer_map.get(qid) or {}
            merged.append({
                **q,
                "questionId": qid,
                "answerText": ans_row.get("answerText") or q.get("answerText") or "",
                "selectedOptionKey": ans_row.get("selectedOptionKey") or "",
                "selectedOptionText": ans_row.get("selectedOptionText") or "",
                "correctAnswer": q.get("correctAnswer") or "",
            })
        logger.info("[AdminReportAPI][Aptitude] MERGED result: %d questions (%d with answers)", len(merged), sum(1 for m in merged if m.get("answerText")))
        return merged

    # Fallback: if no Mongo questions, return selected answers only
    if answer_map:
        logger.info("[AdminReportAPI][Aptitude] Mongo questions empty — returning %d selected-answer rows", len(answer_map))
        return [
            {
                "questionId": qid,
                "answerText": v["answerText"],
                "selectedOptionKey": v.get("selectedOptionKey") or "",
                "selectedOptionText": v.get("selectedOptionText") or "",
                "correctAnswer": "",
            }
            for qid, v in answer_map.items()
        ]

    logger.warning("[AdminReportAPI][Aptitude] MongoDB returned no aptitude question/answer data — r4_data will be []")
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
            _fetch_conversational_data(client, cand_url, candidate_id, position_id, resolved_tenant_id, question_set_id or ""),
            _fetch_coding_data(client, cand_url, candidate_id, position_id),
            _fetch_aptitude_data(client, cand_url, candidate_id, position_id, question_set_id or "", client_id or ""),
            _fetch_assessment_summary(client, admin_url, headers, position_id, candidate_id, resolved_tenant_id),
            return_exceptions=False,
        )

        resolved_question_set_id = (
            (question_set_id or "").strip()
            or str((base_data.get("candidateProfile") or {}).get("questionSetId") or "").strip()
        )

        if (not r3_data or not _extract_coding_sets(r3_data)) or (not r4_data):
            sections_data = await _fetch_question_sections_data(
                client=client,
                admin_url=admin_url,
                headers=headers,
                question_set_id=resolved_question_set_id,
                tenant_id=resolved_tenant_id,
            )

            if (not r3_data or not _extract_coding_sets(r3_data)) and sections_data:
                seeded_coding = await _seed_coding_if_missing(
                    client=client,
                    cand_url=cand_url,
                    candidate_id=candidate_id,
                    position_id=position_id,
                    question_set_id=resolved_question_set_id,
                    sections_data=sections_data,
                )
                if seeded_coding:
                    r3_data = await _fetch_coding_data(client, cand_url, candidate_id, position_id)

            if (not r4_data) and sections_data:
                seeded_aptitude = await _seed_aptitude_if_missing(
                    client=client,
                    cand_url=cand_url,
                    candidate_id=candidate_id,
                    position_id=position_id,
                    question_set_id=resolved_question_set_id,
                    sections_data=sections_data,
                )
                if seeded_aptitude:
                    r4_data = await _fetch_aptitude_data(
                        client,
                        cand_url,
                        candidate_id,
                        position_id,
                        resolved_question_set_id,
                        client_id or "",
                    )

    # CandidateTest source parity: when DB conversational payload is empty,
    # fetch round questions via /ws/test and shape to categories/conversationSets.
    if not r1r2_data:
        ws_fallback = await _fetch_conversational_data_via_ws_test(
            candidate_id=candidate_id,
            position_id=position_id,
            client_id=client_id,
            tenant_id=resolved_tenant_id,
            question_set_id=question_set_id or "",
            assessment_summary_id=str((assessment_summary or {}).get("id") or ""),
        )
        if ws_fallback:
            r1r2_data = ws_fallback

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

        async def _return_hydrated_existing(existing_doc: Dict[str, Any]) -> JSONResponse:
            payload = await fetch_report_payload_data(
                position_id=positionId,
                candidate_id=candidateId,
                client_id=clientId,
                tenant_id=resolved_tenant_id,
                question_set_id=questionSetId,
            )
            hydrated = _hydrate_existing_report_doc(existing_doc, payload)
            if hydrated != existing_doc:
                try:
                    await rg._save_report(positionId, candidateId, hydrated)
                    logger.info("[AdminReportAPI] Hydrated stale existing report for %s/%s", positionId, candidateId)
                except Exception as save_err:
                    logger.warning("[AdminReportAPI] Failed to persist hydrated report for %s/%s: %s", positionId, candidateId, save_err)
            return JSONResponse(
                status_code=200,
                content=rg._build_completed_response(hydrated, is_existing=True),
            )

        # ── Step 1: Check MySQL is_generated flag ────────────────────────────
        if not forceRegenerate:
            generation_flag = await _get_report_generation_flag(positionId, candidateId)
            if generation_flag.get("isGenerated"):
                existing_doc = await rg._get_existing_report(positionId, candidateId)
                if existing_doc and existing_doc.get("isGenerated"):
                    logger.info("[AdminReportAPI] Returning hydrated existing report (is_generated=true) for %s/%s", positionId, candidateId)
                    return await _return_hydrated_existing(existing_doc)
                get_resp = await rg.get_report(positionId, candidateId)
                get_body = _json_from_response(get_resp)
                if get_resp.status_code == 200:
                    logger.info("[AdminReportAPI] Returning existing report body (is_generated=true) for %s/%s", positionId, candidateId)
                    return JSONResponse(status_code=200, content=get_body)
                # MySQL said generated but MongoDB returned nothing — fall through to regenerate

        # ── Step 2: Check MongoDB directly (catches MySQL out-of-sync) ───────
        existing = await rg._get_existing_report(positionId, candidateId)
        if not forceRegenerate and existing and existing.get("isGenerated"):
            logger.info("[AdminReportAPI] Report found in MongoDB (MySQL flag out-of-sync) for %s/%s", positionId, candidateId)
            return await _return_hydrated_existing(existing)

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
