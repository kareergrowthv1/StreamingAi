"""
Streaming AI — FastAPI app.
- Single WebSocket /ws/test: init (all IDs), calibration screenshot, questions fetch, answer save,
  assessment-summary updates, proctoring. Conversational: cross-questions; non-conversational: no cross-questions.
  Coding/aptitude: generate questions. No frontend API calls during test.
- Screenshot upload, chunk upload, /ws/video for binary streaming, merge.
"""
import asyncio
import base64
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

import httpx
import websockets
import edge_tts
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Body, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from merge_utils import do_merge_chunks
from face_detection import analyze as face_detection_analyze
from cross_question import generate_cross_question
from screening_utils import (
    build_flat_qa_list,
    build_screening_categories_from_sections,
    parse_seconds,
    flat_index_to_conv_and_pair,
    followup_count_for_conv_key,
    get_next_unanswered_from_screening,
    questions_for_round,
    sort_conv_key,
)
from api_admin import (
    get_assessment_summaries as admin_get_assessment_summaries,
    get_question_sections as admin_get_question_sections,
    get_cross_question_settings as admin_get_cross_question_settings,
    put_round_timing as admin_put_round_timing,
    put_update_interview_status as admin_put_update_interview_status,
    patch_complete_interview as admin_patch_complete_interview,
)
from api_candidate import (
    get_candidate_interview as candidate_get_interview,
    post_candidate_interview as candidate_post_interview,
    put_candidate_interview as candidate_put_interview,
    get_question_answers as candidate_get_question_answers,
    post_question_answer as candidate_post_question_answer,
    get_interview_responses as ir_get,
    post_interview_responses as ir_post,
    patch_interview_responses as ir_patch,
)
# No DB in Streaming AI — Q&A save/fetch via CandidateBackend; proctoring/session storage is in other services.
# AI routes: resume ATS, skills, JD – all use dynamic config (Superadmin GET /superadmin/settings/ai-config)
from ResumeAts import router as resume_ats_router
from skill_generator import router as skill_generator_router
from jd_generator import router as jd_generator_router
from aptitude_generator import generate_aptitude_questions
from coding_question import router as coding_question_router
from interview_question_generator import router as interview_question_router
from admin_report_api import router as admin_report_router
from daily_quiz import router as daily_quiz_router
from Conversationalmessage import router as conversational_message_router
from ResumeReport import router as resume_report_router
from AIResumecontent import router as resume_ai_content_router
from EmailTemplate import router as email_template_router
from stt_streaming import AssemblyAIStreamRunner
from fakeoffer import router as fake_offer_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def init_db():
    """No-op: Streaming AI does not use a database; Q&A and session data live in CandidateBackend/AdminBackend."""
    pass


app = FastAPI(title="Streaming AI", description="Test portal: single WebSocket for screenshot, questions, answers, streaming, proctoring", version="1.0.0")


# --- Request/Response logging: log every API call and response ---
@app.middleware("http")
async def log_requests(request, call_next):
    """Log every incoming HTTP request and response status (and body when small)."""
    method = request.method
    path = request.url.path
    query = str(request.url.query) if request.url.query else ""
    logger.info("[API IN] %s %s%s", method, path, f"?{query}" if query else "")
    response = await call_next(request)
    logger.info("[API OUT] %s %s -> %s", method, path, response.status_code)
    return response


# CORS setup:
# • If CORS_ORIGINS contains a single "*" we use allow_origin_regex instead
#   (Starlette disallows allow_origins=["*"] together with allow_credentials=True)
# • Otherwise use the explicit list from config (comma-separated env var CORS_ORIGINS)
_cors_origins = config.CORS_ORIGINS
if _cors_origins == ["*"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# AI routes: resume ATS, skills, JD, coding, interview questions (dynamic API config – no hardcoded keys)
app.include_router(resume_ats_router)
app.include_router(skill_generator_router)
app.include_router(jd_generator_router)
app.include_router(coding_question_router)
app.include_router(interview_question_router)
app.include_router(admin_report_router)
app.include_router(daily_quiz_router)
app.include_router(conversational_message_router)
app.include_router(resume_report_router)
app.include_router(resume_ai_content_router)
app.include_router(fake_offer_router)
from verify_admin_token import verify_admin_token
# app.include_router(email_template_router, prefix="/ai") # Moved to main.py directly for reliability

# Report generator (FIFO queue, all 4 rounds, MongoDB + MySQL updates)
from report_generator import router as report_router, _start_worker as _start_report_worker, _ensure_report_index
app.include_router(report_router)


# --- Pydantic models ---
class EmailTemplateAIRequest(BaseModel):
    mode: str = Field(..., description="'generate' or 'refine'")
    prompt: str = Field(..., description="User's prompt or instruction")
    currentBody: Optional[str] = Field(None, description="Current email body (for refinement)")
    variables: Optional[List[str]] = Field(None, description="List of available dynamic variables")


class ProctoringEvent(BaseModel):
    event: str = Field(..., description="no_face | multiple_faces | head_turned | looking_left | looking_right | looking_up | looking_down")
    confidence: float = Field(..., ge=0, le=1)
    timestamp: float = Field(..., description="Unix timestamp")


class CrossQuestionRequest(BaseModel):
    code: str


class ScheduleInterviewRequest(BaseModel):
    """Request for schedule-interview (private link). Same as Streaming. Called by AdminBackend."""
    candidateId: str
    email: str
    positionId: str
    questionSetId: str
    clientId: str
    interviewPlatform: str = "BROWSER"
    linkActiveAt: str
    linkExpiresAt: str
    createdBy: Optional[str] = None
    sendInviteBy: Optional[str] = "EMAIL"
    candidateName: str
    companyName: str
    organizationId: Optional[str] = None
    positionName: str
    verificationCode: Optional[str] = None


class RunCodeRequest(BaseModel):
    code: str


class TTSRequest(BaseModel):
    text: str
    voice: str


class SubmitAnswerAndGetNextRequest(BaseModel):
    """Submit answer for round 1 or 2 and get next question (main or cross)."""
    candidateId: str
    positionId: str
    questionSetId: str
    round: int  # 1 or 2
    questionId: str  # flat index as string, e.g. "0", "1"
    answer: str = ""
    tenantId: Optional[str] = None
    clientId: Optional[str] = None
    isConversational: Optional[bool] = None
    crossQuestionCountGeneral: Optional[int] = None
    crossQuestionCountPosition: Optional[int] = None


@app.post("/ai/generate-email-template")
@app.post("/api/ai/generate-email-template")
async def generate_email_template(request: EmailTemplateAIRequest = Body(...), _: bool = Depends(verify_admin_token)):
    """Generate or refine an email template body using AI."""
    from ai_config_loader import get_ai_config
    
    available_vars = ", ".join(request.variables) if request.variables is not None else "{candidate_name}, {Position_title}, {company_name}"
    
    system_prompt = f"""
You are a professional HR and Recruitment Assistant. Your goal is to write high-quality, professional emails.
You MUST use dynamic variables in curly braces like {{candidate_name}}.

Available variables you can use:
{available_vars}

IMPORTANT: If `{company_name}` is empty, not provided, or marked as optional, you MUST use the `{Position_title}` as the primary identifier in the email context (e.g., "You have been selected for the {{Position_title}} role" instead of "You have been selected for a role at {{company_name}}").

If the user provides a prompt, generate a complete email Subject and Body.
If the user provides a 'currentBody', refine and improve both the subject and the body while keeping the core message and variables intact.

Response format: Return a JSON object with two keys:
1. "subject": A professional subject line (can include variables, prefer Position_title over company_name if ambiguous).
2. "body": The complete email body.
"""

    if request.mode == "refine" and request.currentBody:
        user_prompt = f"Refine the following email based on this instruction: {request.prompt}\n\nCurrent Body:\n{request.currentBody}"
    else:
        user_prompt = f"Generate a professional email subject and body for: {request.prompt}"

    def _get_openai_client(cfg: dict):
        api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Set in Superadmin Settings > AI Config or OPENAI_API_KEY.")
        import openai
        return openai.OpenAI(
            api_key=api_key,
            base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
            timeout=cfg.get("timeout", 300),
            max_retries=cfg.get("maxRetries", 3),
        )

    try:
        cfg = await get_ai_config()
        client = _get_openai_client(cfg)
        model = cfg.get("model", "gpt-4o")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1000,
        )
        
        data = json.loads(response.choices[0].message.content or "{}")
        return {
            "success": True, 
            "subject": data.get("subject", ""),
            "body": data.get("body", "")
        }
    except Exception as e:
        logger.exception("Email template AI generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit-answer-and-get-next")
@app.post("/api/submit-answer-and-get-next")
async def submit_answer_and_get_next(request: SubmitAnswerAndGetNextRequest):
    """
    Save the candidate's answer for round 1 or 2 and return the next question (next main question or generated cross-question).
    Frontend uses this to advance after each answer without relying on WebSocket next_question.
    """
    logger.info(f"[API IN] POST /api/submit-answer-and-get-next | Round {request.round} | Q#{request.questionId} | isConversational={request.isConversational}")
    
    if request.round not in (1, 2):
        raise HTTPException(status_code=400, detail="round must be 1 or 2")
    cand_url = getattr(config, "CANDIDATE_BACKEND_URL", None)
    admin_url = getattr(config, "ADMIN_BACKEND_URL", None)
    if not cand_url:
        raise HTTPException(status_code=503, detail="CANDIDATE_BACKEND_URL not configured")
    cid = request.candidateId.strip()
    pid = request.positionId.strip()
    qset_id = request.questionSetId.strip()
    tenant_id = (request.tenantId or "").strip()
    round_num = request.round
    answer = (request.answer or "").strip()
    flat_index = int(request.questionId) if str(request.questionId).isdigit() else 0
    is_conversational = bool(request.isConversational) if request.isConversational is not None else False

    # 1) GET or create screening doc
    r_status, get_body = await candidate_get_interview(cand_url, cid, pid, qset_id)
    screening_id = None
    doc = None
    if r_status == 404 and admin_url:
        section_row = {}
        try:
            sr_status, sr_json = await admin_get_question_sections(admin_url, qset_id, tenant_id)
            if sr_status == 200 and sr_json:
                raw = (sr_json.get("data") or [])
                section_row = raw[0] if isinstance(raw, list) and raw else {}
        except Exception:
            pass
        categories_new = build_screening_categories_from_sections(section_row)
        pr_status, post_body = await candidate_post_interview(cand_url, {
            "candidateId": cid, "positionId": pid, "questionSetId": qset_id,
            "clientId": request.clientId or "", "categories": categories_new,
            "isScreeningCompleted": False, "type": "CONVERSATIONAL",
        }, tenant_id=tenant_id)
        if 200 <= pr_status < 300 and post_body:
            screening_id = post_body.get("id")
            doc = post_body
        else:
            raise HTTPException(status_code=500, detail="Failed to create screening document")
    elif r_status == 200:
        screening_id = get_body.get("id")
        doc = get_body
    else:
        raise HTTPException(status_code=500 if r_status != 200 else 502, detail="Failed to load screening document")

    # 2) Update answer in categories and PUT
    raw_cats = (doc or {}).get("categories") or doc or {}
    if not isinstance(raw_cats, dict):
        raw_cats = {}
    categories = json.loads(json.dumps(raw_cats))
    cat_key = "generalQuestion" if round_num == 1 else "positionSpecificQuestion"
    if cat_key not in categories:
        categories[cat_key] = {}
    cat = categories[cat_key]
    if "conversationSets" not in cat:
        cat["conversationSets"] = {}
    conv_sets = cat["conversationSets"]
    conv_key, pair_idx = flat_index_to_conv_and_pair(doc, round_num, flat_index)
    if conv_key is None:
        conv_key = f"conversationQuestion{flat_index + 1}"
        pair_idx = 0
    if conv_key not in conv_sets or not isinstance(conv_sets[conv_key], list) or pair_idx is None or pair_idx >= len(conv_sets[conv_key]):
        raise HTTPException(status_code=400, detail="Invalid question index for this round")
    conv_sets[conv_key][pair_idx]["answer"] = answer
    put_status, put_body = await candidate_put_interview(cand_url, screening_id, {"categories": categories})
    if put_status < 200 or put_status >= 300:
        raise HTTPException(status_code=500, detail="Failed to save answer")

    # 3) Fresh doc for next question (GET after PUT)
    get_status, fresh_doc = await candidate_get_interview(cand_url, cid, pid, qset_id)
    if get_status == 200:
        doc = fresh_doc
    else:
        doc = put_body if isinstance(put_body, dict) else doc

    flat = build_flat_qa_list(doc, round_num)
    next_idx = flat_index + 1
    next_q_text = ""
    all_done = next_idx >= len(flat)

    # 4) Fetch authoritative cross-question settings from AdminBackend DB (for HTTP stateless calls)
    # Do NOT use frontend-sent values; always fetch what admin configured.
    cross_count_general = 2  # defaults
    cross_count_position = 2
    if admin_url and request.clientId:
        try:
            cq_settings = await admin_get_cross_question_settings(admin_url, request.clientId, tenant_id)
            cross_count_general = cq_settings["crossQuestionCountGeneral"]
            cross_count_position = cq_settings["crossQuestionCountPosition"]
            logger.info(f"[API] Cross-question settings from DB: general={cross_count_general} position={cross_count_position}")
        except Exception as _cq_e:
            logger.warning(f"[API] Cross-question settings fetch failed (using defaults): {_cq_e}")

    # 5) Optional: generate cross-question (if conversational and under per-main-question limit)
    # NOTE: Do NOT gate on `not all_done` — last main question also deserves cross-questions.
    conv_key_for_cross = conv_key
    default_cross = cross_count_general if round_num == 1 else cross_count_position
    cross_count = min(4, max(1, int(default_cross or 2)))
    if is_conversational and conv_key_for_cross and screening_id:
        followups = followup_count_for_conv_key(doc, round_num, conv_key_for_cross)
        logger.info(f"[CROSS-Q] Attempting generation: conversational={is_conversational}, followups={followups}/{cross_count}, conv_key={conv_key_for_cross}")
        if followups < cross_count:
            flat_cross = build_flat_qa_list(doc, round_num)
            current_q = flat_cross[flat_index][2] if flat_index < len(flat_cross) else ""
            history_list = [qq for _ck, _pi, qq, _ in flat_cross if _ck == conv_key_for_cross]
            try:
                new_q = await generate_cross_question(current_q or "", answer or "", history_list, tenant_id=tenant_id)
                if new_q:
                    logger.info(f"[CROSS-Q] Generated successfully: '{new_q[:100]}...'")
                    categories_cross = json.loads(json.dumps((doc or {}).get("categories") or {}))
                    cat_c = categories_cross.get(cat_key) or {}
                    conv_c = cat_c.get("conversationSets") or {}
                    if conv_key_for_cross in conv_c and isinstance(conv_c[conv_key_for_cross], list):
                        conv_c[conv_key_for_cross].append({"question": new_q, "answer": ""})
                        r_c_status, _ = await candidate_put_interview(cand_url, screening_id, {"categories": categories_cross})
                        if 200 <= r_c_status < 300:
                            next_idx = flat_index + 1
                            next_q_text = new_q
                            all_done = False
                            logger.info(f"[CROSS-Q] Cross-question inserted successfully at index {next_idx}")
                        else:
                            logger.warning(f"[CROSS-Q] Failed to save cross-question: PUT returned {r_c_status}")
                else:
                    logger.warning("[CROSS-Q] generate_cross_question returned empty")
            except Exception as e:
                logger.warning("[CROSS-Q] Cross-question generation failed: %s", e)

    if not next_q_text and next_idx < len(flat):
        next_q_text = flat[next_idx][2] or ""

    response_data = {
        "success": True,
        "answerSaved": True,
        "nextQuestionIndex": next_idx if not all_done else None,
        "nextQuestionText": next_q_text or None,
        "allQuestionsAnswered": all_done,
    }
    logger.info(f"[API OUT] POST /api/submit-answer-and-get-next -> 200 | nextIndex={response_data['nextQuestionIndex']} | allDone={all_done} | nextQ='{(next_q_text or '')[:80]}...'")
    return response_data


@app.post("/api/run-code")
async def run_code(request: RunCodeRequest):
    """Compile and run Java code against test cases."""
    import subprocess
    import tempfile
    import re
    
    test_cases = [
        {"input": "hello", "expected": "olleh"},
        {"input": "Java", "expected": "avaJ"},
        {"input": "12345", "expected": "54321"},
        {"input": "", "expected": ""}
    ]
    
    results = []
    
    # Robust wrapping logic
    code_to_write = request.code.strip()
    # If the user code doesn't contain a class definition that wraps everything
    if not re.search(r'class\s+Solution', code_to_write):
        # Check if it has ANY class header at the top
        if re.search(r'^\s*public\s+class\s+', code_to_write) or re.search(r'^\s*class\s+', code_to_write):
            # It has a class but it's not named Solution. Java requires file name == class name.
            # We'll try to rename their class to Solution
            code_to_write = re.sub(r'(class\s+)\w+', r'\1Solution', code_to_write, count=1)
        else:
            # It's likely just a method or a fragment, wrap it
            code_to_write = f"public class Solution {{\n{code_to_write}\n}}"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        user_file = tmp_path / "Solution.java"
        with open(user_file, "w") as f:
            f.write(code_to_write)
            
        # Create a runner file (Runner.java)
        runner_content = """
public class Runner {
    public static void main(String[] args) {
        String[] inputs = {"hello", "Java", "12345", ""};
        String[] expected = {"olleh", "avaJ", "54321", ""};
        
        for (int i = 0; i < inputs.length; i++) {
            System.out.println("TEST_START");
            try {
                // We use dynamic instantiation or static call if possible
                // But for now, we assume Solution has a static reverseString
                String result = Solution.reverseString(inputs[i]);
                if (result == null) result = "null";
                System.out.println(result);
            } catch (Throwable e) {
                System.out.println("ERROR: " + e.toString());
            }
            System.out.println("TEST_END");
        }
    }
}
"""
        runner_file = tmp_path / "Runner.java"
        with open(runner_file, "w") as f:
            f.write(runner_content)
            
        # Compile both
        # We explicitly list Solution.java first
        compile_proc = subprocess.run(
            ["javac", "Solution.java", "Runner.java"],
            cwd=tmpdir,
            capture_output=True,
            text=True
        )
        
        if compile_proc.returncode != 0:
            # Check if it's because of a missing static method
            error_msg = compile_proc.stderr
            if "non-static method reverseString" in error_msg:
                error_msg += "\n\nTip: Make sure your method is 'public static'."
                
            return {
                "success": False,
                "error": "Compilation Error",
                "output": error_msg
            }
            
        # Run
        run_proc = subprocess.run(
            ["java", "Runner"],
            cwd=tmpdir,
            capture_output=True,
            text=True
        )
        
        if run_proc.returncode != 0:
            return {
                "success": False,
                "error": "Runtime Error",
                "output": run_proc.stderr
            }
            
        # Parse output
        raw_output = run_proc.stdout.splitlines()
        test_outputs = []
        capture = False
        current_test_output = []
        
        for line in raw_output:
            if line == "TEST_START":
                capture = True
                current_test_output = []
                continue
            if line == "TEST_END":
                capture = False
                test_outputs.append("\n".join(current_test_output))
                continue
            if capture:
                current_test_output.append(line)
                
        for i, actual in enumerate(test_outputs):
            is_passed = actual == test_cases[i]["expected"]
            results.append({
                "input": test_cases[i]["input"],
                "expected": test_cases[i]["expected"],
                "actual": actual,
                "passed": is_passed
            })
            
    return {
        "success": True,
        "results": results,
        "all_passed": all(r["passed"] for r in results)
    }


@app.post("/api/generate-cross-questions")
async def generate_cross_questions(request: CrossQuestionRequest):
    """Generate 3 cross-questions based on the provided code. Uses dynamic API and model from DB (Superadmin AI config)."""
    from ai_config_loader import get_ai_config

    _req = {"code": (request.code[:200] + "..." if len(request.code) > 200 else request.code)}
    print(f"\n[API] POST /api/generate-cross-questions\n  Request: {json.dumps(_req, indent=2)}\n", flush=True)
    prompt = f"""
    The candidate has written the following Java code for a task:
    ---
    {request.code}
    ---
    Based on this code, generate exactly 3 short, verbal interview questions to ask the candidate.
    The questions should be about:
    1. Why did they choose a specific approach or method?
    2. Explaining a specific line or logic in their code.
    3. A hypothetical 'what if' or improvement question related to their code.

    Return ONLY a JSON array of 3 strings. Example: ["Why did you use a while loop?", "Explain line 5.", "How would you optimize this?"]
    """

    try:
        cfg = await get_ai_config()
        api_key = cfg.get("apiKey") or __import__("os").getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key required. Set in Superadmin Settings > AI Config or OPENAI_API_KEY.")
        base_url = cfg.get("baseUrl", "https://api.openai.com/v1")
        model = cfg.get("model", "gpt-3.5-turbo")
        timeout = int(cfg.get("timeout", 300))
        async with httpx.AsyncClient(timeout=min(timeout, 60)) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a senior Java technical interviewer. Return only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7
                },
            )
            response.raise_for_status()
            result = response.json()
            content = (result.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            if not content:
                raise HTTPException(status_code=502, detail="Empty response from AI")
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            questions = json.loads(content)
            out = {"questions": questions}
            print(f"[API] POST /api/generate-cross-questions -> Response: {json.dumps(out, indent=2)}\n", flush=True)
            return out
    except httpx.HTTPStatusError as e:
        logger.exception("AI provider error: %s", e.response.text)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("Failed to generate cross-questions")
        raise HTTPException(status_code=500, detail=str(e))


# --- Path helpers (pathlib) ---
def _screenshot_path(client_id: str, position_id: str, candidate_id: str) -> Path:
    base = config.MERGED_DIR / client_id / position_id / candidate_id
    base.mkdir(parents=True, exist_ok=True)
    return base / "screenshots"


def _chunks_base(client_id: str, position_id: str, candidate_id: str) -> Path:
    base = config.CHUNKS_DIR / client_id / position_id / candidate_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def _merged_dir(client_id: str, position_id: str, candidate_id: str) -> Path:
    base = config.MERGED_DIR / client_id / position_id / candidate_id
    base.mkdir(parents=True, exist_ok=True)
    return base


# --- REST: Screenshot ---
@app.post("/api/screenshot")
async def upload_screenshot(
    client_id: str = Form(...),
    position_id: str = Form(...),
    candidate_id: str = Form(...),
    event_type: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """Save screenshot with optional event-based subfolder structure."""
    dir_path = _screenshot_path(client_id, position_id, candidate_id)
    
    # If event_type is provided, create a subfolder for it
    if event_type:
        dir_path = dir_path / event_type
        
    dir_path.mkdir(parents=True, exist_ok=True)
    
    import time
    name = f"screenshot_{int(time.time() * 1000)}.png"
    file_path = dir_path / name
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        return {"success": True, "path": str(file_path), "name": name, "event": event_type}
    except Exception as e:
        logger.exception("Screenshot upload failed")
        return {"success": False, "error": str(e)}


# --- WebSocket: Video Streaming (screen recording) ---
@app.websocket("/ws/video/{client_id}/{position_id}/{candidate_id}")
async def websocket_video_stream(
    websocket: WebSocket,
    client_id: str,
    position_id: str,
    candidate_id: str,
):
    """
    Receive screen recording binary data via WebSocket and append to a unique part file.
    Each connection creates a new part file (e.g., part_{timestamp}.webm).
    """
    await websocket.accept()
    base = _chunks_base(client_id, position_id, candidate_id)

    part_name = f"part_{int(time.time() * 1000)}.webm"
    file_path = base / part_name

    _banner = (
        "\n" + "=" * 64 + "\n"
        f"  LIVE STREAM START  —  /ws/video (screen recording)\n"
        f"  client_id={client_id} position_id={position_id} candidate_id={candidate_id}\n"
        f"  Saving to: {file_path}\n"
        "=" * 64 + "\n"
    )
    print(_banner, flush=True)
    logger.info("[LIVE STREAM START] /ws/video (screen) -> %s", part_name)

    chunk_count = 0
    total_bytes = 0
    try:
        with open(file_path, "ab") as f:
            while True:
                data = await websocket.receive_bytes()
                f.write(data)
                f.flush()
                chunk_count += 1
                total_bytes += len(data)
                if chunk_count % 20 == 0:  # Log every 20 chunks to keep it clean
                    print(f"[WS CHUNK] video candidate_id={candidate_id} part={part_name} received={total_bytes} bytes", flush=True)
    except WebSocketDisconnect:
        _close_banner = (
            "\n" + "=" * 64 + "\n"
            f"  LIVE STREAM END  —  /ws/video (screen recording)\n"
            f"  WEBSOCKET CLOSED. File: {part_name}  chunks={chunk_count}  bytes={total_bytes}\n"
            "=" * 64 + "\n"
        )
        print(_close_banner, flush=True)
        logger.info("[LIVE STREAM END] /ws/video -> %s (chunks=%d bytes=%d)", part_name, chunk_count, total_bytes)
    except Exception as e:
        logger.exception("Video WS error: %s", e)
        print(f"[WEBSOCKET ERROR] /ws/video: {e}\n", flush=True)


# --- WebSocket: Camera Recording ---
@app.websocket("/ws/camera/{client_id}/{position_id}/{candidate_id}")
async def websocket_camera_stream(
    websocket: WebSocket,
    client_id: str,
    position_id: str,
    candidate_id: str,
):
    """
    Receive front-camera video binary data via WebSocket.
    Saves chunks as camera_part_{timestamp}.webm in the chunks directory.
    """
    await websocket.accept()
    base = _chunks_base(client_id, position_id, candidate_id)

    part_name = f"camera_part_{int(time.time() * 1000)}.webm"
    file_path = base / part_name

    _cam_banner = (
        "\n" + "=" * 64 + "\n"
        f"  LIVE STREAM START  —  /ws/camera (camera recording)\n"
        f"  client_id={client_id} candidate_id={candidate_id}  file={part_name}\n"
        "=" * 64 + "\n"
    )
    print(_cam_banner, flush=True)
    logger.info("[LIVE STREAM START] /ws/camera -> %s", part_name)

    try:
        with open(file_path, "ab") as f:
            while True:
                data = await websocket.receive_bytes()
                f.write(data)
                f.flush()
                # Log camera chunk concise
                print(f"[WS CHUNK] camera candidate_id={candidate_id} part={part_name} len={len(data)}", flush=True)
    except WebSocketDisconnect:
        _cam_end = (
            "\n" + "=" * 64 + "\n"
            f"  LIVE STREAM END  —  /ws/camera (camera recording)\n"
            f"  WEBSOCKET CLOSED. File: {part_name}\n"
            "=" * 64 + "\n"
        )
        print(_cam_end, flush=True)
        logger.info("[LIVE STREAM END] /ws/camera -> %s", part_name)
    except Exception as e:
        logger.exception("Camera WS error: %s", e)


# --- REST: End test — trigger merge (Background Task) ---
@app.post("/api/end-test")
async def end_test(
    background_tasks: BackgroundTasks,
    client_id: str = Form(...),
    position_id: str = Form(...),
    candidate_id: str = Form(...),
):
    """End test: merge chunks with FFmpeg in a background task."""
    logger.info("end_test received: triggering background merging for %s/%s/%s", client_id, position_id, candidate_id)
    
    async def run_merges():
        # 1. Merge screen recording
        logger.info("[MERGE] Starting screen recording merge for %s/%s/%s", client_id, position_id, candidate_id)
        res_screen = await asyncio.to_thread(do_merge_chunks, client_id, position_id, candidate_id)
        if res_screen.get("success"):
            logger.info("[MERGE] Screen recording merge completed: %s", res_screen.get("merged"))
        else:
            logger.warning("[MERGE] Screen recording merge failed: %s", res_screen.get("error"))

        # 2. Merge camera recording
        logger.info("[MERGE] Starting camera recording merge for %s/%s/%s", client_id, position_id, candidate_id)
        res_camera = await asyncio.to_thread(
            do_merge_chunks,
            client_id, position_id, candidate_id,
            file_prefix="camera_part_",
            output_prefix="camera_recording"
        )
        if res_camera.get("success"):
            logger.info("[MERGE] Camera recording merge completed: %s", res_camera.get("merged"))
        else:
            logger.warning("[MERGE] Camera recording merge failed: %s", res_camera.get("error"))

    background_tasks.add_task(run_merges)
    
    return {
        "success": True,
        "message": "Merge job started in background.",
    }


def _compute_next_round(summary_data: dict) -> tuple:
    """
    From assessment summary data, compute (next_round, is_assessment_completed).
    next_round: 1, 2, 3, or 4; or None if all completed. New or no data -> start from round 1.
    """
    if not summary_data or not isinstance(summary_data, dict):
        return (1, False)
    data = summary_data.get("data") or summary_data

    # Guard: if assessmentStartTime is null the candidate never actually started the test.
    # Any completion flags present are stale/corrupted data — treat as a fresh start.
    assessment_start_time = data.get("assessmentStartTime")
    if not assessment_start_time:
        return (1, False)

    r1_a = data.get("round1Assigned", True)
    r1_c = data.get("round1Completed", False)
    r2_a = data.get("round2Assigned", True)
    r2_c = data.get("round2Completed", False)
    r3_a = data.get("round3Assigned", False)
    r3_c = data.get("round3Completed", False)
    r4_a = data.get("round4Assigned", False)
    r4_c = data.get("round4Completed", False)

    # Always check round progress first — do NOT rely on isAssessmentCompleted flag alone
    # because a DB bug can set it to True while rounds are still incomplete.
    if r1_a and not r1_c:
        return (1, False)
    if r2_a and not r2_c:
        return (2, False)
    if r3_a and not r3_c:
        return (3, False)
    if r4_a and not r4_c:
        return (4, False)

    # All assigned rounds are completed — assessment is truly done
    return (None, True)


async def _assessment_summary_log_loop(admin_url: str, cand_id: str, pos_id: str, tenant_id: str):
    """Every 60s fetch assessment-summaries and print to terminal so round timing is visible."""
    try:
        while True:
            await asyncio.sleep(60)
            try:
                status, body = await admin_get_assessment_summaries(
                    admin_url, cand_id, pos_id, tenant_id or ""
                )
                # Assessment summary fetched silently (no logs)
                pass
            except Exception as e:
                pass
    except asyncio.CancelledError:
        pass


def _full_url(method: str, url: str, params: Optional[dict] = None) -> str:
    """Build full URL with query string for GET requests."""
    if not params or method.upper() != "GET":
        return url
    from urllib.parse import urlencode
    qs = urlencode(params)
    return f"{url.rstrip('/')}?{qs}" if qs else url


def _log_ws_api(api_name: str, method: str, url: str, payload: dict, response_status: int, response_body, params: Optional[dict] = None):
    """Log one-line concise WebSocket-triggered API call and response summary."""
    try:
        query_params = params if params is not None else (payload.get("params") if isinstance(payload, dict) else None)
        full_url_str = _full_url(method, url, query_params)
        
        # One-line summary
        resp_str = json.dumps(response_body) if response_body is not None else "None"
        if len(resp_str) > 150: resp_str = resp_str[:150] + "..."
        
        log_line = f"[API LOG] {api_name} | {method} {full_url_str} -> {response_status} | Payload: {json.dumps(payload)} | Resp: {resp_str}"
        print(log_line, flush=True)
        logger.info(log_line)
    except Exception as e:
        print(f"[LOG ERROR] Failed to log API call: {e}", flush=True)


# --- WebSocket: Proctoring events (no_face, multiple_faces, head_turned) ---
@app.websocket("/ws/proctoring")
async def websocket_proctoring(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            logger.info("Proctoring event: %s", payload)
    except WebSocketDisconnect:
        logger.info("Proctoring WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)


# --- WebSocket: Single test session (init, calibration screenshot, questions, answers, assessment-summary) ---
@app.websocket("/ws/test")
async def websocket_test_session(websocket: WebSocket):
    """
    One WebSocket for the full test: init with IDs, calibration screenshot, questions fetch,
    answer save, assessment-summary updates. No frontend API calls.
    Messages from client: { "type": "init", "client_id", "position_id", "candidate_id", "question_set_id", "tenant_id?", "is_conversational?" }
                         { "type": "calibration_screenshot", "image_base64" }
                         { "type": "proctoring_frame", "image_base64" }  -> backend runs face detection, replies proctoring_result
                         { "type": "submit_answer", "question_id", "round", "answer" }
                         { "type": "get_next_question" }
                         { "type": "proctoring_event", "event", "confidence", "timestamp" }
    """
    await websocket.accept()
    _ws_banner = (
        "\n" + "=" * 64 + "\n"
        "  LIVE STREAM START  —  /ws/test (test session)\n"
        "  WEBSOCKET STARTED: client connected. All API calls and responses logged below.\n"
        "=" * 64 + "\n"
    )
    print(_ws_banner, flush=True)
    logger.info("[LIVE STREAM START] /ws/test")
    session = {
        "client_id": None,
        "position_id": None,
        "candidate_id": None,
        "question_set_id": None,
        "assessment_summary_id": None,
        "tenant_id": None,
        "is_conversational": False,
        "assessment_summary": None,
        "next_round": 1,
        "is_assessment_completed": False,
        "assemblyai_ws": None,
        "assemblyai_reader_task": None,
        "assemblyai_runner": None,
        # Cache screening docs per exam instance (keyed by "positionId_candidateId_questionSetId")
        "screening_cache": {},
        # Current exam instance keys (for backward compatibility, points to current screening)
        "screening_id": None,
        "screening_doc": None,
    }
    assessment_log_task = None
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue
            msg_type = (msg.get("type") or "").strip() if msg.get("type") is not None else ""
            # Log every incoming WS message to terminal (type + short payload)
            _in_payload = {k: v for k, v in msg.items() if k != "image_base64" and v is not None}
            if "image_base64" in msg:
                _in_payload["image_base64"] = f"<{len(msg.get('image_base64', ''))} chars>"
            _in_str = json.dumps(_in_payload, indent=2)[:500]
            if len(json.dumps(_in_payload)) > 500:
                _in_str += "..."
            print(f"\n[WS IN] type={msg_type}\n  payload: {_in_str}\n", flush=True)
            logger.info("[WS] message received: type=%s", msg_type)
            if msg_type == "init":
                session["client_id"] = msg.get("client_id")
                session["position_id"] = msg.get("position_id")
                session["candidate_id"] = msg.get("candidate_id")
                session["question_set_id"] = msg.get("question_set_id") or ""
                session["assessment_summary_id"] = msg.get("assessment_summary_id") or ""
                session["tenant_id"] = msg.get("tenant_id")
                # Store is_conversational from init (client sends it after email/OTP verify); session is source of truth
                session["is_conversational"] = bool(msg.get("is_conversational"))
                # Cross-question counts: start with frontend-provided values as baseline,
                # then override with authoritative DB values during the init sequence below.
                session["cross_question_count_general"] = min(4, max(1, int(msg.get("cross_question_count_general") or 2)))
                session["cross_question_count_position"] = min(4, max(1, int(msg.get("cross_question_count_position") or 2)))
                logger.info("Test session init (client_id, position_id, candidate_id): %s", {
                    "client_id": session["client_id"],
                    "position_id": session["position_id"],
                    "candidate_id": session["candidate_id"],
                })
                # Call assessment-summaries to know if new / in-between / completed; then send init_ok with nextRound
                admin_url = getattr(config, "ADMIN_BACKEND_URL", None)
                tenant_id = session.get("tenant_id") or ""
                cand_id = session.get("candidate_id")
                pos_id = session.get("position_id")
                # Fetch cross-question settings from AdminBackend DB (authoritative source).
                # Overrides any frontend-sent values so the count always matches AdminFrontend settings.
                if admin_url and session.get("client_id") and tenant_id:
                    try:
                        cq_settings = await admin_get_cross_question_settings(
                            admin_url, session["client_id"], tenant_id
                        )
                        session["cross_question_count_general"] = cq_settings["crossQuestionCountGeneral"]
                        session["cross_question_count_position"] = cq_settings["crossQuestionCountPosition"]
                        logger.info("[init] Cross-question settings from DB: general=%s position=%s",
                                    session["cross_question_count_general"], session["cross_question_count_position"])
                    except Exception as _cq_e:
                        logger.warning("[init] Cross-question settings DB fetch failed (keeping frontend values): %s", _cq_e)
                if admin_url and cand_id and pos_id:
                    try:
                        url = f"{admin_url.rstrip('/')}/candidates/assessment-summaries"
                        params = {"candidateId": cand_id, "positionId": pos_id}
                        status, resp_json = await admin_get_assessment_summaries(
                            admin_url, cand_id, pos_id, tenant_id
                        )
                        _log_ws_api(
                            "GET assessment-summaries",
                            "GET",
                            url,
                            {"params": params, "headers": {"X-Tenant-Id": tenant_id}},
                            status,
                            resp_json,
                        )
                        session["assessment_summary"] = resp_json
                        next_round, is_completed = _compute_next_round(resp_json)
                        session["next_round"] = next_round
                        session["is_assessment_completed"] = is_completed
                        # Extract the admin-created summary id from the GET response and override
                        # whatever the frontend sent (which may be empty) so init_ok echoes the
                        # correct id — all subsequent PATCHes from CandidateTest will use it.
                        fetched_summary_id = None
                        if isinstance(resp_json, dict):
                            _d = resp_json.get("data") or {}
                            if isinstance(_d, dict):
                                fetched_summary_id = _d.get("id")
                        if fetched_summary_id:
                            session["assessment_summary_id"] = str(fetched_summary_id)
                    except Exception as e:
                        _log_ws_api("GET assessment-summaries (init) FAILED", "GET", url, {"params": params}, 0, {"error": str(e)})
                        logger.exception("init: assessment-summaries failed: %s", e)
                        session["next_round"] = 1
                        session["is_assessment_completed"] = False
                init_ok_payload = {
                    "type": "init_ok",
                    "session": {
                        "client_id": session["client_id"],
                        "position_id": session["position_id"],
                        "candidate_id": session["candidate_id"],
                        "question_set_id": session["question_set_id"],
                        "assessment_summary_id": session["assessment_summary_id"],
                        "tenant_id": session["tenant_id"],
                        "is_conversational": session["is_conversational"],
                    },
                    "assessmentSummary": session.get("assessment_summary"),
                    "nextRound": session.get("next_round"),
                    "isAssessmentCompleted": session.get("is_assessment_completed", False),
                }
                try:
                    logger.info("[WS] init_ok sent to client (nextRound=%s, isAssessmentCompleted=%s): %s", session.get("next_round"), session.get("is_assessment_completed"), json.dumps(init_ok_payload, indent=2)[:1500])
                except Exception:
                    pass
                await websocket.send_json(init_ok_payload)
                print(f"[WS OUT] sent init_ok (nextRound={session.get('next_round')}, isAssessmentCompleted={session.get('is_assessment_completed')})\n", flush=True)
                # Start 1-min periodic assessment summary log (round timing visibility)
                if assessment_log_task is None and admin_url and cand_id and pos_id:
                    assessment_log_task = asyncio.create_task(
                        _assessment_summary_log_loop(admin_url, cand_id, pos_id, tenant_id or "")
                    )
                    # Assessment summary loop started silently
            elif msg_type == "get_screening":
                # GET screening first; if 404 (first time) -> get question set, POST screening with empty answers; return full doc
                cand_url = getattr(config, "CANDIDATE_BACKEND_URL", None)
                admin_url = getattr(config, "ADMIN_BACKEND_URL", None)
                cid = session.get("candidate_id")
                pid = session.get("position_id")
                qset_id = session.get("question_set_id")
                client_id = session.get("client_id")
                tenant_id = session.get("tenant_id") or ""
                if not cand_url or not cid or not pid or not qset_id:
                    await websocket.send_json({"type": "error", "message": "Init required or missing candidate/position/questionSetId"})
                    continue
                
                # Create cache key for this specific exam instance
                assessment_summary_id = session.get("assessment_summary_id") or ""
                cache_key = f"{pid}_{cid}_{qset_id}_{assessment_summary_id}"
                screening_cache = session.get("screening_cache") or {}
                
                try:
                    get_url = f"{cand_url}/candidate/assessment-summary"
                    params = {"candidateId": cid, "positionId": pid, "questionSetId": qset_id}
                    tenant_id = session.get("tenant_id") or ""
                    r_status, get_body = await candidate_get_interview(cand_url, cid, pid, qset_id, tenant_id=tenant_id)
                    _log_ws_api("GET assessment-summary (get_screening)", "GET", get_url, {"params": params}, r_status, get_body)
                    if r_status == 404:
                        # First time: get question set, build categories with answer "", POST
                        if not admin_url:
                            await websocket.send_json({"type": "error", "message": "Admin backend not configured"})
                            continue
                        sections_url = f"{admin_url}/internal/question-sections/question-set/{qset_id}"
                        sr_status, sections_json = await admin_get_question_sections(admin_url, qset_id, tenant_id)
                        _log_ws_api("GET question-sections (get_screening first-time)", "GET", sections_url, {}, sr_status, sections_json)
                        if sr_status != 200:
                            await websocket.send_json({"type": "error", "message": "Failed to fetch question set"})
                            continue
                        raw = sections_json.get("data") or {}
                        section_row = raw[0] if isinstance(raw, list) and len(raw) > 0 else raw
                        # Note: assessments_summary in MySQL doesn't use categories/doc model as heavily as MongoDB,
                        # but we keep the logic for backward compatibility if the backend handles it.
                        post_url = f"{cand_url}/candidate/assessment-summary"
                        post_payload = {
                            "candidateId": cid,
                            "positionId": pid,
                            "questionId": qset_id,
                            "totalRoundsAssigned": 4,
                            "round1Assigned": 1,
                            "round2Assigned": 1,
                            "round3Assigned": 1,
                            "round4Assigned": 1
                        }
                        tenant_id = session.get("tenant_id") or ""
                        pr_status, post_body = await candidate_post_interview(cand_url, post_payload, tenant_id=tenant_id)
                        _log_ws_api("POST assessment-summary (get_screening first-time)", "POST", post_url, post_payload, pr_status, post_body)
                        if pr_status not in (200, 201):
                            await websocket.send_json({"type": "error", "message": "Failed to create screening"})
                            continue
                        # Update both cache and session
                        screening_cache[cache_key] = {"id": post_body.get("id"), "doc": post_body}
                        session["screening_cache"] = screening_cache
                        session["screening_id"] = post_body.get("id")
                        session["screening_doc"] = post_body
                        await websocket.send_json({"type": "screening_ok", "screening": post_body})
                        print(f"[WS OUT] sent screening_ok (created new, id={post_body.get('id')})\n", flush=True)
                    else:
                        if r_status != 200:
                            await websocket.send_json({"type": "error", "message": "Failed to get screening"})
                            continue
                        # Update both cache and session
                        screening_cache[cache_key] = {"id": get_body.get("id"), "doc": get_body}
                        session["screening_cache"] = screening_cache
                        session["screening_id"] = get_body.get("id")
                        session["screening_doc"] = get_body
                        await websocket.send_json({"type": "screening_ok", "screening": get_body})
                        print(f"[WS OUT] sent screening_ok (existing, id={get_body.get('id')})\n", flush=True)
                except Exception as e:
                    _log_ws_api("get_screening FAILED", "GET", get_url if cand_url else "", {}, 0, {"error": str(e)})
                    logger.exception("get_screening failed: %s", e)
                    await websocket.send_json({"type": "error", "message": str(e)})
            elif msg_type == "calibration_screenshot":
                logger.info("[WS] calibration_screenshot received")
                image_b64 = msg.get("image_base64")
                if image_b64 and session.get("client_id") and session.get("position_id") and session.get("candidate_id"):
                    try:
                        raw = base64.b64decode(image_b64)
                        dir_path = _screenshot_path(session["client_id"], session["position_id"], session["candidate_id"]) / "calibration"
                        dir_path.mkdir(parents=True, exist_ok=True)
                        name = f"screenshot_{int(time.time() * 1000)}.png"
                        (dir_path / name).write_bytes(raw)
                        await websocket.send_json({"type": "calibration_ok", "saved": name})
                        print(f"[WS OUT] sent calibration_ok (saved={name})\n", flush=True)
                    except Exception as e:
                        logger.exception("Calibration screenshot save failed")
                        await websocket.send_json({"type": "error", "message": str(e)})
                else:
                    await websocket.send_json({"type": "error", "message": "Init first or missing image_base64"})
            elif msg_type == "submit_answer":
                question_id = msg.get("question_id")
                round_name = msg.get("round")
                answer = msg.get("answer", "")
                if not session.get("candidate_id") or question_id is None:
                    await websocket.send_json({"type": "error", "message": "Init required or missing question_id"})
                    continue
                round_str = str(round_name) if round_name is not None else ""
                round_num = int(round_str) if (isinstance(round_str, str) and round_str.isdigit()) else (int(round_name) if isinstance(round_name, (int, float)) else 0)
                cand_url = getattr(config, "CANDIDATE_BACKEND_URL", None)
                answer_saved_ok = False
                # Round 1 & 2: update screening document (PUT candidate-interviews/:id) — save answer first, then send answer_saved
                if cand_url and round_num in (1, 2):
                    # Ensure we have screening_id and screening_doc for this specific exam instance
                    cid = session.get("candidate_id")
                    pid = session.get("position_id")
                    qset_id = session.get("question_set_id")
                    assessment_summary_id = session.get("assessment_summary_id") or ""
                    cache_key = f"{pid}_{cid}_{qset_id}_{assessment_summary_id}"
                    screening_cache = session.get("screening_cache") or {}
                    
                    if cache_key not in screening_cache:
                        try:
                            tenant_id = session.get("tenant_id") or ""
                            admin_url = getattr(config, "ADMIN_BACKEND_URL", None)
                            get_url = f"{cand_url}/candidate-interviews/candidate/{cid}/position/{pid}"
                            tenant_id = session.get("tenant_id") or ""
                            r_status, get_body = await candidate_get_interview(cand_url, cid, pid, qset_id, tenant_id=tenant_id)
                            if r_status == 404 and admin_url:
                                section_row = {}
                                try:
                                    sr_status, sr_json = await admin_get_question_sections(admin_url, qset_id, tenant_id)
                                    if sr_status == 200 and sr_json:
                                        raw = (sr_json.get("data") or [])
                                        section_row = raw[0] if isinstance(raw, list) and raw else {}
                                except Exception:
                                    pass
                                categories_new = build_screening_categories_from_sections(section_row)
                                pr_status, post_body = await candidate_post_interview(cand_url, {
                                    "candidateId": cid, "positionId": pid, "questionSetId": qset_id,
                                    "clientId": session.get("client_id"), "categories": categories_new,
                                    "isScreeningCompleted": False, "type": "CONVERSATIONAL",
                                }, tenant_id=tenant_id)
                                if 200 <= pr_status < 300 and post_body:
                                    screening_cache[cache_key] = {"id": post_body.get("id"), "doc": post_body}
                            elif r_status == 200:
                                screening_cache[cache_key] = {"id": get_body.get("id"), "doc": get_body}
                            session["screening_cache"] = screening_cache
                        except Exception as e:
                            logger.warning("submit_answer: could not ensure screening_doc: %s", e)
                    
                    # Get screening for current exam instance
                    cached_screening = screening_cache.get(cache_key, {})
                    session["screening_id"] = cached_screening.get("id")
                    session["screening_doc"] = cached_screening.get("doc")
                    if session.get("screening_id") and session.get("screening_doc"):
                        try:
                            doc = session.get("screening_doc") or {}
                            raw_cats = doc.get("categories") or doc
                            if not isinstance(raw_cats, dict):
                                raw_cats = {}
                            # Deep copy so we send explicit payload with answer (avoid reference issues)
                            categories = json.loads(json.dumps(raw_cats))
                            cat_key = "generalQuestion" if round_num == 1 else "positionSpecificQuestion"
                            if cat_key not in categories:
                                categories[cat_key] = {}
                            cat = categories[cat_key]
                            if "conversationSets" not in cat:
                                cat["conversationSets"] = {}
                            conv_sets = cat["conversationSets"]
                            flat_index = int(question_id) if isinstance(question_id, (int, float)) else (int(question_id) if isinstance(question_id, str) and question_id.isdigit() else 0)
                            conv_key, pair_idx = flat_index_to_conv_and_pair(doc, round_num, flat_index)
                            if conv_key is None:
                                conv_key = f"conversationQuestion{flat_index + 1}"
                                pair_idx = 0
                            if conv_key in conv_sets and isinstance(conv_sets[conv_key], list) and pair_idx is not None and pair_idx < len(conv_sets[conv_key]):
                                conv_sets[conv_key][pair_idx]["answer"] = answer or ""
                                put_url = f"{cand_url}/candidate/assessment-summary"
                                tenant_id = session.get("tenant_id") or ""
                                r_status, resp_body = await candidate_put_interview(
                                    cand_url, session["screening_id"], {"candidateId": session.get("candidate_id"), "positionId": session.get("position_id"), "categories": categories}, tenant_id=tenant_id
                                )
                                _log_ws_api("PATCH assessment-summary (submit_answer)", "PATCH", put_url, {"categories": "..."}, r_status, resp_body)
                                if 200 <= r_status < 300:
                                    answer_saved_ok = True
                                    print(f"\n[ANSWER SAVED] round={round_str} question_id={question_id} flat={flat_index} (screening)\n", flush=True)
                                    try:
                                        get_url = f"{cand_url}/candidate/assessment-summary"
                                        tenant_id = session.get("tenant_id") or ""
                                        get_status, fresh_doc = await candidate_get_interview(
                                            cand_url,
                                            session.get("candidate_id"),
                                            session.get("position_id"),
                                            session.get("question_set_id") or "",
                                            tenant_id=tenant_id
                                        )
                                        if get_status == 200:
                                            # Update both cache and session
                                            screening_cache[cache_key] = {"id": fresh_doc.get("id"), "doc": fresh_doc}
                                            session["screening_cache"] = screening_cache
                                            session["screening_doc"] = fresh_doc
                                            _log_ws_api("GET assessment-summary (after save)", "GET", get_url, None, get_status, fresh_doc)
                                        else:
                                            # Update both cache and session
                                            screening_cache[cache_key] = {"id": resp_body.get("id"), "doc": resp_body}
                                            session["screening_cache"] = screening_cache
                                            session["screening_doc"] = resp_body
                                    except Exception as get_err:
                                        logger.warning("submit_answer: GET after PUT failed, using PUT response: %s", get_err)
                                        # Update both cache and session
                                        screening_cache[cache_key] = {"id": resp_body.get("id"), "doc": resp_body}
                                        session["screening_cache"] = screening_cache
                                        session["screening_doc"] = resp_body
                                else:
                                    logger.warning("submit_answer PUT failed: %s %s", r_status, resp_body)
                                    print(f"\n[ANSWER SAVE FAILED] PUT %s round={round_str} question_id={question_id}\n", r_status, flush=True)
                            else:
                                logger.warning("submit_answer: conv_key=%s pair_idx=%s not valid in screening_doc", conv_key, pair_idx)
                        except Exception as e:
                            _log_ws_api("submit_answer (screening) FAILED", "PATCH", f"{cand_url or ''}/candidate/assessment-summary", {}, 0, {"error": str(e)})
                            logger.exception("submit_answer screening update failed: %s", e)
                            print(f"\n[ANSWER SAVE FAILED] round={round_str} question_id={question_id} error={e}\n", flush=True)
                elif cand_url and round_num not in (1, 2):
                    # Round 3/4: use question-answers (or coding/aptitude APIs later)
                    payload = {
                        "clientId": session.get("client_id"),
                        "candidateId": session.get("candidate_id"),
                        "positionId": session.get("position_id"),
                        "questionSetId": session.get("question_set_id") or "",
                        "assessmentSummaryId": session.get("assessment_summary_id") or "",
                        "round": round_str,
                        "questionId": question_id,
                        "answerText": answer or "",
                    }
                    # Round 4 (aptitude): look up the correct answer from session cache so it gets persisted with the answer
                    if round_num == 4:
                        _qset_id = session.get("question_set_id") or ""
                        _cid = session.get("candidate_id")
                        _pid = session.get("position_id")
                        _cache_key = f"aptitude_qs_{_qset_id}_{_cid}_{_pid}"
                        _apt_qs = session.get(_cache_key) or []
                        _correct = ""
                        for _q in _apt_qs:
                            if str(_q.get("id")) == str(question_id):
                                _correct = str(_q.get("correctAnswer") or "")
                                break
                        if _correct:
                            payload["correctAnswer"] = _correct
                    url = f"{cand_url.rstrip('/')}/public/question-answers"
                    try:
                        tenant_id = session.get("tenant_id") or ""
                        r_status, resp_body = await candidate_post_question_answer(cand_url, payload, tenant_id=tenant_id)
                        _log_ws_api("POST question-answers (submit_answer)", "POST", url, payload, r_status, resp_body)
                        if 200 <= r_status < 300:
                            answer_saved_ok = True
                        else:
                            logger.warning("submit_answer persist failed: %s %s", r_status, resp_body)
                    except Exception as e:
                        _log_ws_api("POST question-answers (submit_answer) FAILED", "POST", url, payload, 0, {"error": str(e)})
                        logger.exception("submit_answer persist failed: %s", e)
                # Only send answer_saved after we actually saved (so frontend can safely proceed to next question)
                if answer_saved_ok:
                    logger.info("Answer saved: round=%s question_id=%s", round_str, question_id)
                    await websocket.send_json({"type": "answer_saved", "question_id": question_id})
                    print(f"[WS OUT] sent answer_saved (question_id={question_id})\n", flush=True)
                    # Round 1 & 2: next question or all_questions_answered; optionally generate cross-question
                    if round_num in (1, 2) and session.get("screening_doc"):
                        fresh_doc = session["screening_doc"]
                        flat = build_flat_qa_list(fresh_doc, round_num)
                        # Explicit next index = current + 1 so we always advance after save (no reliance on doc answer state)
                        _flat_index = int(question_id) if isinstance(question_id, (int, float)) else (int(question_id) if isinstance(question_id, str) and str(question_id).isdigit() else 0)
                        if _flat_index + 1 < len(flat):
                            next_idx, next_q_text = _flat_index + 1, (flat[_flat_index + 1][2] or "")
                        else:
                            next_idx, next_q_text = None, None
                        _conv_key, _ = flat_index_to_conv_and_pair(fresh_doc, round_num, _flat_index)
                        conv_key_for_cross = _conv_key
                        cross_count = session.get("cross_question_count_general") if round_num == 1 else session.get("cross_question_count_position")
                        cross_count = min(4, max(1, int(cross_count or 2)))
                        if session.get("is_conversational") and conv_key_for_cross and cand_url and session.get("screening_id"):
                            followups = followup_count_for_conv_key(fresh_doc, round_num, conv_key_for_cross)
                            if followups < cross_count:
                                flat_cross = build_flat_qa_list(fresh_doc, round_num)
                                current_q = ""
                                history_list = []
                                if _flat_index < len(flat_cross):
                                    current_q = flat_cross[_flat_index][2]
                                    for _ck, _pi, qq, _ in flat_cross:
                                        if _ck == conv_key_for_cross:
                                            history_list.append(qq)
                                new_q = await generate_cross_question(current_q or "", answer or "", history_list, tenant_id=session.get("tenant_id", ""))
                                if new_q:
                                    categories_cross = json.loads(json.dumps(fresh_doc.get("categories") or {}))
                                    cat_c = categories_cross.get("generalQuestion" if round_num == 1 else "positionSpecificQuestion") or {}
                                    conv_c = cat_c.get("conversationSets") or {}
                                    if conv_key_for_cross in conv_c and isinstance(conv_c[conv_key_for_cross], list):
                                        # Inherit timing from the parent question (same conv_key, first pair)
                                        _sa_t_list = session.get(f"q_timing_{session.get('question_set_id', '')}_{round_num}") or []
                                        _sa_ci = sort_conv_key(conv_key_for_cross) - 1
                                        _sa_at, _sa_pt = _sa_t_list[_sa_ci] if 0 <= _sa_ci < len(_sa_t_list) else (120, 5)
                                        conv_c[conv_key_for_cross].append({"question": new_q, "answer": "", "answerTime": _sa_at, "prepareTime": _sa_pt})
                                        put_url_c = f"{cand_url}/candidate/assessment-summary"
                                        r_c_status, _ = await candidate_put_interview(
                                            cand_url, session["screening_id"], {"candidateId": session.get("candidate_id"), "positionId": session.get("position_id"), "categories": categories_cross}, tenant_id=tenant_id
                                        )
                                        _log_ws_api("PATCH assessment-summary (append cross-question)", "PATCH", put_url_c, {"cross_question": new_q[:80]}, r_c_status, {})
                                        if 200 <= r_c_status < 300:
                                            try:
                                                get_status2, fresh_doc2 = await candidate_get_interview(
                                                    cand_url,
                                                    session.get("candidate_id"),
                                                    session.get("position_id"),
                                                    session.get("question_set_id") or "",
                                                    tenant_id=tenant_id
                                                )
                                                if get_status2 == 200:
                                                    # Update both cache and session
                                                    cid_cross = session.get("candidate_id")
                                                    pid_cross = session.get("position_id")
                                                    qset_id_cross = session.get("question_set_id")
                                                    assessment_summary_id_cross = session.get("assessment_summary_id") or ""
                                                    cache_key_cross = f"{pid_cross}_{cid_cross}_{qset_id_cross}_{assessment_summary_id_cross}"
                                                    screening_cache[cache_key_cross] = {"id": fresh_doc2.get("id"), "doc": fresh_doc2}
                                                    session["screening_cache"] = screening_cache
                                                    session["screening_doc"] = fresh_doc2
                                            except Exception as get_ex:
                                                logger.warning("GET after cross-question PUT failed: %s", get_ex)
                                            next_idx, next_q_text = _flat_index + 1, new_q
                        if next_idx is not None:
                            # Compute timing for the next question
                            _nq_at, _nq_pt = 120, 5
                            _nq_doc = session.get("screening_doc") or fresh_doc or {}
                            _nq_flat = build_flat_qa_list(_nq_doc, round_num)
                            if 0 <= next_idx < len(_nq_flat):
                                _nq_ck = _nq_flat[next_idx][0]
                                _nq_t_list = session.get(f"q_timing_{session.get('question_set_id', '')}_{round_num}") or []
                                _nq_ci = sort_conv_key(_nq_ck) - 1
                                if 0 <= _nq_ci < len(_nq_t_list):
                                    _nq_at, _nq_pt = _nq_t_list[_nq_ci]
                            await websocket.send_json({
                                "type": "next_question",
                                "round": round_num,
                                "question_id": str(next_idx),
                                "question_index": next_idx,
                                "question": next_q_text,
                                "timeToAnswer": _nq_at,
                                "timeToPrepare": _nq_pt,
                            })
                            print(f"[WS OUT] sent next_question round={round_num} index={next_idx}\n", flush=True)
                        else:
                            await websocket.send_json({
                                "type": "all_questions_answered",
                                "round": round_num,
                            })
                            print(f"[WS OUT] sent all_questions_answered round={round_num}\n", flush=True)
                else:
                    await websocket.send_json({"type": "error", "message": "Failed to save answer. Please try again.", "question_id": question_id})
                    print(f"[WS OUT] sent error (answer not saved, question_id={question_id})\n", flush=True)
            elif msg_type == "get_round_questions":
                round_num = msg.get("round")
                if round_num is None:
                    await websocket.send_json({"type": "error", "message": "round required"})
                    continue
                rn = int(round_num) if isinstance(round_num, (int, float)) else (int(round_num) if isinstance(round_num, str) and str(round_num).isdigit() else 0)
                qset_id = session.get("question_set_id")
                tenant_id = session.get("tenant_id") or ""
                admin_url = getattr(config, "ADMIN_BACKEND_URL", None)
                cand_url = getattr(config, "CANDIDATE_BACKEND_URL", None)
                cid = session.get("candidate_id")
                pid = session.get("position_id")
                # Round 1 & 2: use screening document (ensure we have it, then return questions in order with answer field)
                if rn in (1, 2) and cand_url and cid and pid and qset_id:
                    try:
                        # Use ir_cache (session-level) to avoid repeated GETs.
                        # ir_cache is also updated on every answer PATCH in submit_and_next,
                        # so get_round_questions always sees fresh answers after a submit.
                        ir_cache_key = f"{pid}_{cid}"
                        ir_cache = session.get("ir_cache") or {}

                        logger.info(f"[Round {rn}] get_round_questions: cached={ir_cache_key in ir_cache}")

                        if ir_cache_key not in ir_cache:
                            r_status, get_body = await ir_get(cand_url, cid, pid, tenant_id=tenant_id)
                            logger.info(f"[Round {rn}] GET interview-responses: status={r_status}, has_categories={isinstance((get_body or {}).get('categories'), dict)}")
                            if r_status == 404 and admin_url:
                                # First visit — build empty categories from admin, POST to MongoDB
                                sr_status, sr_json = await admin_get_question_sections(admin_url, qset_id, tenant_id)
                                raw = (sr_json.get("data") or []) if sr_status == 200 else []
                                section_row = raw[0] if isinstance(raw, list) and raw else {}
                                categories = build_screening_categories_from_sections(section_row)
                                pr_status, post_body = await ir_post(cand_url, {
                                    "candidateId": cid, "positionId": pid, "questionSetId": qset_id,
                                    "categories": categories,
                                }, tenant_id=tenant_id)
                                logger.info(f"[Round {rn}] POST interview-responses: status={pr_status}")
                                if 200 <= pr_status < 300 and post_body:
                                    ir_cache[ir_cache_key] = post_body
                            elif r_status == 200:
                                ir_cache[ir_cache_key] = get_body
                            session["ir_cache"] = ir_cache

                        doc = ir_cache.get(ir_cache_key) or {}
                        flat_qa = build_flat_qa_list(doc, rn)
                        logger.info(f"[Round {rn}] flat_qa: count={len(flat_qa)}, answered={sum(1 for *_, a in flat_qa if a.strip())}")
                        
                        # Enrich each question with timeToPrepare / timeToAnswer from Admin question-sections
                        _t_cache_key = f"q_timing_{qset_id}_{rn}"
                        _t_list = session.get(_t_cache_key)
                        if _t_list is None and admin_url and qset_id:
                            try:
                                _ts, _tj = await admin_get_question_sections(admin_url, qset_id, tenant_id)
                                if _ts == 200 and _tj:
                                    _ts_d = (_tj.get("data") or [])
                                    _ts_d = _ts_d[0] if isinstance(_ts_d, list) and _ts_d else (_ts_d if isinstance(_ts_d, dict) else {})
                                    _ts_qs = questions_for_round(_ts_d, rn) or []
                                    _t_list = [
                                        (
                                            parse_seconds((_q or {}).get("answerTime", (_q or {}).get("timeToAnswer")), 120)
                                            if isinstance(_q, dict)
                                            else 120,
                                            parse_seconds((_q or {}).get("prepareTime", (_q or {}).get("timeToPrepare")), 10)
                                            if isinstance(_q, dict)
                                            else 10,
                                        )
                                        for _q in _ts_qs
                                    ]
                                    session[_t_cache_key] = _t_list
                            except Exception as _te:
                                logger.warning("get_round_questions: timing fetch failed: %s", _te)
                        if not _t_list:
                            _t_list = []
                        questions_ordered = []
                        for _ck, _pi, q, a in flat_qa:
                            _ci = sort_conv_key(_ck) - 1
                            _at, _pt = _t_list[_ci] if 0 <= _ci < len(_t_list) else (120, 5)
                            questions_ordered.append({"question": q, "answer": a, "timeToAnswer": _at, "timeToPrepare": _pt})

                        # Fallback: if screening fetch failed or empty, serve questions directly from Admin question-sections
                        if len(questions_ordered) == 0 and admin_url:
                            logger.info("get_round_questions: questions_ordered is empty, attempting fallback to Admin question-sections (qset_id=%s)", qset_id)
                            try:
                                fb_status, fb_json = await admin_get_question_sections(admin_url, qset_id, tenant_id or "")
                                logger.info("get_round_questions: Admin fallback status=%s data_keys=%s", fb_status, list(fb_json.get("data", [{}])[0].keys()) if fb_json.get("data") else "none")
                                if fb_status == 200 and fb_json:
                                    fb_raw = fb_json.get("data") or {}
                                    fb_data = fb_raw[0] if isinstance(fb_raw, list) and len(fb_raw) > 0 else fb_raw
                                    fb_questions = questions_for_round(fb_data, rn) or []
                                    logger.info("get_round_questions: Extracted %s questions from fallback data", len(fb_questions))
                                    mapped = []
                                    for i, item in enumerate(fb_questions):
                                        _fb_at = parse_seconds(item.get("answerTime", item.get("timeToAnswer")), 120) if isinstance(item, dict) else 120
                                        _fb_pt = parse_seconds(item.get("prepareTime", item.get("timeToPrepare")), 10) if isinstance(item, dict) else 10
                                        if 0 <= i < len(_t_list): _fb_at, _fb_pt = _t_list[i]
                                        q_text = (item.get("question") or "").strip() if isinstance(item, dict) else str(item).strip()
                                        mapped.append({"question": q_text, "answer": "", "timeToAnswer": _fb_at, "timeToPrepare": _fb_pt})
                                    questions_ordered = mapped
                            except Exception as fb_err:
                                logger.warning("get_round_questions fallback failed: %s", fb_err)

                        # Calculate total round time from the questions if it's missing or zero
                        total_time_sec = sum((q.get("timeToAnswer", 120) + q.get("timeToPrepare", 5)) for q in questions_ordered)
                        mm, ss = divmod(total_time_sec, 60)
                        hh, mm = divmod(mm, 60)
                        total_time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

                        await websocket.send_json({
                            "type": "round_questions", 
                            "round": round_num, 
                            "questions": questions_ordered,
                            "totalTime": total_time_str
                        })
                        print(f"[WS OUT] sent round_questions (round={round_num}, count={len(questions_ordered)}, time={total_time_str})\n", flush=True)
                    except Exception as e:
                        logger.exception("get_round_questions screening failed: %s", e)
                        await websocket.send_json({"type": "round_questions", "round": round_num, "questions": []})
                    continue
                if not admin_url or not qset_id:
                    await websocket.send_json({"type": "round_questions", "round": round_num, "questions": []})
                    continue
                try:
                    url = f"{admin_url.rstrip('/')}/internal/question-sections/question-set/{qset_id}"
                    payload = {"round": round_num, "headers": {"X-Tenant-Id": tenant_id or ""}}
                    r_status, resp_json = await admin_get_question_sections(admin_url, qset_id, tenant_id or "")
                    _log_ws_api("GET question-sections (get_round_questions)", "GET", url, payload, r_status, resp_json)
                    if r_status != 200:
                        logger.warning("get_round_questions failed: %s %s", r_status, resp_json)
                        await websocket.send_json({"type": "round_questions", "round": round_num, "questions": []})
                        continue
                    raw = resp_json.get("data") or {}
                    data = raw[0] if isinstance(raw, list) and len(raw) > 0 else raw
                except Exception as e:
                    _log_ws_api("GET question-sections (get_round_questions) FAILED", "GET", url, payload, 0, {"error": str(e)})
                    logger.exception("get_round_questions failed: %s", e)
                    await websocket.send_json({"type": "round_questions", "round": round_num, "questions": []})
                    continue
                    # ── Round 4 (Aptitude): generate MCQ questions via AI ──────────────
                    if rn == 4:
                        cache_key = f"aptitude_qs_{qset_id}_{cid}_{pid}"
                        cached_qs = session.get(cache_key)
                        if cached_qs:
                            print(f"[Round 4] ✅ session cache hit — returning {len(cached_qs)} questions\n", flush=True)
                            await websocket.send_json({"type": "round_questions", "round": round_num, "questions": cached_qs})
                            continue

                        # ── Step 1: Fetch from MongoDB (skip AI if already saved) ────
                        generated = None
                        _cand_url_r4 = getattr(config, "CANDIDATE_BACKEND_URL", None)
                        if _cand_url_r4 and cid and pid and qset_id:
                            _fetch_url = f"{_cand_url_r4.rstrip('/')}/candidate-aptitude-responses/generated-questions"
                            print(f"[Round 4] checking MongoDB: GET {_fetch_url} candidateId={cid} positionId={pid} questionSetId={qset_id}\n", flush=True)
                            try:
                                async with httpx.AsyncClient(timeout=10.0) as _cl:
                                    _r = await _cl.get(_fetch_url, params={
                                        "candidateId": cid,
                                        "positionId": pid,
                                        "questionSetId": qset_id,
                                        "tenantId": tenant_id
                                    })
                                print(f"[Round 4] MongoDB fetch status: {_r.status_code}\n", flush=True)
                                if _r.status_code == 200:
                                    _body = _r.json()
                                    _db_qs = _body.get("questions") if isinstance(_body, dict) else None
                                    if _db_qs and isinstance(_db_qs, list) and len(_db_qs) > 0:
                                        generated = _db_qs
                                        print(f"[Round 4] ✅ loaded {len(generated)} questions from MongoDB — NO AI call needed\n", flush=True)
                                    else:
                                        print(f"[Round 4] MongoDB returned 200 but no questions — will generate\n", flush=True)
                                else:
                                    print(f"[Round 4] MongoDB 404/error (not yet saved) — will generate\n", flush=True)
                            except Exception as _e:
                                print(f"[Round 4] ⚠️ MongoDB fetch failed: {_e} — will generate\n", flush=True)
                        else:
                            print(f"[Round 4] ⚠️ missing cid/pid/qset_id (cid={cid}, pid={pid}, qset={qset_id}) — skipping DB check\n", flush=True)

                        # ── Step 2: AI generation only if not found in DB ────────────
                        if generated is None:
                            topic_blocks = questions_for_round(data, 4) or []
                            print(f"[Round 4] generating MCQ questions for {len(topic_blocks)} topic block(s)...\n", flush=True)
                            generated = await generate_aptitude_questions(topic_blocks)
                            print(f"[Round 4] ✅ AI generated {len(generated)} questions\n", flush=True)

                            # ── Step 3: Save to MongoDB so refresh fetches instead of re-generating ──
                            if _cand_url_r4 and generated and cid and pid and qset_id:
                                _upsert_url = f"{_cand_url_r4.rstrip('/')}/candidate-aptitude-responses/upsert"
                                print(f"[Round 4] saving to MongoDB: POST {_upsert_url}\n", flush=True)
                                try:
                                    async with httpx.AsyncClient(timeout=15.0) as _cl:
                                        _save_r = await _cl.post(_upsert_url, json={
                                            "candidateId": cid,
                                            "positionId": pid,
                                            "questionSetId": qset_id,
                                            "totalQuestions": len(generated),
                                            "generatedQuestions": generated,
                                            "tenantId": tenant_id
                                        })
                                    if 200 <= _save_r.status_code < 300:
                                        print(f"[Round 4] ✅ saved {len(generated)} questions to MongoDB (status {_save_r.status_code})\n", flush=True)
                                    else:
                                        print(f"[Round 4] ❌ MongoDB save FAILED — status {_save_r.status_code}: {_save_r.text[:300]}\n", flush=True)
                                except Exception as _e:
                                    print(f"[Round 4] ❌ MongoDB save EXCEPTION: {_e}\n", flush=True)
                            else:
                                print(f"[Round 4] ⚠️ skipping MongoDB save — missing cid/pid/qset: cid={cid} pid={pid} qset={qset_id}\n", flush=True)

                        session[cache_key] = generated
                        await websocket.send_json({"type": "round_questions", "round": round_num, "questions": generated})
                        print(f"[WS OUT] sent round_questions (round=4, count={len(generated)})\n", flush=True)
                        continue
                    # ── Rounds 3: return raw questions as-is ─────────────────────────
                    questions = questions_for_round(data, rn)
                    await websocket.send_json({"type": "round_questions", "round": round_num, "questions": questions})
                    print(f"[WS OUT] sent round_questions (round={round_num}, count={len(questions)})\n", flush=True)
                except Exception as e:
                    _log_ws_api("GET question-sections (get_round_questions) FAILED", "GET", url, payload, 0, {"error": str(e)})
                    logger.exception("get_round_questions failed: %s", e)
                    await websocket.send_json({"type": "round_questions", "round": round_num, "questions": []})
            elif msg_type == "get_saved_answers":
                round_param = msg.get("round")
                cand_url = getattr(config, "CANDIDATE_BACKEND_URL", None)
                if not cand_url:
                    await websocket.send_json({"type": "saved_answers", "round": round_param, "answers": []})
                    continue
                params = {
                    "clientId": session.get("client_id"),
                    "candidateId": session.get("candidate_id"),
                    "positionId": session.get("position_id"),
                    "questionSetId": session.get("question_set_id") or "",
                    "assessmentSummaryId": session.get("assessment_summary_id") or "",
                }
                if round_param is not None:
                    params["round"] = str(round_param)
                url = f"{cand_url.rstrip('/')}/public/question-answers"
                print(f"[WS] Calling CandidateBackend: GET {url} (params: {params})\n", flush=True)
                try:
                    r_status, body = await candidate_get_question_answers(cand_url, params, tenant_id=tenant_id)
                    _log_ws_api("GET question-answers (get_saved_answers)", "GET", url, {"params": params}, r_status, body)
                    if r_status == 404:
                        await websocket.send_json({"type": "saved_answers", "round": round_param, "answers": []})
                        continue
                    if r_status != 200:
                        await websocket.send_json({"type": "saved_answers", "round": round_param, "answers": []})
                        continue
                    answers = (body.get("data") or body.get("answers") or [])
                    await websocket.send_json({"type": "saved_answers", "round": round_param, "answers": answers})
                    print(f"[WS OUT] sent saved_answers (round={round_param}, count={len(answers)})\n", flush=True)
                except Exception as e:
                    _log_ws_api("GET question-answers (get_saved_answers) FAILED", "GET", url, {"params": params}, 0, {"error": str(e)})
                    logger.exception("get_saved_answers failed: %s", e)
                    await websocket.send_json({"type": "saved_answers", "round": round_param, "answers": []})
            elif msg_type == "get_next_question":
                logger.info("get_next_question (stub)")
                await websocket.send_json({"type": "question", "question_id": None, "text": "", "round": "", "done": True})
            elif msg_type == "proctoring_event":
                logger.info("Proctoring: %s", msg.get("event"))
            elif msg_type == "proctoring_frame":
                # Backend runs face detection; send result so client can show alert
                image_b64 = msg.get("image_base64")
                if not image_b64:
                    await websocket.send_json({"type": "proctoring_result", "event": "ok", "confidence": 0.0})
                    continue
                try:
                    result = await asyncio.to_thread(face_detection_analyze, image_b64)
                    event = result.get("event", "ok")
                    confidence = float(result.get("confidence", 0))
                    logger.info("Proctoring result: %s (confidence=%.2f)", event, confidence)
                    
                    # Automated violation screenshot storage
                    if event != "ok":
                        try:
                            client_id = session.get("client_id")
                            position_id = session.get("position_id")
                            candidate_id = session.get("candidate_id")
                            if client_id and position_id and candidate_id:
                                # Save screenshot to a subfolder named after the event
                                dir_path = _screenshot_path(client_id, position_id, candidate_id) / event
                                dir_path.mkdir(parents=True, exist_ok=True)
                                
                                # Decode the image for local saving
                                header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
                                img_bytes = base64.b64decode(encoded)
                                
                                name = f"violation_{event}_{int(time.time() * 1000)}.png"
                                file_path = dir_path / name
                                with open(file_path, "wb") as f:
                                    f.write(img_bytes)
                                logger.info("Violation screenshot saved: %s", name)
                        except Exception as ss_err:
                            logger.error("Failed to save violation screenshot: %s", ss_err)

                    await websocket.send_json({
                        "type": "proctoring_result",
                        "event": event,
                        "confidence": confidence,
                    })
                except Exception as e:
                    logger.exception("proctoring_frame analysis failed: %s", e)
                    await websocket.send_json({"type": "proctoring_result", "event": "ok", "confidence": 0.0})
            elif msg_type == "start_listening":
                # AssemblyAI real-time STT via SDK (same as KareerGrowth AiService) – no token API, avoids 422
                if session.get("assemblyai_runner") is not None:
                    await websocket.send_json({"type": "transcript", "error": "Already listening"})
                    continue
                api_key = (config.ASSEMBLYAI_API_KEY or "").strip()
                if not api_key:
                    await websocket.send_json({"type": "transcript", "error": "AssemblyAI not configured"})
                    continue
                try:
                    loop = asyncio.get_running_loop()
                    runner = AssemblyAIStreamRunner(api_key, loop, lambda msg: websocket.send_json(msg))
                    runner.start(client_sample_rate=16000)
                    session["assemblyai_runner"] = runner
                    print("[WS] start_listening: AssemblyAI (SDK) connected, sending transcripts on same WebSocket\n", flush=True)
                    await websocket.send_json({"type": "listening_started"})
                except Exception as e:
                    logger.exception("start_listening failed: %s", e)
                    await websocket.send_json({"type": "transcript", "error": str(e)})
            elif msg_type == "speak":
                text = msg.get("text")
                voice = msg.get("voice", "en-US-AvaMultilingualNeural")
                if not text:
                    continue
                try:
                    communicate = edge_tts.Communicate(text, voice)
                    audio_data = b""
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]
                    
                    if audio_data:
                        b64_audio = base64.b64encode(audio_data).decode("utf-8")
                        await websocket.send_json({
                            "type": "tts_audio",
                            "audio": b64_audio,
                            "text": text
                        })
                        print(f"[WS OUT] sent tts_audio ({len(audio_data)} bytes) for text: {text[:50]}...\n", flush=True)
                except Exception as e:
                    logger.error("WebSocket TTS failed: %s", e)
                    await websocket.send_json({"type": "tts_error", "message": str(e)})

            elif msg_type == "audio_chunk":
                # PCM audio (base64) → forward to AssemblyAI runner
                runner = session.get("assemblyai_runner")
                if runner is None:
                    continue
                try:
                    b64 = msg.get("data") or msg.get("audio")
                    if b64:
                        raw = base64.b64decode(b64)
                        runner.put_audio(raw)
                except Exception as e:
                    logger.warning("audio_chunk forward: %s", e)
            elif msg_type is None or msg_type == "":
                # Malformed or empty type; avoid "Unknown type: None"
                continue
            elif msg_type == "stop_listening":
                runner = session.get("assemblyai_runner")
                if runner:
                    runner.stop()
                    session["assemblyai_runner"] = None
                session["assemblyai_ws"] = None
                session["assemblyai_reader_task"] = None
                print("[WS] stop_listening: AssemblyAI disconnected\n", flush=True)
                await websocket.send_json({"type": "listening_stopped"})
            elif msg_type == "submit_and_next":
                # Submit answer and get next question via WebSocket (replaces HTTP /api/submit-answer-and-get-next)
                round_num = msg.get("round")
                question_id = msg.get("questionId")
                answer = msg.get("answer", "")
                is_conversational = msg.get("isConversational", False)
                # Use AUTHORITATIVE DB values from session (set during init), NOT frontend message parameters
                # This ensures the admin-configured cross question limits are enforced exactly
                cross_general = session.get("cross_question_count_general", 2)
                cross_position = session.get("cross_question_count_position", 2)
                
                logger.info(f"[WS] submit_and_next: Round {round_num} | Q#{question_id} | isConversational={is_conversational} | crossGeneral={cross_general} crossPosition={cross_position}")
                
                if round_num not in (1, 2):
                    await websocket.send_json({"type": "error", "message": "round must be 1 or 2"})
                    continue
                
                cand_url = getattr(config, "CANDIDATE_BACKEND_URL", None)
                admin_url = getattr(config, "ADMIN_BACKEND_URL", None)
                if not cand_url:
                    await websocket.send_json({"type": "error", "message": "CANDIDATE_BACKEND_URL not configured"})
                    continue
                
                cid = session.get("candidate_id")
                pid = session.get("position_id")
                qset_id = session.get("question_set_id")
                tenant_id = session.get("tenant_id", "")
                
                flat_index = int(question_id) if str(question_id).isdigit() else 0

                # Session ir_cache: avoid extra GETs; updated after every PATCH
                ir_cache_key = f"{pid}_{cid}"
                ir_cache = session.get("ir_cache") or {}

                try:
                    # 1) Ensure interview-responses doc is loaded (MongoDB via CandidateBackend)
                    doc = ir_cache.get(ir_cache_key)
                    if doc is None:
                        r_status, get_body = await ir_get(cand_url, cid, pid, tenant_id=tenant_id)
                        if r_status == 200:
                            doc = get_body
                        else:
                            # 404 or error — build from admin questions and create in MongoDB
                            categories_new = {}
                            if admin_url:
                                try:
                                    sr_status, sr_json = await admin_get_question_sections(admin_url, qset_id, tenant_id)
                                    if sr_status == 200 and sr_json:
                                        raw = (sr_json.get("data") or [])
                                        section_row = raw[0] if isinstance(raw, list) and raw else {}
                                        categories_new = build_screening_categories_from_sections(section_row)
                                except Exception:
                                    pass
                            pr_status, post_body = await ir_post(cand_url, {
                                "candidateId": cid, "positionId": pid, "questionSetId": qset_id,
                                "categories": categories_new,
                            }, tenant_id=tenant_id)
                            if 200 <= pr_status < 300 and post_body:
                                doc = post_body
                            else:
                                # MongoDB unavailable — work fully in-memory for this session
                                doc = {"candidateId": cid, "positionId": pid, "questionSetId": qset_id,
                                       "categories": categories_new, "isScreeningCompleted": False}
                        ir_cache[ir_cache_key] = doc
                        session["ir_cache"] = ir_cache

                    cat_key = "generalQuestion" if round_num == 1 else "positionSpecificQuestion"

                    # 2) Resolve conv_key / pair_idx from flat_index
                    conv_key, pair_idx = flat_index_to_conv_and_pair(doc, round_num, flat_index)
                    if conv_key is None:
                        conv_key = f"conversationQuestion{flat_index + 1}"
                        pair_idx = 0

                    # 3) Persist answer — targeted PATCH (only updates this one Q&A pair)
                    patch_status, patch_body = await ir_patch(cand_url, {
                        "candidateId": cid, "positionId": pid,
                        "round": round_num, "convKey": conv_key,
                        "pairIdx": pair_idx, "answer": answer,
                    }, tenant_id=tenant_id)

                    if 200 <= patch_status < 300 and patch_body:
                        doc = patch_body
                        logger.info(f"[WS] Answer saved in MongoDB: Round {round_num} Q#{question_id} conv_key={conv_key} pair_idx={pair_idx}")
                    else:
                        # MongoDB unavailable — update in-memory doc so flow continues
                        logger.warning(f"[WS] ir_patch failed ({patch_status}), updating in-memory doc only")
                        doc = json.loads(json.dumps(doc))   # deep copy
                        _cats = doc.get("categories") or {}
                        _cat = _cats.get(cat_key) or {}
                        _conv_sets = _cat.get("conversationSets") or {}
                        if conv_key in _conv_sets and isinstance(_conv_sets[conv_key], list) and pair_idx < len(_conv_sets[conv_key]):
                            _conv_sets[conv_key][pair_idx]["answer"] = answer

                    ir_cache[ir_cache_key] = doc
                    session["ir_cache"] = ir_cache

                    # 4) Determine next question index from updated doc
                    flat = build_flat_qa_list(doc, round_num)
                    next_idx = flat_index + 1
                    next_q_text = ""
                    all_done = next_idx >= len(flat)
                    _resp_next_at, _resp_next_pt = 120, 5
                    _timing_set_by_cross = False

                    # 5) Cross-question generation (conversational mode, under per-main-Q limit)
                    conv_key_for_cross = conv_key
                    cross_count = min(4, max(1, int(cross_general if round_num == 1 else cross_position)))
                    if is_conversational and conv_key_for_cross:
                        followups = followup_count_for_conv_key(doc, round_num, conv_key_for_cross)
                        logger.info(f"[WS CROSS-Q] followups={followups}/{cross_count} conv_key={conv_key_for_cross}")
                        if followups < cross_count:
                            flat_cross = build_flat_qa_list(doc, round_num)
                            current_q = flat_cross[flat_index][2] if flat_index < len(flat_cross) else ""
                            history_list = [qq for _ck, _pi, qq, _ in flat_cross if _ck == conv_key_for_cross]
                            try:
                                new_q = await generate_cross_question(current_q or "", answer or "", history_list, tenant_id=session.get("tenant_id", ""))
                                if new_q:
                                    _san_t_list = session.get(f"q_timing_{qset_id}_{round_num}") or []
                                    _san_ci = sort_conv_key(conv_key_for_cross) - 1
                                    _san_at, _san_pt = _san_t_list[_san_ci] if 0 <= _san_ci < len(_san_t_list) else (120, 5)
                                    # Append cross-question pair to MongoDB
                                    cq_status, cq_body = await ir_patch(cand_url, {
                                        "candidateId": cid, "positionId": pid,
                                        "round": round_num, "convKey": conv_key_for_cross,
                                        "appendQuestion": new_q,
                                        "appendAnswerTime": _san_at, "appendPrepareTime": _san_pt,
                                    }, tenant_id=tenant_id)
                                    if 200 <= cq_status < 300 and cq_body:
                                        doc = cq_body
                                    else:
                                        # In-memory fallback
                                        doc = json.loads(json.dumps(doc))
                                        _cats_c = doc.get("categories") or {}
                                        _cat_c = _cats_c.get(cat_key) or {}
                                        _conv_c = _cat_c.get("conversationSets") or {}
                                        if conv_key_for_cross in _conv_c and isinstance(_conv_c[conv_key_for_cross], list):
                                            _conv_c[conv_key_for_cross].append({"question": new_q, "answer": "", "answerTime": _san_at, "prepareTime": _san_pt})
                                    ir_cache[ir_cache_key] = doc
                                    session["ir_cache"] = ir_cache
                                    # Rebuild flat list to include the new cross-question
                                    flat = build_flat_qa_list(doc, round_num)
                                    if flat_index + 1 < len(flat):
                                        next_idx = flat_index + 1
                                        next_q_text = flat[next_idx][2] or new_q
                                        all_done = False
                                        _resp_next_at = _san_at
                                        _resp_next_pt = _san_pt
                                        _timing_set_by_cross = True
                                        logger.info(f"[WS CROSS-Q] Generated and persisted: '{new_q[:80]}'")
                                else:
                                    logger.warning("[WS CROSS-Q] generate_cross_question returned empty")
                            except Exception as _cq_err:
                                logger.warning("[WS CROSS-Q] Failed: %s", _cq_err)

                    if not next_q_text and next_idx < len(flat):
                        next_q_text = flat[next_idx][2] or ""

                    # Compute timing for next question (if not set by cross-question block)
                    if not _timing_set_by_cross and not all_done and next_idx < len(flat):
                        try:
                            _rn_ck = flat[next_idx][0]
                            _rn_t = session.get(f"q_timing_{qset_id}_{round_num}") or []
                            _rn_ci = sort_conv_key(_rn_ck) - 1
                            if 0 <= _rn_ci < len(_rn_t):
                                _resp_next_at, _resp_next_pt = _rn_t[_rn_ci]
                        except Exception:
                            pass

                    response_data = {
                        "type": "submit_and_next_response",
                        "success": True,
                        "answerSaved": True,
                        "nextQuestionIndex": next_idx if not all_done else None,
                        "nextQuestionText": next_q_text or None,
                        "allQuestionsAnswered": all_done,
                        "nextQuestionTimeToAnswer": _resp_next_at if not all_done else None,
                        "nextQuestionTimeToPrepare": _resp_next_pt if not all_done else None,
                    }
                    logger.info(f"[WS OUT] submit_and_next_response: nextIndex={response_data['nextQuestionIndex']} | allDone={all_done}")
                    await websocket.send_json(response_data)
                    
                except Exception as e:
                    logger.exception("[WS] submit_and_next failed: %s", e)
                    await websocket.send_json({"type": "error", "message": f"Submit and next failed: {str(e)}"})

            # ─── Coding round: session init (upsert MongoDB record) ───────────────
            elif msg_type == "coding_session_init":
                cand_url = getattr(config, "CANDIDATE_BACKEND_URL", None)
                cand_id = session.get("candidate_id")
                pos_id = session.get("position_id")
                q_set_id = session.get("question_set_id", "")
                total_qs = int(msg.get("totalQuestions", 1))
                cq_sets = msg.get("codingQuestionSets", [])
                if cand_url and cand_id and pos_id:
                    try:
                        url = f"{cand_url.rstrip('/')}/candidate-coding-responses/upsert"
                        payload = {
                            "candidateId": cand_id,
                            "positionId": pos_id,
                            "questionSetId": q_set_id,
                            "totalQuestions": total_qs,
                            "codingQuestionSets": cq_sets,
                        }
                        async with httpx.AsyncClient(timeout=20.0) as cl:
                            r = await cl.post(url, json=payload)
                            resp_body = r.json() if "application/json" in r.headers.get("content-type", "") else {"raw": r.text}
                        _log_ws_api("POST candidate-coding-responses/upsert (coding_session_init)", "POST", url, payload, r.status_code, resp_body)
                        session["coding_response_id"] = resp_body.get("id") or resp_body.get("_id") if 200 <= r.status_code < 300 else None
                        await websocket.send_json({
                            "type": "coding_session_init_ok",
                            "responseId": session.get("coding_response_id"),
                            "record": resp_body if 200 <= r.status_code < 300 else None,
                            "error": None if 200 <= r.status_code < 300 else resp_body.get("message", "Upsert failed"),
                        })
                    except Exception as e:
                        logger.exception("coding_session_init failed: %s", e)
                        await websocket.send_json({"type": "coding_session_init_ok", "responseId": None, "record": None, "error": str(e)})
                else:
                    await websocket.send_json({"type": "coding_session_init_ok", "responseId": None, "record": None, "error": "Missing session IDs"})
                print(f"[WS OUT] sent coding_session_init_ok (responseId={session.get('coding_response_id')})\n", flush=True)

            # ─── Coding round: generate question via AI ────────────────────────────
            elif msg_type == "coding_generate_question":
                from coding_question import GenerateQuestionRequest, generate_coding_question as _gen_coding_q
                q_idx = int(msg.get("questionIndex", 0))
                try:
                    req_obj = GenerateQuestionRequest(
                        programmingLanguage=msg.get("programmingLanguage", "JavaScript"),
                        difficultyLevel=msg.get("difficultyLevel", "Easy"),
                        questionSource=msg.get("questionSource", "Coding Library"),
                        topicTags=msg.get("topicTags") or None,
                        questionIndex=q_idx,
                    )
                    result = await _gen_coding_q(req_obj)
                    # FastAPI endpoint returns JSONResponse; get body as dict
                    if hasattr(result, "body"):
                        question_data = json.loads(result.body)
                    elif isinstance(result, dict):
                        question_data = result
                    else:
                        question_data = result.__dict__ if hasattr(result, "__dict__") else {}
                    await websocket.send_json({
                        "type": "coding_generate_question_ok",
                        "questionIndex": q_idx,
                        "question": question_data,
                    })
                    print(f"[WS OUT] sent coding_generate_question_ok (qIdx={q_idx})\n", flush=True)
                except Exception as e:
                    logger.exception("coding_generate_question failed (qIdx=%s): %s", q_idx, e)
                    await websocket.send_json({
                        "type": "coding_generate_question_ok",
                        "questionIndex": q_idx,
                        "question": None,
                        "error": str(e),
                    })

            # ─── Coding round: save generated question content to MongoDB ──────────
            elif msg_type == "coding_save_question":
                cand_url = getattr(config, "CANDIDATE_BACKEND_URL", None)
                resp_id = msg.get("responseId") or session.get("coding_response_id")
                s_idx = int(msg.get("setIndex", 0))
                q_idx = int(msg.get("qIndex", 0))
                q_data = msg.get("questionData", {})
                if cand_url and resp_id and q_data:
                    try:
                        url = f"{cand_url.rstrip('/')}/candidate-coding-responses/{resp_id}/set/{s_idx}/question/{q_idx}/content"
                        async with httpx.AsyncClient(timeout=15.0) as cl:
                            r = await cl.put(url, json=q_data)
                            resp_body = r.json() if "application/json" in r.headers.get("content-type", "") else {"raw": r.text}
                        _log_ws_api(f"PUT coding content q{q_idx} (coding_save_question)", "PUT", url, q_data, r.status_code, resp_body)
                        await websocket.send_json({"type": "coding_save_question_ok", "qIndex": q_idx, "success": 200 <= r.status_code < 300})
                    except Exception as e:
                        logger.exception("coding_save_question failed (qIdx=%s): %s", q_idx, e)
                        await websocket.send_json({"type": "coding_save_question_ok", "qIndex": q_idx, "success": False, "error": str(e)})
                else:
                    await websocket.send_json({"type": "coding_save_question_ok", "qIndex": q_idx, "success": False, "error": "Missing responseId or questionData"})

            # ─── Coding round: complete — update assessment summary + round timing ─
            elif msg_type == "coding_round_complete":
                _cand_url = getattr(config, "CANDIDATE_BACKEND_URL", None)
                _admin_url = getattr(config, "ADMIN_BACKEND_URL", None)
                _pos_id = session.get("position_id")
                _cand_id = session.get("candidate_id")
                _tenant_id = session.get("tenant_id", "") or ""
                _round_num = int(msg.get("roundNumber", 3))
                _end_time = msg.get("roundEndTime", "")
                _start_time = msg.get("roundStartTime", "")
                _time_taken = msg.get("roundTimeTaken", "")
                _time_formatted = msg.get("roundTimeFormatted", "")
                _summary_ok = False
                _round_ok = False
                # 1) PATCH assessment-summary in CandidateBackend
                if _cand_url and _cand_id and _pos_id:
                    try:
                        url = f"{_cand_url.rstrip('/')}/candidate/assessment-summary"
                        payload = {
                            "candidateId": _cand_id,
                            "positionId": _pos_id,
                            f"round{_round_num}Completed": True,
                            f"round{_round_num}EndTime": _end_time[:19].replace("T", " ") if _end_time else "",
                            f"round{_round_num}TimeTaken": _time_formatted,
                        }
                        qs = f"?positionId={_pos_id}&candidateId={_cand_id}"
                        t_hdrs = {"X-Tenant-Id": _tenant_id} if _tenant_id else {}
                        async with httpx.AsyncClient(timeout=15.0) as cl:
                            r = await cl.patch(f"{url}{qs}", json=payload, headers=t_hdrs)
                            rb = r.json() if "application/json" in r.headers.get("content-type", "") else {"raw": r.text}
                        _log_ws_api("PATCH assessment-summary (coding_round_complete)", "PATCH", url, payload, r.status_code, rb)
                        _summary_ok = 200 <= r.status_code < 300
                    except Exception as e:
                        logger.exception("coding_round_complete: assessment-summary patch failed: %s", e)
                # 2) PUT round-timing in AdminBackend
                if _admin_url and _pos_id and _cand_id:
                    try:
                        rt_payload = {
                            "positionId": _pos_id,
                            "candidateId": _cand_id,
                            "roundNumber": _round_num,
                            "roundCompleted": True,
                            "roundTimeTaken": _time_taken,
                            "roundStartTime": _start_time,
                            "roundEndTime": _end_time,
                        }
                        r_status, rb = await admin_put_round_timing(_admin_url, _tenant_id, rt_payload)
                        _log_ws_api("PUT round-timing (coding_round_complete)", "PUT", f"{_admin_url}/candidates/assessment-summaries/round-timing", rt_payload, r_status, rb)
                        _round_ok = 200 <= r_status < 300
                    except Exception as e:
                        logger.exception("coding_round_complete: round-timing failed: %s", e)
                await websocket.send_json({
                    "type": "coding_round_complete_ok",
                    "roundTimingUpdated": _round_ok,
                    "summaryUpdated": _summary_ok,
                })
                print(f"[WS OUT] sent coding_round_complete_ok (roundTiming={_round_ok}, summary={_summary_ok})\n", flush=True)

            elif msg_type == "round_complete":
                # Backend calls Admin API to update round timing (no frontend API call)
                pos_id = session.get("position_id")
                cand_id = session.get("candidate_id")
                tenant_id = session.get("tenant_id")
                round_num = msg.get("roundNumber")
                time_taken = msg.get("roundTimeTaken", "")
                start_time = msg.get("roundStartTime", "")
                end_time = msg.get("roundEndTime", "")
                if pos_id and cand_id and getattr(config, "ADMIN_BACKEND_URL", None) and round_num is not None:
                    try:
                        admin_url = getattr(config, "ADMIN_BACKEND_URL", None)
                        url = f"{admin_url or ''}/candidates/assessment-summaries/round-timing"
                        payload = {
                            "positionId": pos_id,
                            "candidateId": cand_id,
                            "roundNumber": int(round_num),
                            "roundCompleted": True,
                            "roundTimeTaken": time_taken,
                            "roundStartTime": start_time,
                            "roundEndTime": end_time,
                        }
                        r_status, resp_body = await admin_put_round_timing(
                            admin_url or "", tenant_id or "", payload
                        )
                        _log_ws_api("PUT assessment-summaries/round-timing (round_complete)", "PUT", url, payload, r_status, resp_body)
                        if 200 <= r_status < 300:
                            logger.info("Round %s timing updated", round_num)
                            print(f"\n[ROUND TIMING UPDATED] round={round_num} roundStartTime={start_time} roundEndTime={end_time} roundTimeTaken={time_taken}\n  payload: {json.dumps(payload, indent=2)}\n  response: {json.dumps(resp_body, indent=2)}\n", flush=True)
                        else:
                            logger.warning("round_complete failed: %s %s", r_status, resp_body)
                    except Exception as e:
                        logger.exception("round_complete failed: %s", e)
                await websocket.send_json({"type": "round_complete_ok"})
                print("[WS OUT] sent round_complete_ok\n", flush=True)
            elif msg_type == "test_complete":
                # Backend calls Admin API to mark interview as taken (no frontend API call)
                pos_id = session.get("position_id")
                cand_id = session.get("candidate_id")
                tenant_id = session.get("tenant_id")
                admin_url = getattr(config, "ADMIN_BACKEND_URL", None)
                if pos_id and cand_id and admin_url:
                    try:
                        url = f"{admin_url.rstrip('/')}/private-links/update-interview-status"
                        params = {"positionId": pos_id, "candidateId": cand_id}
                        r_status, resp_body = await admin_put_update_interview_status(
                            admin_url, pos_id, cand_id
                        )
                        _log_ws_api("PUT private-links/update-interview-status (test_complete)", "PUT", url, {"params": params}, r_status, resp_body)
                        if 200 <= r_status < 300:
                            logger.info("Interview status updated (test_complete): %s %s", pos_id, cand_id)
                        else:
                            logger.warning("update-interview-status failed: %s %s", r_status, resp_body)
                    except Exception as e:
                        logger.exception("Call to update-interview-status failed: %s", e)

                    # ── Set interview_completed_at on candidate_positions ────────
                    try:
                        ci_url = f"{admin_url.rstrip('/')}/internal/complete-interview"
                        ci_payload = {"positionId": pos_id, "candidateId": cand_id, "tenantId": tenant_id or ""}
                        ci_status, ci_body = await admin_patch_complete_interview(
                            admin_url, tenant_id, pos_id, cand_id
                        )
                        _log_ws_api("PATCH internal/complete-interview (test_complete)", "PATCH", ci_url, ci_payload, ci_status, ci_body)
                        if 200 <= ci_status < 300:
                            logger.info("interview_completed_at set (test_complete): pos=%s cand=%s", pos_id, cand_id)
                            print(f"[WS] interview_completed_at updated for pos={pos_id} cand={cand_id}\n", flush=True)
                        else:
                            logger.warning("complete-interview failed: %s %s", ci_status, ci_body)
                    except Exception as e:
                        logger.exception("Call to complete-interview failed: %s", e)

                await websocket.send_json({"type": "test_complete_ok"})
                print("[WS OUT] sent test_complete_ok\n", flush=True)
            else:
                # Never send "Unknown type" for audio_chunk (STT); server may receive it before start_listening
                if msg_type != "audio_chunk":
                    await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
    except WebSocketDisconnect:
        _test_close = (
            "\n" + "=" * 64 + "\n"
            "  LIVE STREAM END  —  /ws/test (test session)\n"
            "  WEBSOCKET CLOSED: client disconnected\n"
            "=" * 64 + "\n"
        )
        print(_test_close, flush=True)
        logger.info("[LIVE STREAM END] /ws/test")
    except Exception as e:
        logger.exception("Test WebSocket error: %s", e)
    finally:
        if assessment_log_task and not assessment_log_task.done():
            assessment_log_task.cancel()
            try:
                await assessment_log_task
            except asyncio.CancelledError:
                pass


@app.get("/api/merged/{client_id}/{position_id}/{candidate_id}")
def get_merged_video(client_id: str, position_id: str, candidate_id: str):
    """Stream the latest merged recording (e.g. for playback)."""
    dir_path = _merged_dir(client_id, position_id, candidate_id)
    # Glob for recording_*.mp4 and pick the latest one by mtime
    recordings = sorted(dir_path.glob("recording_*.mp4"), key=lambda p: p.stat().st_mtime)
    
    if not recordings:
        # Fallback to check if legacy recording.mp4 exists
        legacy_file = dir_path / "recording.mp4"
        if legacy_file.exists():
            return FileResponse(legacy_file, media_type="video/mp4", filename="recording.mp4")
        raise HTTPException(status_code=404, detail="Recording not found")
    
    latest_recording = recordings[-1]
    return FileResponse(latest_recording, media_type="video/mp4", filename=latest_recording.name)


@app.get("/api/transcription/token")
async def get_transcription_token():
    """Fetch a temporary token from AssemblyAI for real-time transcription (v3).
    AssemblyAI requires expires_in_seconds between 1 and 600; use 600 (10 min)."""
    api_key = (config.ASSEMBLYAI_API_KEY or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="AssemblyAI API key not configured")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://streaming.assemblyai.com/v3/token",
                params={"expires_in_seconds": 600},
                headers={"Authorization": api_key},
                timeout=10.0,
            )
        if response.status_code != 200:
            logger.error("AssemblyAI token %s: %s", response.status_code, response.text)
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch AssemblyAI token")
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error("AssemblyAI token request failed: %s", e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail="Failed to fetch AssemblyAI token")
    except Exception as e:
        logger.exception("Unexpected error fetching AssemblyAI token")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Generate high-quality speech using edge-tts."""
    try:
        dir_path = Path("temp_tts")
        dir_path.mkdir(exist_ok=True)
        
        import time
        file_name = f"tts_{int(time.time() * 1000)}.mp3"
        file_path = dir_path / file_name
        
        communicate = edge_tts.Communicate(request.text, request.voice)
        await communicate.save(str(file_path))
        
        return FileResponse(
            path=file_path,
            media_type="audio/mpeg",
            filename=file_name,
            # We add a background task to delete the file after it's sent
            background=asyncio.create_task(delete_later(file_path))
        )
    except Exception as e:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(e))


async def delete_later(path: Path):
    """Delete file after a short delay to ensure it's been sent."""
    await asyncio.sleep(10) # Wait 10 seconds
    try:
        if path.exists():
            os.remove(path)
    except Exception as e:
        logger.error(f"Failed to delete temp TTS file {path}: {e}")


@app.get("/ready")
def readiness_check():
    """Readiness check (same as Streaming)."""
    return {"ready": True}


@app.post("/schedule-interview")
async def schedule_interview(body: ScheduleInterviewRequest):
    """Schedule interview (private link). Same API as Streaming. Called by AdminBackend (proxy from AdminFrontend)."""
    request_id = f"{body.positionId}_{body.candidateId}"
    return {
        "success": True,
        "message": "Interview scheduling request received successfully",
        "data": {"requestId": request_id, "status": "processing"},
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "Streaming AI"}


@app.on_event("startup")
async def startup_log():
    port = getattr(config, "PORT", 9000)
    cand_url = getattr(config, "CANDIDATE_BACKEND_URL", "http://localhost:8003")
    banner = (
        f"\n{'='*60}\n"
        f"  Streaming AI — port {port}\n"
        f"  CandidateBackend (Q&A save/fetch): {cand_url}\n"
        f"  → Ensure CandidateBackend is RUNNING on that URL (e.g. port 8003)\n"
        f"  Logs: [API IN] / [API OUT] = HTTP; [WS] / [WS API] = WebSocket & outbound calls\n"
        f"{'='*60}\n"
    )
    print(banner, flush=True)
    logger.info("Streaming AI started — request/response logs will appear below.")
    # Start the FIFO report-generation background worker
    await _start_report_worker()
    # Ensure unique index on Candidates_Report (positionId + candidateId)
    await _ensure_report_index()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
