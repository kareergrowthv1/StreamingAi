"""
Conversational Message API – AI Mock Interview endpoints.
- POST /ai/generate-topics        → Generate 4 key focus concepts
- POST /ai/mock/generate-questions → Generate interview questions (non-conv: all, conv: first Q)
- POST /ai/mock/cross-question    → Generate 1 cross-question based on candidate's answer
- POST /ai/mock/save-session      → Save completed session via CandidateBackend API
"""
import logging
import os
import json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
import config
from ai_config_loader import get_ai_config
from aptitude_generator import generate_aptitude_questions

logger = logging.getLogger(config.APP_NAME)

router = APIRouter(prefix="/ai", tags=["Conversational Message"])

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _chat(system_prompt: str, user_prompt: str, json_mode: bool = False, temperature: float = 0.7) -> str:
    """Call the AI engine with dynamic config and return the response text."""
    import httpx
    cfg = await get_ai_config()
    api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key required. Set in Superadmin Settings > AI Config.")
    
    base_url = cfg.get("baseUrl", "https://api.openai.com/v1")
    model = cfg.get("model", "gpt-4o-mini") # Use a better default if not set

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            logger.error("AI API Error: %s - %s", e.response.status_code, e.response.text)
            raise HTTPException(status_code=500, detail=f"AI Engine Error: {e.response.text}")
        except Exception as e:
            logger.error("Internal AI Error: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))


def _clean_json(raw: str, key: str = None):
    """Strip markdown fences and parse JSON. Optionally extract a key."""
    text = raw
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence)[1].split("```")[0].strip()
        parsed = json.loads(text)
        if key and isinstance(parsed, dict):
            return parsed.get(key, parsed)
        return parsed


def _question_count(duration_minutes: int) -> int:
    mapping = {5: 3, 10: 6, 15: 9, 20: 12}
    return mapping.get(int(duration_minutes), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate Topics
# ─────────────────────────────────────────────────────────────────────────────

class TopicGenerationRequest(BaseModel):
    roundTitle: str = Field(..., description="Interview round name")
    messageText: str = Field(..., description="Candidate's input focus area")
    candidateName: str = Field("Candidate")

@router.post("/generate-topics")
async def generate_topics(request: TopicGenerationRequest = Body(...)):
    """Generate 4 key concepts from the candidate's stated focus area."""
    system = "You are a senior interview coach. Return ONLY a JSON object with key 'concepts' containing an array of 4 strings."
    user = (
        f"Round: {request.roundTitle}\n"
        f"Candidate focus: \"{request.messageText}\"\n\n"
        f"Identify exactly 4 key concepts or skills to focus this {request.roundTitle} interview session on. "
        'Return JSON: {"concepts": ["...", "...", "...", "..."]}'
    )
    try:
        raw = await _chat(system, user, json_mode=True)
        concepts = _clean_json(raw, key="concepts") or {}
        if not isinstance(concepts, list):
            vals = list(concepts.values())
            concepts = vals
        return {"success": True, "concepts": concepts[:4] if isinstance(concepts, list) else []}
    except Exception as e:
        logger.exception("Topic generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generate Questions
# ─────────────────────────────────────────────────────────────────────────────

class GenerateQuestionsRequest(BaseModel):
    roundTitle: str
    concepts: List[str]
    candidateName: str = "Candidate"
    mode: str = Field("conversational", description="'conversational' or 'non_conversational'")
    durationMinutes: int = Field(10)
    difficulty: Optional[str] = Field("Medium")

@router.post("/mock/generate-questions")
async def generate_questions(request: GenerateQuestionsRequest = Body(...)):
    """
    Non-conversational: return all N questions upfront.
    Conversational: return only the 1st opening question.
    """
    logger.info("generate_questions triggered: roundTitle='%s', mode='%s', dur=%d", request.roundTitle, request.mode, request.durationMinutes)
    
    # AGGRESSIVE DETECTION: If it's non-conversational and the word "aptitude" is anywhere,
    # OR if durationMinutes is one of our standard counts (5, 10, 15) and mode is non_conversational,
    # we treat it as an Aptitude MCQ assessment.
    is_aptitude = "aptitude" in str(request.roundTitle).lower() or request.roundTitle == "Aptitude"
    
    if request.mode == "non_conversational" and is_aptitude:
        count = request.durationMinutes
        topic_blocks = [{
            "question": "Aptitude Assessment",
            "topics": request.concepts,
            "count": count,
            "difficulty": request.difficulty or 'MEDIUM'
        }]
        try:
            questions = await generate_aptitude_questions(topic_blocks)
            return {
                "success": True, 
                "questions": questions, 
                "total": len(questions), 
                "mode": "non_conversational",
                "isMCQ": True
            }
        except Exception as e:
            logger.exception("Aptitude generation failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    # Generic logic for other rounds or conversational mode
    total = _question_count(request.durationMinutes)
    concepts_str = ", ".join(request.concepts)

    if request.mode == "non_conversational":
        # Use exact count if provided in durationMinutes, otherwise use the mapping
        count = request.durationMinutes if request.durationMinutes > 0 else _question_count(request.durationMinutes)
        system = (
            "You are a strict, professional interviewer. "
            "Return ONLY a JSON object with key 'questions' containing an array of strings."
        )
        user = (
            f"Create exactly {count} distinct interview questions for a '{request.roundTitle}' round.\n"
            f"Focus areas: {concepts_str}\n"
            f"Candidate: {request.candidateName}\n"
            "Questions should vary in depth — start moderate, build to challenging.\n"
            'Return JSON: {"questions": ["Q1", "Q2", ...]}'
        )
        try:
            raw = await _chat(system, user, json_mode=True)
            questions = _clean_json(raw, key="questions") or {}
            if not isinstance(questions, list):
                vals = list(questions.values())
                questions = vals
            return {"success": True, "questions": questions[:count] if isinstance(questions, list) else [], "total": count, "mode": "non_conversational"}
        except Exception as e:
            logger.exception("Question generation failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Conversational: just get the opening question
        system = (
            "You are a professional interviewer conducting a verbal mock interview. "
            "Ask one concise, engaging opening question. Return ONLY a JSON object with key 'question'."
        )
        user = (
            f"Start a '{request.roundTitle}' mock interview for {request.candidateName}.\n"
            f"Focus areas: {concepts_str}\n"
            "Ask a clear opening question to begin the session.\n"
            'Return JSON: {"question": "..."}'
        )
        try:
            raw = await _chat(system, user, json_mode=True)
            data = _clean_json(raw) or {}
            question = data.get("question", "Tell me about yourself and what brings you here today.") if isinstance(data, dict) else "Tell me about yourself and what brings you here today."
            return {"success": True, "question": question, "total": total, "mode": "conversational"}
        except Exception as e:
            logger.exception("Opening question generation failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross Question
# ─────────────────────────────────────────────────────────────────────────────

class CrossQuestionRequest(BaseModel):
    roundTitle: str
    concepts: List[str]
    previousQuestion: str
    candidateAnswer: str
    candidateName: str = "Candidate"
    questionsSoFar: int = 1
    totalQuestions: Optional[int] = None

@router.post("/mock/cross-question")
async def cross_question(request: CrossQuestionRequest = Body(...)):
    """Generate 1 follow-up question based on the candidate's answer."""
    concepts_str = ", ".join(request.concepts)
    total_qs = request.totalQuestions
    remaining = (total_qs - request.questionsSoFar) if total_qs is not None else 99
    system = (
        "You are a sharp professional interviewer. "
        "Return ONLY a JSON object with key 'question' containing the follow-up question."
    )
    closing_clause = "This is the last question — make it a closing reflective one.\n" if (request.totalQuestions and remaining <= 1) else ""
    user = (
        f"Round: {request.roundTitle} | Candidate: {request.candidateName}\n"
        f"Focus areas: {concepts_str}\n\n"
        f"Previous question: \"{request.previousQuestion}\"\n"
        f"Candidate answered: \"{request.candidateAnswer}\"\n\n"
        f"Generate 1 sharp follow-up question that probes deeper or transitions to the next concept.\n"
        f"Questions remaining after this: {remaining}.\n"
        f"{closing_clause}"
        'Return JSON: {"question": "..."}'
    )
    try:
        raw = await _chat(system, user, json_mode=True)
        data = _clean_json(raw) or {}
        question = data.get("question", "Can you elaborate further on that point?") if isinstance(data, dict) else "Can you elaborate further on that point?"
        return {"success": True, "question": question}
    except Exception as e:
        logger.exception("Cross question generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save Session to MongoDB
# ─────────────────────────────────────────────────────────────────────────────

class QAPair(BaseModel):
    question: str
    answer: str = ""
    correctAnswer: Optional[str] = None
    explanation: Optional[str] = None
    isCross: bool = False

class SaveSessionRequest(BaseModel):
    candidateId: str
    round: int
    roundTitle: str
    mode: str
    durationMinutes: int
    concepts: List[str]
    questions: List[QAPair]
    status: str = "completed"
    reportLevel: str = "standard"  # Added for subscription-based scaling
    startedAt: Optional[str] = None
    completedAt: Optional[str] = None

@router.post("/mock/save-session")
async def save_session(request: SaveSessionRequest = Body(...)):
    """Save completed AI mock session via CandidateBackend internal API (no direct DB in Streaming AI)."""
    import httpx

    cand_url = (getattr(config, "CANDIDATE_BACKEND_URL", "") or "").rstrip("/")
    if not cand_url:
        raise HTTPException(status_code=503, detail="CANDIDATE_BACKEND_URL not configured")

    token = (
        getattr(config, "INTERNAL_SERVICE_TOKEN", "")
        or getattr(config, "ADMIN_SERVICE_TOKEN", "")
        or ""
    ).strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Service-Token"] = token

    payload = {
        "candidateId": request.candidateId,
        "round": request.round,
        "roundTitle": request.roundTitle,
        "mode": request.mode,
        "durationMinutes": request.durationMinutes,
        "concepts": request.concepts,
        "questions": [q.dict() if hasattr(q, "dict") else q.model_dump() for q in request.questions],
        "status": request.status,
        "startedAt": request.startedAt or datetime.now(timezone.utc).isoformat(),
        "completedAt": request.completedAt or datetime.now(timezone.utc).isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{cand_url}/internal/streaming/ai-mock/save-session",
                headers=headers,
                json=payload,
            )

        if resp.status_code >= 300:
            detail = resp.text[:400]
            logger.error("CandidateBackend save-session failed: %s %s", resp.status_code, detail)
            raise HTTPException(status_code=502, detail=f"CandidateBackend save-session failed: {detail}")

        data = resp.json() if resp.content else {}
        return {
            "success": bool(data.get("success", True)),
            "sessionId": data.get("sessionId"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CandidateBackend save-session request failed: %s", e)
        raise HTTPException(status_code=500, detail=f"save-session failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Generate Interview Report
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/mock/generate-report")
async def generate_report(request: SaveSessionRequest = Body(...)):
    """Generate a final comprehensive assessment report and percentage score for the candidate."""
    try:
        # Build text representation, handling "No response" markers
        qa_lines = []
        is_aptitude = "aptitude" in str(request.roundTitle).lower()
        
        for i, q in enumerate(request.questions, 1):
            ans = q.answer.strip()
            if not ans or ans.lower() == "no response" or ans.lower() == "skipped":
                ans = "[Candidate skipped/provided no response]"
            
            line = f"Q{i}: {q.question}\nCandidate Answer: {ans}"
            if is_aptitude and q.correctAnswer:
                line += f"\nCorrect Answer: {q.correctAnswer}"
            qa_lines.append(line)
            
        qa_text = "\n\n".join(qa_lines)
        concepts_str = ", ".join(request.concepts)

        # Subscription-based scaling
        depth_instructions = {
            "min": "Provide a BREIF, high-level summary. Focus only on the most critical points. Keep the analysis concise (~30% of normal length).",
            "standard": "Provide a BALANCED assessment with moderate detail. Cover the main strengths and improvements (~60% of normal length).",
            "complete": "Provide a COMPREHENSIVE, deep-dive evaluation. Analyze every nuance of the candidate's responses in great detail (100% depth)."
        }
        scale_guidance = depth_instructions.get(request.reportLevel, depth_instructions["standard"])

        if is_aptitude:
            system = (
                "You are an expert aptitude evaluation system. Your task is to generate a structured MCQ review. "
                "Return ONLY a JSON object with keys 'score' (integer 0-100) and 'analysis' (detailed markdown report)."
            )
            user = (
                f"Round: {request.roundTitle} | Mode: {request.mode}\n"
                f"Evaluation Data:\n{qa_text}\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. DO NOT include any sections titled 'Strengths', 'Areas for Improvement', 'Overall Feedback', or any general textual commentary.\n"
                "2. The 'analysis' field MUST only contain a question-by-question breakdown in Markdown format.\n"
                "3. For EACH question, use the following template:\n"
                "### Question [N]: [Question Text]\n"
                "- **Candidate Answer**: [Answer]\n"
                "- **Correct Answer**: [Correct Answer Key]\n"
                "- **Solution/Explanation**: [Detailed step-by-step explanation including any formulas used to solve the problem.]\n"
                "---\n"
                "4. Calculate the 'score' as the percentage of correct answers.\n"
                "5. Only provide the question-by-question walkthrough. Avoid any other summary or feedback."
            )
        else:
            system = (
                "You are an expert technical interviewer evaluating a candidate's mock interview performance. "
                "Evaluate the provided interview transcript rigorously but fairly. "
                "Return ONLY a JSON object with keys 'score' (integer 0-100) and 'analysis' (detailed markdown report)."
            )
            user = (
                f"Round: {request.roundTitle} | Mode: {request.mode} | Report Level: {request.reportLevel}\n"
                f"Focus areas: {concepts_str}\n\n"
                f"Interview Transcript:\n{qa_text}\n\n"
                "INSTRUCTIONS:\n"
                f"0. DEPTH GUIDANCE: {scale_guidance}\n"
                "1. Analyze the candidate's technical depth, clarity, and communication.\n"
                "2. If the candidate has multiple 'no response' markers, penalize the score significantly.\n"
                "3. If they gave valid answers for some questions, evaluate those answers' quality.\n"
                "4. Provide a numerical score (0-100) and an analysis in Markdown.\n"
                "5. The Markdown should include sections: Strengths, Areas for Improvement, and Overall Feedback."
            )
        
        raw = await _chat(system, user, json_mode=True, temperature=0.3) # Lower temperature for stricter adherence
        data = _clean_json(raw) or {}
        score = data.get("score", 0) if isinstance(data, dict) else 0
        analysis = data.get("analysis", "Evaluation could not be finalized.") if isinstance(data, dict) else "Analysis incomplete."
        
        return {"success": True, "score": score, "analysis": analysis}
    except Exception as e:
        logger.exception("Report generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Report Error: {str(e)}")

