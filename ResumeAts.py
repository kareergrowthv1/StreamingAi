"""
Resume ATS – resume vs job description match scoring.
Uses dynamic AI config from Superadmin (GET /superadmin/settings/ai-config); no hardcoded API keys.
- POST /resume-ats/score: fetch JD/resume from AdminBackend, run AI, return score.
- POST /resume-ats/calculate-score: direct scoring when text is provided.
"""
import json
import logging
import os

import httpx
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

import config
from ai_config_loader import get_ai_config

logger = logging.getLogger(config.APP_NAME)

router = APIRouter(prefix="/resume-ats", tags=["Resume ATS"])


class ResumeAtsRequest(BaseModel):
    """Request body: resume and job description text (for direct calculate-score)."""
    resumeText: str = Field(..., description="Full resume text")
    jobDescriptionText: str = Field(..., description="Full job description or position requirements text")


class ScoreByIdsRequest(BaseModel):
    """Request body: identifiers so we fetch JD/resume from AdminBackend and run AI."""
    positionId: str = Field(..., description="Position UUID")
    candidateId: str = Field(..., description="Candidate UUID")
    positionCandidateId: str = Field(..., description="Position-candidate link ID")
    tenantId: str = Field(..., description="Tenant DB/schema name (X-Tenant-Id)")


class ResumeAtsResponse(BaseModel):
    """Response: overall score (0–100) and optional category scores."""
    overallScore: float = Field(..., description="Overall match score 0–100")
    categoryScores: dict = Field(default_factory=dict, description="Scores per category")


def _get_openai_client(cfg: dict):
    """Build OpenAI client from dynamic config (Superadmin or env fallback)."""
    try:
        import openai
        api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Set in Superadmin Settings > AI Config or OPENAI_API_KEY in .env")
        return openai.OpenAI(
            api_key=api_key,
            base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
            timeout=cfg.get("timeout", 300),
            max_retries=cfg.get("maxRetries", 3),
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="openai package not installed. Add 'openai' to requirements.txt and install.",
        )


async def _generate_completion(prompt: str, system_message: str, temperature_override: float = None) -> str:
    """Call OpenAI chat completion using dynamic config."""
    cfg = await get_ai_config()
    client = _get_openai_client(cfg)
    model = cfg.get("model", "gpt-3.5-turbo")
    temperature = float(temperature_override if temperature_override is not None else cfg.get("temperature", 0.7))
    max_tokens = min(int(cfg.get("maxTokens", 1024)), 500)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("OpenAI call failed: %s", e)
        raise HTTPException(status_code=502, detail=f"AI provider error: {str(e)}")


def _process_match_response(response: str) -> dict:
    """Parse JSON from model response and validate scores."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON object in response")
        data = json.loads(response[start:end])
        if "overallScore" not in data:
            raise ValueError("Missing overallScore")
        overall = float(data["overallScore"])
        overall = max(0.0, min(100.0, overall))
        categories = data.get("categoryScores") or {}
        expected = ["technicalSkills", "experience", "education", "industryKnowledge", "roleSpecific"]
        for k in expected:
            if k not in categories:
                categories[k] = 50.0
            else:
                categories[k] = max(0.0, min(100.0, float(categories[k])))
        return {"overallScore": overall, "categoryScores": categories}
    except json.JSONDecodeError as e:
        logger.error("Resume ATS JSON parse error: %s", e)
        raise HTTPException(status_code=502, detail="Invalid JSON from AI response")
    except Exception as e:
        logger.error("Resume ATS process error: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


async def _fetch_score_input_from_admin(position_id: str, candidate_id: str, tenant_id: str):
    """Fetch jobDescriptionText and resumeText from AdminBackend internal API."""
    admin_url = getattr(config, "ADMIN_BACKEND_URL", None) or os.getenv("ADMIN_BACKEND_URL", "http://localhost:8002")
    admin_url = admin_url.rstrip("/")
    token = getattr(config, "ADMIN_SERVICE_TOKEN", None) or os.getenv("ADMIN_SERVICE_TOKEN", "")
    if not token:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_SERVICE_TOKEN not configured. Required to fetch JD/resume from AdminBackend.",
        )
    url = f"{admin_url}/internal/score-resume-input"
    payload = {"positionId": position_id, "candidateId": candidate_id, "tenantId": tenant_id}
    headers = {"Content-Type": "application/json", "X-Service-Token": token}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                err = resp.text or resp.reason_phrase
                raise HTTPException(
                    status_code=502,
                    detail=f"AdminBackend score-resume-input failed: {resp.status_code} {err}",
                )
            data = resp.json()
            if not data.get("success"):
                raise HTTPException(status_code=502, detail=data.get("message", "AdminBackend returned success=false"))
            min_exp = data.get("minExperience")
            max_exp = data.get("maxExperience")
            if min_exp is not None and not isinstance(min_exp, (int, float)):
                try:
                    min_exp = int(min_exp)
                except (TypeError, ValueError):
                    min_exp = None
            if max_exp is not None and not isinstance(max_exp, (int, float)):
                try:
                    max_exp = int(max_exp)
                except (TypeError, ValueError):
                    max_exp = None
            return (
                (data.get("jobDescriptionText") or "").strip(),
                (data.get("resumeText") or "").strip(),
                min_exp,
                max_exp,
            )
    except httpx.RequestError as e:
        logger.exception("Request to AdminBackend failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Failed to fetch score input from AdminBackend: {e}")


def _build_prompt_and_system(jd_text: str, resume_text: str, min_experience: int = None, max_experience: int = None):
    """Build prompt and system message for resume vs JD scoring. Strict on experience match (ref-style)."""
    experience_block = ""
    if min_experience is not None and max_experience is not None and (min_experience > 0 or max_experience > 0):
        experience_block = f"""
REQUIRED EXPERIENCE FOR THIS ROLE: {min_experience}-{max_experience} years.
CRITICAL: If the candidate has significantly LESS experience than required (e.g. fresher, 0-2 years when the role requires {min_experience}+ years), you MUST:
- Set "experience" in categoryScores to at most 35.
- Set overallScore to at most 40 (reject). Do NOT score 60+ when experience requirement is clearly not met.
Only score 60 or above if the candidate's total years of experience meet or nearly meet the required range.
"""

    prompt = f"""
Analyze the following JOB DESCRIPTION and RESUME to calculate a highly accurate match score. Be STRICT and CONSERVATIVE; prefer under-scoring over over-scoring.
{experience_block}

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

SCORING CRITERIA:
1. TECHNICAL SKILLS: How well the candidate's skills align with the JD.
2. EXPERIENCE: Years and type of experience MUST match the role's seniority. If the role requires X+ years and the candidate has far less, score experience and overall LOW (0-40).
3. EDUCATION: Degree and certifications vs requirements.
4. INDUSTRY KNOWLEDGE: Similar domains or industries.
5. ROLE ALIGNMENT: Overall fit for the specific role.

STRICT SCORING (like ref backend_ai-main):
- 0-19: No match; unrelated or experience far below requirement (e.g. fresher when 10+ years required).
- 20-39: Weak match; minimal overlap or experience gap too large.
- 40-59: Fair match; some relevant skills but missing significant requirements/experience.
- 60-74: Good match; meets most core requirements including experience level.
- 75-100: Strong match; meets or exceeds requirements.

RULES:
1. If the position requires substantial experience (e.g. 10-20 years) and the resume shows fresher or 0-2 years, overallScore MUST be at most 40.
2. Only give 60+ for strong, clear matches with evidence.
3. Output only a single JSON object, no other text.
"""
    system_message = """
You are an expert HR evaluator with strict standards. Score resume vs job description accurately. Experience level is critical: if the role requires X-Y years and the candidate has far less, you MUST score low (overall and experience category). Return only valid JSON.

Output your analysis as a JSON object only:
{
    "overallScore": <number 0-100>,
    "categoryScores": {
        "technicalSkills": <0-100>,
        "experience": <0-100>,
        "education": <0-100>,
        "industryKnowledge": <0-100>,
        "roleSpecific": <0-100>
    }
}
"""
    return prompt, system_message


@router.post("/score", response_model=ResumeAtsResponse)
async def score_resume_by_ids(request: ScoreByIdsRequest = Body(...)):
    """Full resume score: fetch JD and resume text from AdminBackend, then run AI (dynamic config)."""
    try:
        job_description_text, resume_text, min_exp, max_exp = await _fetch_score_input_from_admin(
            request.positionId, request.candidateId, request.tenantId
        )
        if len(resume_text) < 50:
            raise HTTPException(
                status_code=400,
                detail="resumeText too short (min 50 characters). Ensure resume is extracted when adding candidate.",
            )
        if len(job_description_text) < 20:
            raise HTTPException(
                status_code=400,
                detail="jobDescriptionText too short (min 20 characters). Ensure JD is extracted for the position.",
            )
        prompt, system_message = _build_prompt_and_system(
            job_description_text, resume_text, min_experience=min_exp, max_experience=max_exp
        )
        response_text = await _generate_completion(prompt, system_message, temperature_override=0.1)
        result = _process_match_response(response_text)
        return ResumeAtsResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("resume-ats/score unhandled error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Resume scoring failed: {getattr(e, 'message', str(e))}",
        ) from e


@router.post("/calculate-score", response_model=ResumeAtsResponse)
async def calculate_resume_match_score(request: ResumeAtsRequest = Body(...)):
    """Calculate match score when text is already provided. Uses dynamic AI config."""
    resume_text = (request.resumeText or "").strip()
    jd_text = (request.jobDescriptionText or "").strip()
    if len(resume_text) < 50:
        raise HTTPException(status_code=400, detail="resumeText too short (min 50 characters)")
    if len(jd_text) < 20:
        raise HTTPException(status_code=400, detail="jobDescriptionText too short (min 20 characters)")
    prompt, system_message = _build_prompt_and_system(jd_text, resume_text)
    response_text = await _generate_completion(prompt, system_message, temperature_override=0.1)
    result = _process_match_response(response_text)
    return ResumeAtsResponse(**result)
