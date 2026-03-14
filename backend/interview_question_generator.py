"""
AI Interview Question Generator.

Generates ONLY conversational/speech-type interview questions — verbal Q&A format.
NO coding challenges, NO written tasks, NO MCQ, NO aptitude puzzles.

Uses dynamic AI config from Superadmin (GET /superadmin/settings/ai-config); no hardcoded keys.
"""
import logging
import os
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, Body, HTTPException
from pydantic import BaseModel, Field

import config
from ai_config_loader import get_ai_config
from verify_admin_token import verify_admin_token

logger = logging.getLogger(config.APP_NAME)

router = APIRouter(prefix="/ai", tags=["Interview Question Generator"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InterviewQuestionRequest(BaseModel):
    position: str = Field(..., description="Job title / position name")
    minExperience: int = Field(0, description="Minimum years of experience")
    maxExperience: int = Field(0, description="Maximum years of experience")
    mandatorySkills: List[str] = Field(default_factory=list, description="Mandatory / required skills")
    optionalSkills: List[str] = Field(default_factory=list, description="Optional / preferred skills")
    section: str = Field(
        "position_specific",
        description=(
            "Which section to generate for. "
            "'general' → HR / behavioural general questions. "
            "'position_specific' → role / skills / experience questions."
        ),
    )
    count: int = Field(5, ge=1, le=20, description="Number of questions to generate (1-20)")


class GeneratedQuestion(BaseModel):
    text: str
    prepareTime: str = "5 secs"
    answerTime: str = "2 mins"


class InterviewQuestionResponse(BaseModel):
    questions: List[GeneratedQuestion]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_openai_client(cfg: dict):
    api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Set in Superadmin Settings > AI Config or OPENAI_API_KEY env var."
        )
    import openai
    return openai.OpenAI(
        api_key=api_key,
        base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
        timeout=cfg.get("timeout", 300),
        max_retries=cfg.get("maxRetries", 3),
    )


def _build_prompts(req: InterviewQuestionRequest):
    """Return (system_prompt, user_prompt) for the given section."""
    man_text = ", ".join(req.mandatorySkills) if req.mandatorySkills else "N/A"
    opt_text = ", ".join(req.optionalSkills) if req.optionalSkills else "N/A"
    exp_range = (
        f"{req.minExperience}–{req.maxExperience} years"
        if req.maxExperience > req.minExperience
        else f"{req.minExperience}+ years"
    )

    system_prompt = """You are an expert technical interviewer who specialises in conversational, verbal interview questions.

CRITICAL RULES — follow these exactly:
1. Generate ONLY conversational / spoken-answer questions. These are questions a candidate ANSWERS BY SPEAKING — like a real face-to-face or video interview.
2. NEVER generate:
   - Coding challenges ("write a function", "implement", "code a solution", "debug this snippet")
   - Written assignments or take-home tasks
   - Multiple-choice / true-false questions
   - Aptitude, maths, or puzzle questions
   - Any question that requires a whiteboard or written output
3. Every question must be open-ended and suitable for verbal response (typically 1–3 minutes spoken answer).
4. Questions must be relevant to the position, skills, and experience level provided.
5. Return ONLY valid JSON — an array of objects, each with the key "text" (the question string). No markdown, no extra keys."""

    if req.section == "general":
        user_prompt = (
            f"Generate {req.count} general HR / behavioural conversational interview questions "
            f"for a {req.position} role requiring {exp_range} of experience.\n\n"
            f"Mandatory Skills context: {man_text}\n"
            f"Optional Skills context: {opt_text}\n\n"
            f"Guidelines:\n"
            f"- Mix motivational questions (why this role, career goals), behavioural (STAR-format friendly), "
            f"and soft-skill questions (teamwork, conflict resolution, leadership).\n"
            f"- Keep them relevant to someone applying for a {req.position} position.\n"
            f"- VERBAL answers only — no coding, no written tasks.\n\n"
            f"Return JSON array: [{{'text': 'Question 1'}}, {{'text': 'Question 2'}}, ...]"
        )
    else:
        user_prompt = (
            f"Generate {req.count} position-specific conversational interview questions "
            f"for a {req.position} role requiring {exp_range} of experience.\n\n"
            f"Mandatory Skills: {man_text}\n"
            f"Optional Skills: {opt_text}\n\n"
            f"Guidelines:\n"
            f"- Questions must probe the candidate's knowledge and experience with the listed skills.\n"
            f"- Include questions about architectural decisions, problem-solving approach, real-world scenarios, "
            f"past projects, and best practices in the relevant skill areas.\n"
            f"- VERBAL answers only — no 'write code', no 'implement', no whiteboard tasks.\n"
            f"- Suitable for a spoken video interview where the candidate explains their thinking out loud.\n\n"
            f"Return JSON array: [{{'text': 'Question 1'}}, {{'text': 'Question 2'}}, ...]"
        )

    return system_prompt, user_prompt


def _parse_questions(raw: str, count: int) -> List[GeneratedQuestion]:
    """Parse the raw JSON string returned by the model into a list of GeneratedQuestion."""
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        data = json.loads(text)
        questions = []
        for item in data:
            if isinstance(item, dict) and "text" in item:
                q_text = str(item["text"]).strip()
                if q_text:
                    questions.append(
                        GeneratedQuestion(
                            text=q_text,
                            prepareTime=item.get("prepareTime", "5 secs"),
                            answerTime=item.get("answerTime", "2 mins"),
                        )
                    )
        return questions[:count]
    except Exception as exc:
        logger.warning("Failed to parse AI JSON response: %s | Raw: %.300s", exc, raw)
        return []


def _fallback_questions(req: InterviewQuestionRequest) -> List[GeneratedQuestion]:
    """Return generic fallback questions if AI call fails."""
    if req.section == "general":
        texts = [
            "Tell me about yourself and your professional journey so far.",
            "Why are you interested in this position and our company?",
            "What are your greatest professional strengths and how have they helped you in your career?",
            "Describe a challenging situation you faced at work and how you handled it.",
            "Where do you see yourself professionally in the next three to five years?",
            "How do you prioritise tasks when working on multiple projects simultaneously?",
            "Can you give an example of a time you worked effectively in a team?",
            "What motivates you to perform at your best every day?",
            "How do you handle feedback or criticism of your work?",
            "Do you have any questions for us about the role or the company?",
        ]
    else:
        skills_hint = (
            f" particularly around {req.mandatorySkills[0]}"
            if req.mandatorySkills
            else ""
        )
        texts = [
            f"Can you walk me through your experience as a {req.position}{skills_hint}?",
            f"What is the most complex technical challenge you solved in your {req.position} role?",
            f"How do you stay updated with the latest developments in {', '.join(req.mandatorySkills[:2]) or 'your field'}?",
            f"Describe a project where you had to make a critical architectural or design decision. What did you decide and why?",
            f"How do you ensure the quality and reliability of your work as a {req.position}?",
            f"Can you explain how you have used {req.mandatorySkills[0] if req.mandatorySkills else 'your key skills'} in a real-world project?",
            f"Tell me about a time you had to learn a new technology or tool quickly. How did you approach it?",
            f"How do you collaborate with cross-functional teams such as product managers, designers, or QA engineers?",
            f"What strategies do you use to debug or troubleshoot difficult issues in your work?",
            f"How do you balance technical debt with delivering new features on time?",
        ]
    return [
        GeneratedQuestion(text=t, prepareTime="5 secs", answerTime="2 mins")
        for t in texts[: req.count]
    ]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/generate-interview-questions",
    response_model=InterviewQuestionResponse,
    summary="Generate conversational interview questions for a position",
)
async def generate_interview_questions(
    request: InterviewQuestionRequest = Body(...),
    _: bool = Depends(verify_admin_token),
):
    """
    Generate conversational / speech-type interview questions.

    - Section **general**: HR / behavioural questions (motivation, teamwork, career goals …)
    - Section **position_specific**: role & skill specific verbal questions

    Returns a list of question objects (text, prepareTime, answerTime).
    NEVER returns coding tasks, MCQ, or written assignments.
    """
    system_prompt, user_prompt = _build_prompts(request)

    try:
        cfg = await get_ai_config()
        client = _get_openai_client(cfg)
        model = cfg.get("model", "gpt-3.5-turbo")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=cfg.get("temperature", 0.7),
            max_tokens=1500,
        )

        raw = (response.choices[0].message.content or "").strip()
        logger.info(
            "AI interview questions generated for '%s' section='%s' count=%d",
            request.position,
            request.section,
            request.count,
        )

        questions = _parse_questions(raw, request.count)

        if not questions:
            logger.warning("AI returned empty/invalid JSON, using fallback questions.")
            questions = _fallback_questions(request)

        return InterviewQuestionResponse(questions=questions)

    except Exception as exc:
        logger.exception(
            "Interview question generation failed for '%s': %s", request.position, exc
        )
        questions = _fallback_questions(request)
        return InterviewQuestionResponse(questions=questions)
