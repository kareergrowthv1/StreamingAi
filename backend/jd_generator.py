"""
AI Job Description Generator – generate JD text from position, experience, and skills.
Uses dynamic AI config from Superadmin (GET /superadmin/settings/ai-config); no hardcoded keys.
"""
import logging
import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field

import config
from ai_config_loader import get_ai_config
from verify_admin_token import verify_admin_token

logger = logging.getLogger(config.APP_NAME)

router = APIRouter(prefix="/ai", tags=["JD Generator"])


class JDGeneratorRequest(BaseModel):
    position: str = Field(..., description="Job title / position")
    minExperience: int = Field(..., description="Minimum years of experience")
    maxExperience: int = Field(..., description="Maximum years of experience")
    manSkills: List[str] = Field(..., description="Mandatory skills")
    optSkills: List[str] = Field(default_factory=list, description="Optional skills")


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


def _validate_jd(
    text: str,
    position: str,
    min_experience: int,
    mandatory_skills: List[str],
    optional_skills: List[str],
) -> str:
    if not text or not text.strip():
        return f"job title: {position} ({min_experience}+ years of experience)\n\n(No content generated.)"
    if not text.lower().strip().startswith("job title:"):
        text = f"job title: {position} ({min_experience}+ years of experience)\n\n{text}"
    return text


def _fallback_jd(
    position: str,
    min_experience: int,
    mandatory_skills: List[str],
    optional_skills: List[str],
) -> str:
    man = ", ".join(mandatory_skills) if mandatory_skills else "N/A"
    opt = ", ".join(optional_skills) if optional_skills else "N/A"
    return (
        f"job title: {position} ({min_experience}+ years of experience)\n\n"
        f"Company: [Company-name]\n"
        f"[Company-name] is a leading tech company specializing in innovative software solutions.\n\n"
        f"Location: [Insert Location]\n\n"
        f"Work Mode: [Insert Work Mode]\n\n"
        f"Position Overview:\n"
        f"We are seeking a {position} with {min_experience}+ years of experience. Proficiency in {man}.\n\n"
        f"Key Responsibilities:\n"
        f"Develop high-quality solutions using relevant technologies\n"
        f"Collaborate with the team to design and implement features\n"
        f"Write clean, maintainable code and conduct testing\n"
        f"Participate in code reviews\n"
        f"Troubleshoot and optimize performance\n\n"
        f"Required Qualifications\n"
        f"{min_experience}+ years of experience. Proficiency in {man}.\n\n"
        f"Preferred Qualifications\n"
        f"Experience with {opt}. Bachelor's degree or related field.\n\n"
        f"Benefits:\n"
        f"[Insert benefits information]\n\n"
        f"Qualified candidates are encouraged to apply at [Company-name]."
    )


@router.post("/generate-job-description")
async def generate_job_description(request: JDGeneratorRequest = Body(...), _: bool = Depends(verify_admin_token)):
    """Generate job description text. Uses dynamic AI config."""
    mandatory = request.manSkills or []
    optional = request.optSkills or []
    man_text = ", ".join(mandatory)
    opt_text = ", ".join(optional)

    system_prompt = """
You are a professional technical recruiter specializing in creating job descriptions.
Generate a job description based on the provided position, experience level, and skills.

The job description MUST follow this EXACT format with all sections:

job title: [Position Title] ([Min Experience]+ years of experience)

Company: [Company-name]
[Company-name] is a leading tech company specializing in innovative software solutions for a wide range of industries. We are committed to pushing the boundaries of technology to deliver cutting-edge products to our clients.

Location: [Insert Location]

Work Mode: [Insert Work Mode]

Position Overview:
[Write a concise overview paragraph tailored to the position and skills. Mention the mandatory skills.]

Key Responsibilities:
[5 responsibilities specific to the position and skills, each on a new line without bullets]

Required Qualifications
[Experience requirement] years of experience in [relevant field]
Proficiency in [list all mandatory skills]
[One more relevant qualification]

Preferred Qualifications
Experience with [list all optional skills]
Bachelor's degree in [relevant field] or related field

Benefits:
[Insert benefits information]

Qualified candidates with the required experience and skills are encouraged to apply and be part of our innovative team at [Company-name]
"""

    user_prompt = (
        f"Create a job description for a {request.position} position requiring {request.minExperience}+ years of experience.\n\n"
        f"Mandatory Skills: {man_text}\n\n"
        f"Optional Skills: {opt_text}\n\n"
        f"Important instructions:\n"
        f"1. Follow the exact format from the system prompt\n"
        f"2. DO NOT replace placeholders like [Company-name], [Insert Location], [Insert Work Mode], [Insert benefits information]\n"
        f"3. Write content specific to the {request.position} role and the listed skills\n"
        f"4. Key responsibilities: 5 items without bullet points\n"
        f"5. Return only the formatted job description template\n"
    )

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
            temperature=0.7,
            max_tokens=1500,
        )
        text = (response.choices[0].message.content or "").strip()
        result = _validate_jd(
            text,
            request.position,
            request.minExperience,
            mandatory,
            optional,
        )
        return {"jobDescription": result}
    except Exception as e:
        logger.exception("JD generation failed: %s", e)
        result = _fallback_jd(
            request.position,
            request.minExperience,
            mandatory,
            optional,
        )
        return {"jobDescription": result}
