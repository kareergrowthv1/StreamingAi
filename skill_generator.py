"""
AI Skill Generator – generate mandatory and optional skills for a position.
Uses dynamic AI config from Superadmin (GET /superadmin/settings/ai-config); no hardcoded keys.
Output: 8 mandatory + 4 optional (comma-separated plain text with ",," between).
"""
import logging
import random
import time
import os

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import config
from ai_config_loader import get_ai_config
from verify_admin_token import verify_admin_token

logger = logging.getLogger(config.APP_NAME)

router = APIRouter(prefix="/ai", tags=["Skill Generator"])


@router.get("", summary="AI routes health")
async def ai_health():
    """Confirm /ai prefix. Skills: POST /ai/generate-skills, JD: POST /ai/generate-job-description."""
    return {"ok": True, "service": "Streaming AI", "routes": ["POST /ai/generate-skills", "POST /ai/generate-job-description"]}


class SkillGeneratorRequest(BaseModel):
    jobTitle: str = Field(..., description="Job title / position title")
    domain: str = Field("", description="Domain e.g. IT, NON-IT")
    minExperience: int = Field(..., description="Minimum years of experience")
    maxExperience: int = Field(..., description="Maximum years of experience")


def _get_openai_client(cfg: dict):
    api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required. Set in Superadmin Settings > AI Config or OPENAI_API_KEY.")
    import openai
    req_timeout = min(int(cfg.get("timeout", 300)), 20)
    req_retries = min(int(cfg.get("maxRetries", 3)), 1)
    return openai.OpenAI(
        api_key=api_key,
        base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
        timeout=req_timeout,
        max_retries=req_retries,
    )


def _process_skills_response(response: str) -> str:
    clean = response.strip()
    if (clean.startswith('"') and clean.endswith('"')) or (clean.startswith("'") and clean.endswith("'")):
        clean = clean[1:-1]
    clean = clean.replace('"', '').replace("'", '')
    skills = [s.strip() for s in clean.split(",")]
    skills = skills[:12]
    while len(skills) < 12:
        skills.append("")
    mandatory = skills[:8]
    optional = skills[8:12]
    return ",".join(mandatory) + ",," + ",".join(optional)


@router.post("/generate-skills", response_class=PlainTextResponse)
async def generate_skills(request: SkillGeneratorRequest = Body(...), _: bool = Depends(verify_admin_token)):
    """Generate 8 mandatory and 4 optional skills. Uses dynamic AI config."""
    experience_range = f"{request.minExperience}-{request.maxExperience} years"
    domain_part = f" in the {request.domain} domain" if (request.domain and request.domain.strip()) else ""
    timestamp = time.time()
    random_seed = random.randint(1000, 9999)

    prompt = f"""
Generate skills for a {request.jobTitle} position{domain_part} with {experience_range} of experience.

First provide exactly 8 mandatory core skills that are essential for this role.
Then provide exactly 4 optional skills that would be nice to have but aren't required.

Output format requirements:
- Provide 8 mandatory skills followed by 4 optional skills
- Format must be a comma-separated list
- No labels, headers, or explanations
- No numbering or bullets
- No additional text whatsoever
- NO QUOTATION MARKS around the output

I will format your response, so just provide 12 skills total (8 mandatory + 4 optional) in a simple comma-separated list.

Unique request ID: {timestamp}-{random_seed} (ignore this, just used to prevent response caching)
"""

    system_message = """
You are a specialized HR assistant that generates specific skill lists for job descriptions.
Your responses must be ONLY comma-separated values with NO additional text or formatting.
Do NOT include any quotation marks in your response.
The first 8 skills should be mandatory core skills, and the last 4 should be optional skills.
"""

    try:
        cfg = await get_ai_config()
        client = _get_openai_client(cfg)
        model = cfg.get("model", "gpt-3.5-turbo")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=min(int(cfg.get("maxTokens", 300)), 180),
        )
        text = (response.choices[0].message.content or "").strip()
        return _process_skills_response(text)
    except Exception as e:
        logger.exception("Skill generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate skills: {str(e)}")
