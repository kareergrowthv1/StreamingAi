"""
Resume Content Enhancer — AI-powered resume section text improvement.
POST /resume-content/enhance : Takes section title, type, and user-typed draft content,
streams back enhanced professional resume content via Server-Sent Events (SSE).
No formatting markers (no dashes, asterisks, bullet points) — clean text only.
"""
import json
import logging
import os

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from ai_config_loader import get_ai_config

logger = logging.getLogger(config.APP_NAME)

router = APIRouter(prefix="/resume-content", tags=["Resume Content"])


class EnhanceRequest(BaseModel):
    sectionTitle: str = Field(..., description="Name of the resume section (e.g. Summary, Experience, Skills)")
    sectionType: str = Field("paragraph", description="Section type: paragraph | bullets | tags")
    content: str = Field(..., description="User-typed draft content for this section")
    role: str = Field("", description="Optional job role / profession for additional context")


def _build_system_prompt() -> str:
    return (
        "You are a professional resume writer with expertise in crafting ATS-optimised, "
        "highly impactful resume content. "
        "Enhance the provided draft text for the resume section given. "
        "\n\nStrict rules:"
        "\n- Use strong, active verbs and quantify achievements where facts allow."
        "\n- Keep language concise, professional, and results-focused."
        "\n- Do NOT use bullet points, dashes, asterisks, hyphens, or any markdown/formatting symbols."
        "\n- Do NOT add section headings or labels."
        "\n- Do NOT add placeholder text like [Year], [Company], [Number], or [Role]."
        "\n- Return ONLY the enhanced content text — nothing else, no preamble, no explanation."
    )


@router.post("/enhance")
async def enhance_section(req: EnhanceRequest):
    """
    Stream AI-enhanced resume section content.
    SSE events: {text: str} deltas while generating, then {done: true}  .
    On error: {error: str}.
    """
    cfg = await get_ai_config()

    system_prompt = _build_system_prompt()
    user_prompt = (
        f"Resume Section: {req.sectionTitle}\n"
        f"Content Type: {req.sectionType}\n"
        + (f"Role / Profession: {req.role}\n" if req.role else "")
        + f"\nDraft content:\n{req.content}\n\n"
        "Write the enhanced resume content for this section:"
    )

    async def generate():
        try:
            import openai

            api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                yield (
                    "data: "
                    + json.dumps({"error": "AI not configured. Set API key in Superadmin → Settings → AI Config."})
                    + "\n\n"
                )
                return

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
                timeout=float(cfg.get("timeout", 60)),
            )

            model = cfg.get("model", "gpt-3.5-turbo")
            temperature = float(cfg.get("temperature", 0.6))
            max_tokens = min(int(cfg.get("maxTokens", 1024)), 500)

            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield "data: " + json.dumps({"text": delta}) + "\n\n"

            yield "data: " + json.dumps({"done": True}) + "\n\n"

        except Exception as exc:
            logger.error("Resume content enhance error: %s", exc)
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
