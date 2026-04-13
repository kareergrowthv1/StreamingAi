from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import logging
import json
import config
from ai_config_loader import get_ai_config
from verify_admin_token import verify_admin_token

logger = logging.getLogger(config.APP_NAME)

router = APIRouter(tags=["Email Template"])

class EmailTemplateAIRequest(BaseModel):
    mode: str = Field(..., description="'generate' or 'refine'")
    prompt: str = Field(..., description="User's prompt or instruction")
    currentBody: Optional[str] = Field(None, description="Current email body (for refinement)")
    variables: Optional[List[str]] = Field(None, description="List of available dynamic variables")

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

@router.post("/generate-email-template")
async def generate_email_template(request: EmailTemplateAIRequest = Body(...), _: bool = Depends(verify_admin_token)):
    """Generate or refine an email template body using AI."""
    
    available_vars = ", ".join(request.variables) if request.variables is not None else "{candidate_name}, {Position_title}, {company_name}"
    
    system_prompt = f"""
You are a professional HR and Recruitment Assistant. Your goal is to write high-quality, professional emails.
You MUST use dynamic variables in curly braces like {{candidate_name}}.

Available variables you can use:
{available_vars}

If the user provides a prompt, generate a complete email Subject and Body.
If the user provides a 'currentBody', refine and improve both the subject and the body while keeping the core message and variables intact.

Response format: Return a JSON object with two keys:
1. "subject": A professional subject line (can include variables).
2. "body": The complete email body.
"""

    if request.mode == "refine" and request.currentBody:
        user_prompt = f"Refine the following email based on this instruction: {request.prompt}\n\nCurrent Body:\n{request.currentBody}"
    else:
        user_prompt = f"Generate a professional email subject and body for: {request.prompt}"

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
