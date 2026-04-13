import json
import logging
import os
import httpx
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List

import config
from ai_config_loader import get_ai_config

logger = logging.getLogger(config.APP_NAME)

router = APIRouter(prefix="/fake-offer", tags=["Fake Offer Detection"])

@router.get("/ping")
async def ping():
    return {"status": "ok", "service": "fake-offer"}

class OfferVerificationRequest(BaseModel):
    text: str = Field(..., description="Extracted text from the offer letter")
    companyName: Optional[str] = Field(None, description="Reported company name")
    website: Optional[str] = Field(None, description="Reported company website")
    address: Optional[str] = Field(None, description="Reported company address")
    ctc: Optional[str] = Field(None, description="Reported annual CTC")

class OfferVerificationResponse(BaseModel):
    isOfferLetter: bool = Field(..., description="Whether the document is actually an offer letter")
    fraudRiskScore: str
    companyAuthenticityScore: float
    structuralValidityScore: float
    languageProfessionalismScore: float
    legalComplianceScore: float
    redFlags: List[str]
    genuineIndicators: List[str]
    finalVerdict: str
    confidenceLevel: float

def _get_openai_client(cfg: dict):
    """Build OpenAI/Groq client from dynamic config."""
    try:
        import openai
        api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Set in Superadmin Settings > AI Config.")
        return openai.OpenAI(
            api_key=api_key,
            base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
            timeout=cfg.get("timeout", 300),
            max_retries=cfg.get("maxRetries", 3),
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="openai package not installed.",
        )

@router.post("/verify", response_model=OfferVerificationResponse)
async def verify_offer(request: OfferVerificationRequest = Body(...)):
    """Analyze offer letter text for fraud indicators and authenticity."""
    cfg = await get_ai_config()
    client = _get_openai_client(cfg)
    model = cfg.get("model", "gpt-4o") # Fallback to gpt-4o if not specified
    
    # Enrich the prompt with user-provided metadata
    metadata_context = ""
    if request.companyName or request.website or request.address or request.ctc:
        metadata_context = "REPORTED METADATA BY USER:\n"
        if request.companyName: metadata_context += f"- Company Name: {request.companyName}\n"
        if request.website: metadata_context += f"- Website: {request.website}\n"
        if request.address: metadata_context += f"- Address: {request.address}\n"
        if request.ctc: metadata_context += f"- Claimed CTC: {request.ctc} LPA\n"
        metadata_context += "\nValidate these against the document text.\n\n"

    system_prompt = """
You are a senior HR compliance auditor and corporate fraud investigator specializing in recruitment fraud detection.
Your FIRST task is to determine if the provided document is actually a job offer letter.

GUIDELINES:
- If the document is a RESUME, ID CARD, PASSPORT, RANDOM ARTICLE, or NOT an offer letter, you MUST:
  1. Set "isOfferLetter" to false.
  2. Set "fraudRiskScore" to "HIGH".
  3. Set "companyAuthenticityScore" to 0.
  4. Set "structuralValidityScore" to 0.
  5. Set "languageProfessionalismScore" to 0.
  6. Set "legalComplianceScore" to 0.
  7. Set "confidenceLevel" to 0.
  8. In "finalVerdict", state clearly that the document is NOT an offer letter and cannot be verified.
- If the document IS an offer letter, evaluate its authenticity based on standard HR practices.

Return ONLY a JSON object with the following structure:
{
  "isOfferLetter": boolean,
  "fraudRiskScore": "LOW" | "MEDIUM" | "HIGH",
  "companyAuthenticityScore": <number 0-10>,
  "structuralValidityScore": <number 0-10>,
  "languageProfessionalismScore": <number 0-10>,
  "legalComplianceScore": <number 0-10>,
  "redFlags": ["reason1", "reason2", ...],
  "genuineIndicators": ["point1", "point2", ...],
  "finalVerdict": "2-3 sentence summary",
  "confidenceLevel": <number 0-100>
}
"""

    user_prompt = f"""
{metadata_context}
Perform a multi-layer analysis of the following offer letter content:

--------------------------------------------------
DOCUMENT CONTENT:
{request.text}
--------------------------------------------------

EVALUATION FRAMEWORK:
1. COMPANY AUTHENTICITY: Verify realistic name, matching domains (no gmail/yahoo for corporate), complete address, and valid formats (CIN/GST if present).
2. STRUCTURAL VALIDITY: Look for letterhead, date, candidate name, CTC breakdown, probation, notice period, and authorized signatory.
3. FINANCIAL FRAUD: Flag requests for security deposits, laptop fees, registration charges, or interviews skipped.
4. PROFESSIONALISM: Grammar, spelling, formatting, and consistent fonts.
5. LEGAL COMPLIANCE: Mention of EPF, ESI, and Indian employment norms.

Be strict. If payment is requested, risk is automatically HIGH.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        logger.exception("Fake offer analysis failed: %s", e)
        raise HTTPException(status_code=502, detail=f"AI provider error: {str(e)}")
