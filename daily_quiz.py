import json
import logging
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ai_config_loader import get_ai_config
import os

logger = logging.getLogger(__name__)

router = APIRouter()

class GenerateQuizRequest(BaseModel):
    category: Optional[str] = "software development"
    difficulty: Optional[str] = "Medium"

@router.post("/api/generate-daily-quiz")
async def generate_daily_quiz(request: GenerateQuizRequest):
    """Generate 5 daily quiz questions using the AI config from Superadmin."""
    logger.info(f"[API IN] POST /api/generate-daily-quiz | category={request.category} difficulty={request.difficulty}")
    
    prompt = f"""
    Generate exactly 5 multiple choice questions for a daily quiz on {request.category} (Difficulty: {request.difficulty}).
    Make the questions engaging and varied (e.g., conceptual, code snippets, best practices).
    
    Requirements:
    1. Each question must have exactly 4 options.
    2. Only ONE option can be correct.
    3. Provide a brief explanation for the correct answer.
    4. Return pure JSON only, in this exact format:
    [
        {{
            "question": "The question text",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correctAnswerIndex": 0,
            "explanation": "Brief reason why it's correct."
        }}
    ]
    Do not use markdown blocks (```json) or introductory text, just the raw JSON array.
    """

    try:
        cfg = await get_ai_config()
        # Fallback to OPENAI_API_KEY in env if not in DB config
        api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY") 
        if not api_key:
            # Fallback to GROQ API KEY here since Candidate backend used it earlier
            api_key = os.getenv("GROQ_API_KEY")
            
        if not api_key:
            raise HTTPException(status_code=500, detail="API key required. Set in Superadmin Settings > AI Config.")
            
        base_url = cfg.get("baseUrl", "https://api.openai.com/v1")
        model = cfg.get("model", "gpt-3.5-turbo")
        timeout = int(cfg.get("timeout", 120))
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are an expert technical Quiz Master. Return only pure JSON."},
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
                
            # Clean up potential markdown formatting
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "").strip()
            elif "```" in content:
                content = content.replace("```", "").strip()
                
            questions = json.loads(content)
            
            if not isinstance(questions, list) or len(questions) != 5:
                logger.warning(f"AI returned {len(questions) if isinstance(questions, list) else 'non-list'} items instead of 5.")
                
            return {"success": True, "data": questions}
            
    except httpx.HTTPStatusError as e:
        logger.exception("AI provider error: %s", e.response.text)
        raise HTTPException(status_code=502, detail=f"AI Provider Error: {e.response.status_code}")
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse AI response as JSON: %s", content)
        raise HTTPException(status_code=502, detail="Invalid JSON response from AI")
    except Exception as e:
        logger.exception("Failed to generate daily quiz")
        raise HTTPException(status_code=500, detail=str(e))
