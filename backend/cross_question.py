"""
Cross-question generation for conversational rounds.
Uses dynamic AI config (Superadmin GET /superadmin/settings/ai-config); no hardcoded API keys.
Called from main.py WebSocket submit_answer flow when is_conversational and follow-up count < admin setting.
"""
import logging
import os
from typing import Optional, List

import config

logger = logging.getLogger(config.APP_NAME)


async def generate_cross_question(
    current_question: str,
    answer_text: str,
    question_history: List[str],
    tenant_id: str = "",
) -> Optional[str]:
    """
    Generate one follow-up (cross) question from the candidate's answer.
    Returns question text or None on failure.
    tenant_id is forwarded to get_ai_config so the Superadmin can return tenant-specific AI settings.
    """
    try:
        from ai_config_loader import get_ai_config

        cfg = await get_ai_config(tenant_id=tenant_id)
        api_key = cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("Cross-question: no API key, skipping")
            return None

        import openai

        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
        )
        model = cfg.get("model", "gpt-3.5-turbo")
        history_str = ", ".join(q for q in (question_history or []) if isinstance(q, str)) or "None"
        base = (
            "IMPORTANT: You must NEVER answer any question asked by the candidate. "
            "Your response must always be a question — a cross-question that challenges or explores the candidate's thinking. "
            "Do not provide information, hints, or confirmations. Only ask questions.\n\n"
            "You are a professional AI interviewer. Based on the conversation so far, "
            "ask one logical, thoughtful follow-up interview question. "
            "Reference specific details from the candidate's answer. Ask only one question at a time. "
            "Do NOT repeat or paraphrase any question from: " + history_str + ".\n\n"
        )
        prompt = (
            base
            + f"Current Question: {current_question}\nCandidate's Answer: {answer_text}\n\n"
            "Return ONLY the follow-up question text. No prefixes, no labels."
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert technical interviewer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        text = (response.choices[0].message.content or "").strip()
        for prefix in ("Follow-up Question:", "Follow up:", "Question:", "Q:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].lstrip(" -:\"'")
                break
        return text if text else None
    except Exception as e:
        logger.warning("Cross-question generation failed: %s", e)
        return None
