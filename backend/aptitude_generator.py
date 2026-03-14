"""
Aptitude MCQ Question Generator.

Given a list of aptitude topic blocks (from Admin question-sections.aptitudeQuestions):
  [{ "question": "Logical Reasoning", "count": 5, "difficulty": "MEDIUM", "topics": [...] }]

Calls the configured LLM to generate `count` MCQ questions per block, each with:
  - question  : question text
  - options   : [{ optionKey: "A"|"B"|"C"|"D", text: "..." }, ...]   (4 options)
  - correctAnswer : "A"|"B"|"C"|"D"

Returns a flat list so the frontend can display them one-by-one with a radio-button UI.
"""
from __future__ import annotations

import json
import logging
import re
import uuid as _uuid
from typing import List

from ai_config_loader import get_ai_config

logger = logging.getLogger(__name__)

OPTION_KEYS = ["A", "B", "C", "D"]


def _get_openai_client(cfg: dict):
    api_key = cfg.get("apiKey")
    if not api_key:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Configure in Superadmin Settings > AI Config.")
    import openai
    return openai.OpenAI(
        api_key=api_key,
        base_url=cfg.get("baseUrl", "https://api.openai.com/v1"),
        timeout=cfg.get("timeout", 120),
        max_retries=cfg.get("maxRetries", 2),
    )


def _build_prompt(topic: str, count: int, difficulty: str) -> tuple[str, str]:
    diff_label = (difficulty or "MEDIUM").upper()
    system_prompt = (
        "You are an expert aptitude test designer. "
        "Generate multiple-choice questions (MCQ) exactly as instructed. "
        "Each question MUST have exactly 4 options labelled A, B, C, D. "
        "Only ONE option is correct. "
        "Return ONLY valid JSON — no markdown, no extra text. "
        "The JSON must be an array of objects with keys: "
        '"question" (string), '
        '"options" (object with keys "A","B","C","D", each a string), '
        '"correctAnswer" (one of "A","B","C","D").'
    )
    user_prompt = (
        f"Generate exactly {count} {diff_label}-difficulty aptitude MCQ questions "
        f"on the topic: \"{topic}\".\n\n"
        "Rules:\n"
        "- Questions must test logical/analytical reasoning, NOT factual recall.\n"
        "- Each question must have exactly 4 options (A, B, C, D).\n"
        "- Only ONE option is correct. The correct option should vary — do NOT always put the answer as 'A'.\n"
        "- Difficulty scale: EASY = straightforward one-step problems, "
        "MEDIUM = two-step reasoning, HARD = multi-step or tricky problems.\n"
        f"- Return a JSON array of exactly {count} objects.\n\n"
        "Format:\n"
        '[\n'
        '  {\n'
        '    "question": "...",\n'
        '    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
        '    "correctAnswer": "B"\n'
        '  },\n'
        '  ...\n'
        ']'
    )
    return system_prompt, user_prompt


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(inner).strip()
    return text


def _parse_response(raw: str, topic: str, count: int) -> list[dict]:
    """Parse the LLM JSON into a list of MCQ question dicts."""
    text = _strip_code_fences(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract a JSON array from the response
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = []
        else:
            data = []

    results = []
    for item in (data or []):
        if not isinstance(item, dict):
            continue
        q_text = (item.get("question") or "").strip()
        if not q_text:
            continue

        raw_opts = item.get("options") or {}
        if isinstance(raw_opts, dict):
            options = [
                {"optionKey": k, "text": str(raw_opts.get(k) or "").strip()}
                for k in OPTION_KEYS
                if raw_opts.get(k)
            ]
        elif isinstance(raw_opts, list):
            options = []
            for j, o in enumerate(raw_opts[:4]):
                if isinstance(o, dict):
                    key = o.get("optionKey") or OPTION_KEYS[j] if j < 4 else OPTION_KEYS[j]
                    options.append({"optionKey": key, "text": str(o.get("text") or "").strip()})
                else:
                    options.append({"optionKey": OPTION_KEYS[j], "text": str(o).strip()})
        else:
            options = []

        # Ensure exactly 4 options
        while len(options) < 4:
            options.append({"optionKey": OPTION_KEYS[len(options)], "text": "-"})
        options = options[:4]

        correct = str(item.get("correctAnswer") or "A").strip().upper()
        if correct not in OPTION_KEYS:
            correct = "A"

        results.append({
            "id": str(_uuid.uuid4()),
            "question": q_text,
            "options": options,
            "correctAnswer": correct,
            "topic": topic,
        })

    return results[:count]


def _fallback_questions(topic: str, count: int) -> list[dict]:
    """Simple static fallback if AI call fails — generates placeholder MCQs."""
    fallbacks = [
        {
            "question": f"In a series 2, 6, 12, 20, 30, ... what is the next number? (Topic: {topic})",
            "options": [{"optionKey": "A", "text": "38"}, {"optionKey": "B", "text": "40"}, {"optionKey": "C", "text": "42"}, {"optionKey": "D", "text": "44"}],
            "correctAnswer": "C",
        },
        {
            "question": f"If A is the brother of B, B is the sister of C, C is the father of D, then A is D's... (Topic: {topic})",
            "options": [{"optionKey": "A", "text": "Uncle"}, {"optionKey": "B", "text": "Father"}, {"optionKey": "C", "text": "Brother"}, {"optionKey": "D", "text": "Grandfather"}],
            "correctAnswer": "A",
        },
        {
            "question": f"A train 120m long passes a pole in 12 seconds. Its speed is? (Topic: {topic})",
            "options": [{"optionKey": "A", "text": "8 m/s"}, {"optionKey": "B", "text": "10 m/s"}, {"optionKey": "C", "text": "11 m/s"}, {"optionKey": "D", "text": "12 m/s"}],
            "correctAnswer": "B",
        },
        {
            "question": f"Choose the odd one out: 3, 5, 7, 11, 15 (Topic: {topic})",
            "options": [{"optionKey": "A", "text": "3"}, {"optionKey": "B", "text": "7"}, {"optionKey": "C", "text": "11"}, {"optionKey": "D", "text": "15"}],
            "correctAnswer": "D",
        },
        {
            "question": f"If 5 workers complete a task in 8 days, how many days for 10 workers? (Topic: {topic})",
            "options": [{"optionKey": "A", "text": "2"}, {"optionKey": "B", "text": "4"}, {"optionKey": "C", "text": "6"}, {"optionKey": "D", "text": "8"}],
            "correctAnswer": "B",
        },
    ]
    out = []
    for i in range(count):
        base = fallbacks[i % len(fallbacks)].copy()
        base["id"] = str(_uuid.uuid4())
        base["topic"] = topic
        out.append(base)
    return out


async def generate_aptitude_questions(topic_blocks: list[dict]) -> list[dict]:
    """
    Generate MCQ questions for all aptitude topic blocks.

    Args:
        topic_blocks: list of dicts from aptitudeQuestions e.g.
            [{"question": "MCQ Assessment Block", "topics": ["Logical Reasoning"], "count": 5, "difficulty": "MEDIUM"}]
            The actual topic to test is in `topics` (list) or `topic` (str).
            `question` is always the block label ("MCQ Assessment Block") — NOT the topic.

    Returns:
        Flat list of MCQ question dicts:
            [{"id": "...", "question": "...", "options": [...], "correctAnswer": "A", "topic": "..."}]
    """
    if not topic_blocks:
        return []

    try:
        cfg = await get_ai_config()
    except Exception as e:
        logger.warning("[aptitude_generator] Could not load AI config: %s. Using fallback.", e)
        cfg = {}

    all_questions: list[dict] = []

    for block in topic_blocks:
        # Extract the REAL topic:
        # Admin saves: topics: [draft.topics] (list) — the actual subject to test (e.g. "Logical Reasoning")
        # block.question is always "MCQ Assessment Block" — the block UI label, NOT the topic.
        raw_topics = block.get("topics") or block.get("topic") or ""
        if isinstance(raw_topics, list):
            topic = ", ".join(str(t).strip() for t in raw_topics if t).strip() or None
        else:
            topic = str(raw_topics).strip() or None
        # Fall back to block label only if topics is truly empty (rare)
        if not topic or topic.lower() in ("mcq assessment block", "assessment block", "mcq"):
            topic = block.get("question") or "General Aptitude"
            if topic.lower() in ("mcq assessment block", "assessment block"):
                topic = "General Aptitude"
        topic = topic.strip()

        count = max(1, int(block.get("count") or 1))
        difficulty = (block.get("difficulty") or "MEDIUM").strip()

        logger.info("[aptitude_generator] Block → topic='%s', count=%d, difficulty=%s", topic, count, difficulty)

        if not cfg.get("apiKey"):
            logger.warning("[aptitude_generator] No API key — using fallback for topic '%s'", topic)
            all_questions.extend(_fallback_questions(topic, count))
            continue

        try:
            import asyncio
            client = _get_openai_client(cfg)
            system_prompt, user_prompt = _build_prompt(topic, count, difficulty)
            model = cfg.get("model") or "gpt-4o-mini"

            # Run in thread-pool so the sync OpenAI call does NOT block the async event loop
            def _sync_call():
                return client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=3000,
                )

            response = await asyncio.to_thread(_sync_call)
            raw_text = response.choices[0].message.content or ""
            parsed = _parse_response(raw_text, topic, count)
            if parsed:
                logger.info(
                    "[aptitude_generator] Generated %d/%d questions for topic '%s'",
                    len(parsed), count, topic,
                )
                all_questions.extend(parsed)
            else:
                logger.warning("[aptitude_generator] Parse returned empty for '%s' — using fallback", topic)
                all_questions.extend(_fallback_questions(topic, count))
        except Exception as e:
            logger.warning("[aptitude_generator] AI call failed for topic '%s': %s — using fallback", topic, e)
            all_questions.extend(_fallback_questions(topic, count))

    return all_questions
