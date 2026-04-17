"""
Screening document and round-question helpers for Round 1 & 2.
Used by main.py for building categories, flat Q&A lists, and next-unanswered logic.
"""

import re


def parse_seconds(value, fallback: int) -> int:
    """Parse numeric/string durations into seconds (e.g., '10 secs', '2 mins', '01:30')."""
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return int(value)
    raw = str(value).strip().lower()
    if not raw:
        return fallback
    if raw.isdigit():
        return int(raw)
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)$", raw)
    if m:
        amount = float(m.group(1))
        unit = m.group(2)
        if unit.startswith("h"):
            return int(round(amount * 3600))
        if unit.startswith("m"):
            return int(round(amount * 60))
        return int(round(amount))
    if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", raw):
        parts = [int(x) for x in raw.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return fallback


def sort_conv_key(k) -> int:
    """Sort key for conversationQuestion1, conversationQuestion2, ..."""
    if isinstance(k, str) and k.startswith("conversationQuestion"):
        n = k.replace("conversationQuestion", "").strip()
        return int(n) if n.isdigit() else 0
    return 0


def build_flat_qa_list(screening_doc: dict, round_num: int) -> list:
    """
    Build flat list of (conv_key, pair_idx, question, answer) in display order.
    Index in list = flat_index for frontend.
    """
    if not screening_doc or round_num not in (1, 2):
        return []
    cat_key = "generalQuestion" if round_num == 1 else "positionSpecificQuestion"
    categories = (screening_doc or {}).get("categories") or {}
    cat = categories.get(cat_key) or {}
    conv_sets = cat.get("conversationSets") or {}
    flat = []
    for key in sorted(conv_sets.keys(), key=sort_conv_key):
        pairs = conv_sets.get(key) or []
        if not isinstance(pairs, list):
            continue
        for pi, p in enumerate(pairs):
            if isinstance(p, dict):
                flat.append(
                    (
                        key,
                        pi,
                        (p.get("question") or "").strip(),
                        (p.get("answer") or "").strip(),
                    )
                )
    return flat


def flat_index_to_conv_and_pair(
    screening_doc: dict, round_num: int, flat_index: int
) -> tuple:
    """Map flat_index to (conv_key, pair_idx). Returns (None, None) if out of range."""
    flat = build_flat_qa_list(screening_doc, round_num)
    if flat_index < 0 or flat_index >= len(flat):
        return (None, None)
    return (flat[flat_index][0], flat[flat_index][1])


def get_next_unanswered_from_screening(
    screening_doc: dict, round_num: int
) -> tuple:
    """
    From updated screening doc, get the next question (by flat index) that still has empty answer.
    Returns (flat_index_0based, question_text) or (None, None) if all answered in this round.
    """
    flat = build_flat_qa_list(screening_doc, round_num)
    for i, (_ck, _pi, q, a) in enumerate(flat):
        if not a:
            return (i, q or "")
    return (None, None)


def followup_count_for_conv_key(
    screening_doc: dict, round_num: int, conv_key: str
) -> int:
    """Number of follow-ups (pairs beyond the first) in this conversation set."""
    if not screening_doc or round_num not in (1, 2):
        return 0
    cat_key = "generalQuestion" if round_num == 1 else "positionSpecificQuestion"
    conv_sets = (screening_doc.get("categories") or {}).get(cat_key) or {}
    conv_sets = conv_sets.get("conversationSets") or {}
    pairs = conv_sets.get(conv_key) or []
    if not isinstance(pairs, list):
        return 0
    return max(0, len(pairs) - 1)


def questions_for_round(data: dict, round_number: int) -> list:
    """Extract questions for round from Admin question-sections response. Matches frontend getRoundQuestions."""
    if not data:
        return []
    if round_number == 1:
        gq = data.get("generalQuestions") or data.get("generalQuestion") or {}
        return gq.get("questions") if isinstance(gq, dict) else []
    if round_number == 2:
        pq = data.get("positionSpecificQuestions") or data.get("positionSpecificQuestion") or {}
        return pq.get("questions") if isinstance(pq, dict) else []
    if round_number == 3:
        cq = data.get("codingQuestions")
        return list(cq) if isinstance(cq, list) else []
    if round_number == 4:
        aq = data.get("aptitudeQuestions")
        return list(aq) if isinstance(aq, list) else []
    return []


def build_screening_categories_from_sections(data: dict) -> dict:
    """
    Build categories payload for candidate-interviews POST from Admin question-sections.
    generalQuestion / positionSpecificQuestion with conversationSets.conversationQuestion1, 2, 3... (answer "").
    """
    categories = {}
    # generalQuestion
    gq = (data or {}).get("generalQuestions") or {}
    g_questions = gq.get("questions") if isinstance(gq, dict) else []
    g_sets = {}
    for i, q in enumerate(g_questions or []):
        key = f"conversationQuestion{i + 1}"
        qd = q if isinstance(q, dict) else {}
        g_sets[key] = [{
            "question": qd.get("question", ""),
            "answer": "",
            "answerTime": parse_seconds(qd.get("answerTime", qd.get("timeToAnswer")), 120),
            "prepareTime": parse_seconds(qd.get("prepareTime", qd.get("timeToPrepare")), 10),
        }]
    if g_sets:
        categories["generalQuestion"] = {"conversationSets": g_sets}
    # positionSpecificQuestion
    pq = (data or {}).get("positionSpecificQuestions") or {}
    p_questions = pq.get("questions") if isinstance(pq, dict) else []
    p_sets = {}
    for i, q in enumerate(p_questions or []):
        key = f"conversationQuestion{i + 1}"
        qd = q if isinstance(q, dict) else {}
        p_sets[key] = [{
            "question": qd.get("question", ""),
            "answer": "",
            "answerTime": parse_seconds(qd.get("answerTime", qd.get("timeToAnswer")), 120),
            "prepareTime": parse_seconds(qd.get("prepareTime", qd.get("timeToPrepare")), 10),
        }]
    if p_sets:
        categories["positionSpecificQuestion"] = {"conversationSets": p_sets}
    return categories
