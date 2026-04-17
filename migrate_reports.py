"""
migrate_reports.py
──────────────────
Backfill existing Candidates_Report documents that have empty
screeningQuestions.generalQuestions / specificQuestions.

For each report that is missing Q&A data this script will:
  1. Find the matching candidate_interview_responses MongoDB doc.
  2. Extract Q&A pairs from categories.generalQuestion.conversationSets
     and categories.positionSpecificQuestion.conversationSets.
  3. Build a proper screeningQuestions structure with STATIC AI comments
     (no live LLM calls — placeholder text used).
  4. $set the screeningQuestions field back on the Candidates_Report doc.

Run from the `Streaming AI/backend/` directory:
    python migrate_reports.py [--dry-run] [--position-id <id>] [--candidate-id <id>]

  --dry-run         Print what would change without writing to MongoDB.
  --position-id     Limit migration to one positionId.
  --candidate-id    Limit migration to one candidateId.
"""

import argparse
import asyncio
import os
import sys

from motor.motor_asyncio import AsyncIOMotorClient

# ── load .env (if present) ────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        print(f"[migrate] Loaded .env from {_env_path}")
    else:
        # try parent dir
        _env_path2 = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(_env_path2):
            load_dotenv(_env_path2)
            print(f"[migrate] Loaded .env from {_env_path2}")
except ImportError:
    print("[migrate] python-dotenv not installed — relying on shell env vars")

MONGODB_URL = (
    os.environ.get("MONGODB_URL")
    or os.environ.get("MONGODB_URI")
    or ""
)
MONGODB_DB_NAME = (
    os.environ.get("MONGODB_DB_NAME")
    or os.environ.get("MONGODB_DB")
    or "kareergrowth"
)

COLLECTION_REPORTS = "Candidates_Report"
COLLECTION_INTERVIEWS = "candidate_interview_responses"
COLLECTION_CODING = "candidate_interview_coding_responses"
COLLECTION_APTITUDE = "candidate_interview_aptitude_responses"

STATIC_AI_COMMENT = "AI analysis not available for this session (retroactive migration)."


def _bson_clean(obj):
    """Recursively convert BSON/datetime types to JSON-safe Python types."""
    from datetime import datetime, date
    if isinstance(obj, dict):
        return {k: _bson_clean(v) for k, v in obj.items() if k != "_id"}
    elif isinstance(obj, list):
        return [_bson_clean(v) for v in obj]
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        # Handle ObjectId, Decimal128, and other BSON types
        obj_type = type(obj).__module__
        if "bson" in str(obj_type):
            return str(obj)
        return obj


# ── Q&A extraction (mirrors report_generator._extract_qa_pairs) ───────────────
def _extract_qa_pairs(interview_doc: dict, round_key: str) -> list:
    """
    Extract Q&A pairs from a candidate_interview_responses document.
    round_key: 'generalQuestion' | 'positionSpecificQuestion'
    """
    category = (interview_doc.get("categories") or {}).get(round_key) or {}
    conv_sets = category.get("conversationSets") or {}
    pairs = []
    for _ck, conv_list in conv_sets.items():
        if isinstance(conv_list, list):
            for item in conv_list:
                if isinstance(item, dict):
                    q = (item.get("question") or "").strip()
                    a = (item.get("answer") or "").strip()
                    if q or a:
                        pairs.append({"question": q, "answer": a})
    return pairs


def _build_screening_questions(general_pairs: list, specific_pairs: list) -> dict:
    """Build the screeningQuestions sub-document with static AI analysis."""
    return {
        "generalQuestions": [
            {
                "questionNumber": idx + 1,
                "question": q.get("question") or "",
                "candidateAnswer": q.get("answer") or "",
                "timeTaken": None,
                "aiComments": STATIC_AI_COMMENT,
                "aiRatings": None,
                "contentType": "general",
            }
            for idx, q in enumerate(general_pairs)
        ],
        "specificQuestions": [
            {
                "questionNumber": idx + 1,
                "question": q.get("question") or "",
                "candidateAnswer": q.get("answer") or "",
                "timeTaken": None,
                "aiComments": STATIC_AI_COMMENT,
                "aiRatings": None,
                "contentType": "technical",
            }
            for idx, q in enumerate(specific_pairs)
        ],
    }


def _needs_screening_migration(report_doc: dict) -> bool:
    """Return True if screeningQuestions inside unifiedReport is empty."""
    ur = report_doc.get("unifiedReport") or {}
    sq = ur.get("screeningQuestions") or {}
    gen = sq.get("generalQuestions") or []
    spec = sq.get("specificQuestions") or []
    return len(gen) == 0 and len(spec) == 0


def _needs_coding_migration(report_doc: dict) -> bool:
    """Return True if codingQuestionSets inside unifiedReport is empty."""
    ur = report_doc.get("unifiedReport") or {}
    cqs = ur.get("codingQuestionSets") or []
    return len(cqs) == 0


async def _fetch_with_dash_variants(collection, candidate_id: str, position_id: str):
    """Try all dash-variants of IDs to find a document."""
    cid_no = candidate_id.replace("-", "")
    pid_no = position_id.replace("-", "")
    for cid_try, pid_try in [
        (candidate_id, position_id),
        (cid_no, pid_no),
        (cid_no, position_id),
        (candidate_id, pid_no),
    ]:
        doc = await collection.find_one(
            {"candidateId": cid_try, "positionId": pid_try},
            sort=[("_id", -1)],
        )
        if doc:
            return doc
    return None


# ── Main migration ─────────────────────────────────────────────────────────────
async def run_migration(dry_run: bool, filter_position: str = "", filter_candidate: str = ""):
    if not MONGODB_URL:
        print("[migrate] ERROR: MONGODB_URL / MONGODB_URI env var not set. Aborting.")
        sys.exit(1)

    print(f"[migrate] Connecting to MongoDB db='{MONGODB_DB_NAME}' …")

    # macOS: Python's bundled certs may not include Atlas CA.
    # Use system cert bundle if present, otherwise allow TLS with system CAs.
    import certifi
    _tls_kwargs: dict = {}
    try:
        # certifi provides a trusted CA bundle that includes Atlas certs
        _tls_kwargs["tlsCAFile"] = certifi.where()
    except ImportError:
        # Fallback: use macOS system cert store
        if os.path.exists("/etc/ssl/cert.pem"):
            _tls_kwargs["tlsCAFile"] = "/etc/ssl/cert.pem"

    motor_client = AsyncIOMotorClient(
        MONGODB_URL,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=30000,
        **_tls_kwargs,
    )
    # Ping to confirm connection before proceeding
    try:
        await motor_client.admin.command("ping")
        print("[migrate] MongoDB ping OK.")
    except Exception as e:
        print(f"[migrate] ERROR: Cannot reach MongoDB — {e}")
        motor_client.close()
        sys.exit(1)
    db = motor_client[MONGODB_DB_NAME]

    col_reports = db[COLLECTION_REPORTS]
    col_interviews = db[COLLECTION_INTERVIEWS]
    col_coding = db[COLLECTION_CODING]
    col_aptitude = db[COLLECTION_APTITUDE]

    # Build query to find reports that need migration
    query: dict = {}
    if filter_position:
        query["positionId"] = filter_position
    if filter_candidate:
        query["candidateId"] = filter_candidate

    total_reports = await col_reports.count_documents(query)
    print(f"[migrate] Found {total_reports} report(s) matching base query.")

    stats = {"scanned": 0, "skipped_ok": 0, "no_interview_doc": 0, "no_qa_data": 0, "updated": 0, "dry_run": 0}

    cursor = col_reports.find(query)
    async for report in cursor:
        stats["scanned"] += 1
        position_id = report.get("positionId") or ""
        candidate_id = report.get("candidateId") or ""

        needs_screening = _needs_screening_migration(report)
        needs_coding = _needs_coding_migration(report)

        if not needs_screening and not needs_coding:
            stats["skipped_ok"] += 1
            print(f"  [skip]   {candidate_id}/{position_id} — already fully populated")
            continue

        update_payload: dict = {}

        # ── Screening Q&A ────────────────────────────────────────────────────
        if needs_screening:
            interview_doc = await _fetch_with_dash_variants(col_interviews, candidate_id, position_id)
            if not interview_doc:
                stats["no_interview_doc"] += 1
                print(f"  [miss]   {candidate_id}/{position_id} — no interview_responses doc found")
            else:
                general_pairs = _extract_qa_pairs(interview_doc, "generalQuestion")
                specific_pairs = _extract_qa_pairs(interview_doc, "positionSpecificQuestion")
                if not general_pairs and not specific_pairs:
                    stats["no_qa_data"] += 1
                    print(f"  [empty]  {candidate_id}/{position_id} — interview doc has no Q&A pairs yet")
                else:
                    screening_questions = _build_screening_questions(general_pairs, specific_pairs)
                    update_payload["unifiedReport.screeningQuestions"] = screening_questions
                    update_payload["screeningQuestions"] = screening_questions
                    print(
                        f"  [screen] {candidate_id}/{position_id} — "
                        f"{len(general_pairs)} general Q, {len(specific_pairs)} specific Q"
                        + (" [DRY RUN]" if dry_run else "")
                    )

        # ── Coding Q sets ────────────────────────────────────────────────────
        if needs_coding:
            coding_doc = await _fetch_with_dash_variants(col_coding, candidate_id, position_id)
            if coding_doc:
                cqs = coding_doc.get("codingQuestionSets") or []
                if cqs:
                    cqs = _bson_clean(cqs)  # strip BSON datetime/ObjectId before saving
                    update_payload["unifiedReport.codingQuestionSets"] = cqs
                    update_payload["codingQuestionSets"] = cqs
                    total_coding_qs = sum(len(s.get("questions") or []) for s in cqs)
                    print(
                        f"  [coding] {candidate_id}/{position_id} — "
                        f"{len(cqs)} set(s), {total_coding_qs} question(s)"
                        + (" [DRY RUN]" if dry_run else "")
                    )
                else:
                    print(f"  [coding] {candidate_id}/{position_id} — coding doc found but codingQuestionSets empty")
            else:
                print(f"  [coding] {candidate_id}/{position_id} — no coding_responses doc found (candidate may not have done coding round)")

        # ── Aptitude ─────────────────────────────────────────────────────────
        aptitude_doc = await _fetch_with_dash_variants(col_aptitude, candidate_id, position_id)
        if aptitude_doc:
            gq = aptitude_doc.get("generatedQuestions") or []
            if gq:
                # Build aptitude questions list for the report
                apt_questions = [
                    {
                        "question": q.get("question") or q.get("questionText") or q.get("text") or "",
                        "options": q.get("options") or [],
                        "candidateAnswer": q.get("answerText") or q.get("answer") or q.get("candidateAnswer") or "",
                        "correctAnswer": q.get("correctAnswer") or "",
                        "isCorrect": (
                            bool(q.get("correctAnswer")) and
                            str(q.get("answerText") or q.get("answer") or q.get("candidateAnswer") or "").strip()
                            == str(q.get("correctAnswer") or "").strip()
                        ),
                    }
                    for q in gq
                ]
                apt_assessment = {"questions": apt_questions, "overallAiReview": ""}
                update_payload["unifiedReport.aptitudeAssessment"] = apt_assessment
                update_payload["aptitudeAssessment"] = apt_assessment
                print(
                    f"  [aptitu] {candidate_id}/{position_id} — "
                    f"{len(apt_questions)} aptitude question(s)"
                    + (" [DRY RUN]" if dry_run else "")
                )

        if not update_payload:
            print(f"  [noop]   {candidate_id}/{position_id} — nothing to update")
            continue

        if not dry_run:
            await col_reports.update_one(
                {"positionId": position_id, "candidateId": candidate_id},
                {"$set": update_payload},
            )
            stats["updated"] += 1
        else:
            stats["dry_run"] += 1

    motor_client.close()

    print("\n── Migration Summary ──────────────────────────────────────────")
    print(f"  Reports scanned          : {stats['scanned']}")
    print(f"  Already fully populated (skipped): {stats['skipped_ok']}")
    print(f"  No interview doc found   : {stats['no_interview_doc']}")
    print(f"  Interview doc but no Q&A : {stats['no_qa_data']}")
    if dry_run:
        print(f"  Would update (dry run)   : {stats['dry_run']}")
    else:
        print(f"  Updated                  : {stats['updated']}")
    print("────────────────────────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Backfill screeningQuestions in Candidates_Report")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing to MongoDB")
    parser.add_argument("--position-id", default="", help="Limit to a single positionId")
    parser.add_argument("--candidate-id", default="", help="Limit to a single candidateId")
    args = parser.parse_args()

    if args.dry_run:
        print("[migrate] DRY RUN mode — no writes will be made.")

    asyncio.run(run_migration(
        dry_run=args.dry_run,
        filter_position=args.position_id,
        filter_candidate=args.candidate_id,
    ))


if __name__ == "__main__":
    main()
