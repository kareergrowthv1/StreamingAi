"""Configuration for Streaming AI backend. All base URLs from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9000"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads")).resolve()
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "./chunks")).resolve()
MERGED_DIR = Path(os.getenv("MERGED_DIR", "./merged")).resolve()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# CORS: comma-separated origins.
# Default covers all local frontend ports:
#   4000 = SuperadminFrontend, 4001 = AdminFrontend, 4002 = CandidateTest, 4003 = CandidateFrontend
_cors = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:4000,http://127.0.0.1:4000,"
    "http://localhost:4001,http://127.0.0.1:4001,"
    "http://localhost:4002,http://127.0.0.1:4002,"
    "http://localhost:4003,http://127.0.0.1:4003,"
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:5174,http://127.0.0.1:5174",
)
# If a single "*" is provided use a special sentinel so main.py can apply allow_origin_regex
CORS_ORIGINS: list = [o.strip() for o in _cors.split(",") if o.strip()]

CELERY_ENABLED = os.getenv("CELERY_ENABLED", "false").lower() == "true"
# AssemblyAI (STT) – same as KareerGrowth AiService; use for real-time streaming
ASSEMBLYAI_API_KEY: str = os.getenv(
    "ASSEMBLYAI_API_KEY",
    "d7041fa22c3648f7ae005e6ad16331aa",
).strip()

# Backend base URLs – WebSocket session uses these for API calls (no frontend API calls during test)
ADMIN_BACKEND_URL = os.getenv("ADMIN_BACKEND_URL", "http://localhost:8002").rstrip("/")
CANDIDATE_BACKEND_URL = os.getenv("CANDIDATE_BACKEND_URL", "http://localhost:8003").rstrip("/")

# Superadmin backend – dynamic AI config (GET /superadmin/settings/ai-config), no hardcoded keys
SUPERADMIN_BACKEND_URL = (
    os.getenv("SUPERADMIN_BACKEND_URL") or os.getenv("VITE_AUTH_API_URL", "http://localhost:8001")
).rstrip("/")
SUPERADMIN_SERVICE_TOKEN = (
    os.getenv("SUPERADMIN_SERVICE_TOKEN") or os.getenv("INTERNAL_SERVICE_TOKEN", "")
)

# AdminBackend service token – for resume-ats (fetch JD/resume) and verify-token (AI routes)
ADMIN_SERVICE_TOKEN = os.getenv("ADMIN_SERVICE_TOKEN", "")

APP_NAME = "Streaming AI"

# MongoDB — used by report_generator for assessment_reports collection
MONGODB_URL = os.getenv("MONGODB_URL", os.getenv("MONGODB_URI", ""))
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", os.getenv("MONGODB_DB", "kareergrowth"))

# Service token for calling AdminBackend internal routes
INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", os.getenv("ADMIN_SERVICE_TOKEN", ""))

# Ensure dirs exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)
