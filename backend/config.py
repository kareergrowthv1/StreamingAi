"""Configuration for Streaming AI backend. All base URLs from .env."""
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _build_ca_bundle_path() -> str:
    extra_ca_paths = []
    configured_extra = os.getenv("EXTRA_CA_CERT_PATHS", "").strip()
    if configured_extra:
        extra_ca_paths.extend([p.strip() for p in configured_extra.split(",") if p.strip()])

    mkcert_root = Path.home() / "Library" / "Application Support" / "mkcert" / "rootCA.pem"
    if mkcert_root.exists():
        extra_ca_paths.append(str(mkcert_root))

    unique_existing_paths = []
    seen = set()
    for path_str in extra_ca_paths:
        path = Path(path_str).expanduser().resolve()
        if path.exists() and str(path) not in seen:
            unique_existing_paths.append(path)
            seen.add(str(path))

    if not unique_existing_paths:
        return ""

    try:
        import certifi

        base_bundle = Path(certifi.where())
    except Exception:
        base_bundle = None

    bundle_path = Path(tempfile.gettempdir()) / "streaming-ai-ca-bundle.pem"
    bundle_parts = []

    if base_bundle and base_bundle.exists():
        bundle_parts.append(base_bundle.read_text(encoding="utf-8"))

    for path in unique_existing_paths:
        bundle_parts.append(path.read_text(encoding="utf-8"))

    bundle_content = "\n".join(part.strip() for part in bundle_parts if part.strip()) + "\n"
    if not bundle_content.strip():
        return ""

    bundle_path.write_text(bundle_content, encoding="utf-8")
    return str(bundle_path)


CA_BUNDLE_PATH = _build_ca_bundle_path()
if CA_BUNDLE_PATH:
    os.environ["SSL_CERT_FILE"] = CA_BUNDLE_PATH
    os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE_PATH

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9000"))
SSL_KEYFILE = os.getenv("SSL_KEY_PATH", "").strip()
SSL_CERTFILE = os.getenv("SSL_CERT_PATH", "").strip()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads")).resolve()
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "./chunks")).resolve()
MERGED_DIR = Path(os.getenv("MERGED_DIR", "./merged")).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", "")

# GCP Cloud Storage (optional)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "").strip()
GCS_BUCKET = os.getenv("GCS_BUCKET", "").strip()
GCP_STORAGE_BASE_URL = os.getenv("GCP_STORAGE_BASE_URL", "").rstrip("/")
GCS_UPLOAD_PREFIX = os.getenv("GCS_UPLOAD_PREFIX", "").strip().strip("/")
GCS_STATIC_TENANT_PATH = os.getenv("GCS_STATIC_TENANT_PATH", "").strip().strip("/")
GCS_STREAM_FOLDER = os.getenv("GCS_STREAM_FOLDER", "").strip().strip("/")
GCS_UPLOAD_TIMEOUT = int(os.getenv("GCS_UPLOAD_TIMEOUT", "1200"))
GOOGLE_APPLICATION_CREDENTIALS = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip().strip('"').strip("'")
GCP_CLIENT_EMAIL = (os.getenv("GCP_CLIENT_EMAIL") or "").strip().strip('"').strip("'")
GCP_PRIVATE_KEY_ID = (os.getenv("GCP_PRIVATE_KEY_ID") or "").strip()
GCP_PRIVATE_KEY_RAW = (os.getenv("GCP_PRIVATE_KEY") or "").strip().strip('"').strip("'")

# Mediapipe Model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

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


def get_gcs_credentials():
    """Build Google credentials from env service account fields or JSON key path."""
    if GOOGLE_APPLICATION_CREDENTIALS and os.path.isfile(GOOGLE_APPLICATION_CREDENTIALS):
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
    if GCP_CLIENT_EMAIL and GCP_PRIVATE_KEY_RAW:
        from google.oauth2 import service_account

        key = GCP_PRIVATE_KEY_RAW.replace("\\n", "\n")
        info = {
            "type": "service_account",
            "project_id": GCP_PROJECT_ID or "",
            "private_key_id": GCP_PRIVATE_KEY_ID,
            "private_key": key,
            "client_email": GCP_CLIENT_EMAIL,
            "client_id": "",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "",
            "universe_domain": "googleapis.com",
        }
        return service_account.Credentials.from_service_account_info(info)
    return None


def local_streaming_base(client_id: str, position_id: str, candidate_id: str) -> Path:
    """Local canonical layout root: streaming/client/position/candidate."""
    return Path("streaming") / client_id / position_id / candidate_id


def local_chunks_dir(client_id: str, position_id: str, candidate_id: str) -> Path:
    return CHUNKS_DIR / local_streaming_base(client_id, position_id, candidate_id) / "chunks"


def local_merged_dir(client_id: str, position_id: str, candidate_id: str) -> Path:
    return MERGED_DIR / local_streaming_base(client_id, position_id, candidate_id) / "merged"


def local_screenshots_dir(client_id: str, position_id: str, candidate_id: str) -> Path:
    return MERGED_DIR / local_streaming_base(client_id, position_id, candidate_id) / "screenshots"


def gcs_session_base(client_id: str, position_id: str, candidate_id: str) -> str:
    """GCS path: static_tenant/streaming/client_id/position_id/candidate_id."""
    if not GCS_STATIC_TENANT_PATH:
        raise ValueError("GCS_STATIC_TENANT_PATH is required in .env")
    if not GCS_STREAM_FOLDER:
        raise ValueError("GCS_STREAM_FOLDER is required in .env")
    if not client_id or not position_id or not candidate_id:
        raise ValueError("client_id, position_id, and candidate_id are required for GCS session path")
    return f"{GCS_STATIC_TENANT_PATH}/{GCS_STREAM_FOLDER}/{client_id}/{position_id}/{candidate_id}"

# Ensure dirs exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)
