"""Create the streaming folder placeholder in GCS so it appears in the bucket console."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import config
from gcs_upload import upload_blob_to_gcs


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: python create_streaming_gcs_folder.py <client_id> <position_id> <candidate_id>")
        return 1

    client_id = sys.argv[1].strip()
    position_id = sys.argv[2].strip()
    candidate_id = sys.argv[3].strip()

    if not client_id or not position_id or not candidate_id:
        print("client_id, position_id, and candidate_id are required.")
        return 1

    if not config.GCS_BUCKET or not config.GCS_UPLOAD_PREFIX or not config.GCS_STREAM_FOLDER or not config.GCP_PROJECT_ID:
        print("GCS_BUCKET, GCS_UPLOAD_PREFIX, GCS_STREAM_FOLDER, and GCP_PROJECT_ID must be configured in .env.")
        return 1

    placeholder_paths = [
        f"{client_id}/{position_id}/{config.GCS_STREAM_FOLDER}/.keep",
    ]
    session_base = config.gcs_session_base(client_id, position_id, candidate_id)
    placeholder_paths.extend([
        f"{session_base}/chunks/.keep",
        f"{session_base}/merge/.keep",
        f"{session_base}/screenshots/.keep",
    ])

    created = []
    for placeholder_rel in placeholder_paths:
        ok = upload_blob_to_gcs(placeholder_rel, b"", content_type="application/octet-stream")
        if not ok:
            print(f"Failed to create placeholder in GCS: {placeholder_rel}")
            print("Set GOOGLE_APPLICATION_CREDENTIALS or GCP_CLIENT_EMAIL/GCP_PRIVATE_KEY in Streaming AI/backend/.env.")
            return 1
        full_blob = f"{config.GCS_UPLOAD_PREFIX}/{placeholder_rel}" if config.GCS_UPLOAD_PREFIX else placeholder_rel
        created.append(f"gs://{config.GCS_BUCKET}/{full_blob}")

    for path in created:
        print(f"Created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
