"""GCS upload helpers for streaming artifacts (chunks, merged videos, screenshots)."""

from __future__ import annotations

import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)
_client = None


def _get_client():
    global _client
    if not config.GCS_BUCKET or not config.GCP_PROJECT_ID:
        return None
    if _client is not None:
        return _client
    try:
        from google.cloud import storage

        creds = config.get_gcs_credentials()
        if creds is None:
            logger.warning("GCS client init failed: credentials must come from .env")
            return None
        _client = storage.Client(project=config.GCP_PROJECT_ID, credentials=creds)
        return _client
    except Exception as e:
        logger.warning("GCS client init failed: %s", e)
        return None


def _blob_path(*parts: str) -> str:
    prefix = (config.GCS_UPLOAD_PREFIX or "").strip().strip("/")
    rel = "/".join(p.strip("/") for p in parts if p)
    return f"{prefix}/{rel}" if prefix else rel


def _upload_file(local_file: Path, blob_rel_path: str, content_type: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    if not local_file.exists() or local_file.stat().st_size == 0:
        return False
    try:
        bucket = client.bucket(config.GCS_BUCKET)
        blob = bucket.blob(_blob_path(blob_rel_path))
        blob.upload_from_filename(str(local_file), content_type=content_type, timeout=config.GCS_UPLOAD_TIMEOUT)
        return True
    except Exception as e:
        logger.warning("GCS upload failed (%s): %s", blob_rel_path, e)
        return False


def upload_blob_to_gcs(blob_rel_path: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        bucket = client.bucket(config.GCS_BUCKET)
        blob = bucket.blob(_blob_path(blob_rel_path))
        blob.upload_from_string(data, content_type=content_type, timeout=config.GCS_UPLOAD_TIMEOUT)
        return True
    except Exception as e:
        logger.warning("GCS raw blob upload failed (%s): %s", blob_rel_path, e)
        return False


def upload_chunk_file(client_id: str, position_id: str, candidate_id: str, local_file: Path) -> bool:
    session_base = config.gcs_session_base(client_id, position_id, candidate_id)
    rel = f"{session_base}/chunks/{local_file.name}"
    return _upload_file(local_file, rel, "video/webm")


def upload_merged_video(client_id: str, position_id: str, candidate_id: str, local_file: Path) -> str | None:
    session_base = config.gcs_session_base(client_id, position_id, candidate_id)
    rel = f"{session_base}/merge/{local_file.name}"
    ok = _upload_file(local_file, rel, "video/mp4")
    if not ok:
        return None
    blob_path = _blob_path(rel)
    base_url = (config.GCP_STORAGE_BASE_URL or "https://storage.googleapis.com").rstrip("/")
    return f"{base_url}/{config.GCS_BUCKET}/{blob_path}"


def upload_screenshot_bytes(
    client_id: str,
    position_id: str,
    candidate_id: str,
    image_bytes: bytes,
    file_name: str,
    subfolder: str | None = None,
) -> bool:
    client = _get_client()
    if client is None or not image_bytes:
        return False
    session_base = config.gcs_session_base(client_id, position_id, candidate_id)
    rel = f"{session_base}/screenshots"
    if subfolder:
        rel = f"{rel}/{subfolder.strip('/')}"
    rel = f"{rel}/{file_name}"
    try:
        bucket = client.bucket(config.GCS_BUCKET)
        blob = bucket.blob(_blob_path(rel))
        blob.upload_from_string(image_bytes, content_type="image/png", timeout=config.GCS_UPLOAD_TIMEOUT)
        return True
    except Exception as e:
        logger.warning("GCS screenshot upload failed (%s): %s", rel, e)
        return False


def delete_chunk_blobs_after_merge(client_id: str, position_id: str, candidate_id: str) -> int:
    client = _get_client()
    if client is None:
        return 0
    session_base = config.gcs_session_base(client_id, position_id, candidate_id)
    prefix = _blob_path(f"{session_base}/chunks/")
    deleted = 0
    try:
        bucket = client.bucket(config.GCS_BUCKET)
        for blob in bucket.list_blobs(prefix=prefix):
            try:
                blob.delete()
                deleted += 1
            except Exception as e:
                logger.warning("Failed to delete chunk blob %s: %s", blob.name, e)
    except Exception as e:
        logger.warning("Listing chunk blobs failed for cleanup: %s", e)
    return deleted


def list_report_screenshot_urls(
    client_id: str,
    position_id: str,
    candidate_id: str,
    limit_per_group: int = 4,
) -> dict:
    """Return categorized screenshot URLs for report UI from GCS.

    Categories:
    - calibration: initial calibration photos (used for profile picture)
    - noFace: no-face violation screenshots
    - multipleFaces: multiple-face violation screenshots
    - allDirection: looking left/right/up/down screenshots
    """
    client = _get_client()
    if client is None:
        return {
            "profilePicture": "",
            "calibration": [],
            "noFace": [],
            "multipleFaces": [],
            "allDirection": [],
            "all": [],
        }

    session_base = config.gcs_session_base(client_id, position_id, candidate_id)
    screenshots_rel = f"{session_base}/screenshots/"
    prefix = _blob_path(screenshots_rel)
    base_url = (config.GCP_STORAGE_BASE_URL or "https://storage.googleapis.com").rstrip("/")

    calibration = []
    no_face = []
    multiple_faces = []
    all_direction = []

    try:
        bucket = client.bucket(config.GCS_BUCKET)
        blobs = list(bucket.list_blobs(prefix=prefix))

        # Newest first so the most relevant screenshots are surfaced.
        blobs.sort(
            key=lambda b: (
                str(getattr(b, "updated", "") or ""),
                str(getattr(b, "time_created", "") or ""),
                b.name,
            ),
            reverse=True,
        )

        for blob in blobs:
            name = str(blob.name or "")
            if not name or name.endswith("/"):
                continue

            rel = name[len(prefix):] if name.startswith(prefix) else name
            rel_l = rel.lower()
            url = f"{base_url}/{config.GCS_BUCKET}/{name}"

            if rel_l.startswith("calibration/"):
                calibration.append(url)
            elif rel_l.startswith("noface/"):
                no_face.append(url)
            elif rel_l.startswith("multiple/"):
                multiple_faces.append(url)
            elif rel_l.startswith("alldirection/"):
                all_direction.append(url)

        profile_picture = (calibration[0] if calibration else "")

        no_face = no_face[: max(0, int(limit_per_group))]
        multiple_faces = multiple_faces[: max(0, int(limit_per_group))]
        all_direction = all_direction[: max(0, int(limit_per_group))]

        combined = []
        if profile_picture:
            combined.append(profile_picture)
        combined.extend(no_face)
        combined.extend(multiple_faces)
        combined.extend(all_direction)

        return {
            "profilePicture": profile_picture,
            "calibration": calibration[: max(0, int(limit_per_group))],
            "noFace": no_face,
            "multipleFaces": multiple_faces,
            "allDirection": all_direction,
            "all": combined,
        }
    except Exception as e:
        logger.warning("GCS screenshot listing failed (%s): %s", screenshots_rel, e)
        return {
            "profilePicture": "",
            "calibration": [],
            "noFace": [],
            "multipleFaces": [],
            "allDirection": [],
            "all": [],
        }
