"""
Streaming Backend — FastAPI app.
- Screenshot upload, chunk upload, WebSocket for proctoring events.
- File handling via os, shutil, pathlib.
- Merge: Celery when available, else sync in-process.
"""
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from merge_utils import do_merge_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Streaming Proctoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---
class ProctoringEvent(BaseModel):
    event: str = Field(..., description="no_face | multiple_faces | head_turned")
    confidence: float = Field(..., ge=0, le=1)
    timestamp: float = Field(..., description="Unix timestamp")


# --- Path helpers (pathlib) ---
def _screenshot_path(client_id: str, position_id: str, candidate_id: str) -> Path:
    base = config.UPLOAD_DIR / client_id / position_id / candidate_id
    base.mkdir(parents=True, exist_ok=True)
    return base / "screenshots"


def _chunks_base(client_id: str, position_id: str, candidate_id: str) -> Path:
    base = config.CHUNKS_DIR / client_id / position_id / candidate_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def _merged_path(client_id: str, position_id: str, candidate_id: str) -> Path:
    base = config.MERGED_DIR / client_id / position_id / candidate_id
    base.mkdir(parents=True, exist_ok=True)
    return base / "recording.mp4"


# --- REST: Screenshot ---
@app.post("/api/screenshot")
async def upload_screenshot(
    client_id: str = Form(...),
    position_id: str = Form(...),
    candidate_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Save screenshot from camera (getUserMedia + Canvas)."""
    dir_path = _screenshot_path(client_id, position_id, candidate_id)
    dir_path.mkdir(parents=True, exist_ok=True)
    # Simple naming: timestamp or sequence
    import time
    name = f"screenshot_{int(time.time() * 1000)}.png"
    file_path = dir_path / name
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        return {"success": True, "path": str(file_path), "name": name}
    except Exception as e:
        logger.exception("Screenshot upload failed")
        return {"success": False, "error": str(e)}


# --- REST: Video chunk (10-sec) ---
@app.post("/api/chunk")
async def upload_chunk(
    client_id: str = Form(...),
    position_id: str = Form(...),
    candidate_id: str = Form(...),
    index: int = Form(..., description="Chunk index (0, 1, 2, ...)"),
    file: UploadFile = File(...),
):
    """Save one 10-second video chunk from MediaRecorder."""
    base = _chunks_base(client_id, position_id, candidate_id)
    # Chunk file: chunk_000.webm, chunk_001.webm, ...
    name = f"chunk_{index:04d}.webm"
    file_path = base / name
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        return {"success": True, "path": str(file_path), "index": index}
    except Exception as e:
        logger.exception("Chunk upload failed")
        return {"success": False, "error": str(e)}


# --- REST: End test — trigger merge (Celery if available, else sync) ---
@app.post("/api/end-test")
async def end_test(
    client_id: str = Form(...),
    position_id: str = Form(...),
    candidate_id: str = Form(...),
):
    """End test: merge chunks with FFmpeg. Uses Celery if installed, else runs merge in-process."""
    try:
        from tasks import merge_video_chunks
        merge_video_chunks.delay(client_id, position_id, candidate_id)
        return {"success": True, "message": "Merge job queued"}
    except Exception as e:
        logger.warning("Celery unavailable (%s), running merge in-process", e)
        try:
            result = await asyncio.to_thread(do_merge_chunks, client_id, position_id, candidate_id)
            if result.get("success"):
                return {"success": True, "message": "Merge completed", "merged": result.get("merged")}
            return {"success": False, "error": result.get("error", "Merge failed")}
        except Exception as err:
            logger.exception("Merge failed")
            return {"success": False, "error": str(err)}


# --- WebSocket: Proctoring events (no_face, multiple_faces, head_turned) ---
@app.websocket("/ws/proctoring")
async def websocket_proctoring(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            # Optional: persist to DB (SQLAlchemy + PostgreSQL)
            logger.info("Proctoring event: %s", payload)
            # Echo or store; no response required
    except WebSocketDisconnect:
        logger.info("Proctoring WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)


@app.get("/api/merged/{client_id}/{position_id}/{candidate_id}")
def get_merged_video(client_id: str, position_id: str, candidate_id: str):
    """Stream the merged recording (e.g. for playback)."""
    out_file = config.MERGED_DIR / client_id / position_id / candidate_id / "recording.mp4"
    if not out_file.exists():
        raise HTTPException(status_code=404, detail="Recording not found")
    return FileResponse(out_file, media_type="video/mp4", filename="recording.mp4")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
