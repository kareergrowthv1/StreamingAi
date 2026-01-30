"""
Shared merge logic: FFmpeg concat + delete chunks.
Used by Celery task (when available) and by sync fallback in main.
"""
import os
import shutil
import subprocess
from pathlib import Path

import config


def do_merge_chunks(client_id: str, position_id: str, candidate_id: str) -> dict:
    """
    Read all chunks in order, concatenate with FFmpeg into single MP4.
    Then delete the chunks directory. Returns {"success": bool, "merged": str, "error": str}.
    """
    chunks_base = config.CHUNKS_DIR / client_id / position_id / candidate_id
    if not chunks_base.exists():
        return {"success": False, "error": "Chunks directory not found"}

    chunk_files = sorted(chunks_base.glob("chunk_*.webm"), key=lambda p: p.name)
    if not chunk_files:
        return {"success": False, "error": "No chunk files found"}

    out_path = config.MERGED_DIR / client_id / position_id / candidate_id
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "recording.mp4"

    list_file = chunks_base / "concat_list.txt"
    try:
        with open(list_file, "w") as f:
            for p in chunk_files:
                # FFmpeg concat: escape single quotes in path
                path_str = p.absolute().as_posix().replace("'", "'\\''")
                f.write(f"file '{path_str}'\n")
    except Exception as e:
        return {"success": False, "error": str(e)}

    # Try concat with -c copy first; if that fails (e.g. webm chunk boundaries), re-encode
    cmd_copy = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_file),
    ]
    try:
        result = subprocess.run(
            cmd_copy, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            # Fallback: re-encode for compatibility (e.g. Chrome webm chunk boundaries)
            cmd_reencode = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart",
                str(out_file),
            ]
            result2 = subprocess.run(
                cmd_reencode, capture_output=True, text=True, timeout=900,
            )
            if result2.returncode != 0:
                return {"success": False, "error": result2.stderr or result2.stdout or "FFmpeg failed"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "FFmpeg timeout"}
    except FileNotFoundError:
        return {"success": False, "error": "FFmpeg not installed. Install with: brew install ffmpeg"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if list_file.exists():
            os.remove(list_file)

    try:
        shutil.rmtree(chunks_base)
    except Exception:
        pass

    return {"success": True, "merged": str(out_file)}
