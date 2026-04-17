"""
Merge webm chunk files into MP4.
- Absolute paths in concat list, -movflags +faststart, re-encode fallback.
- Cumulative recording (merge with existing recording_*.mp4 when present) for local chunks.
"""
import datetime
import logging
import os
import shutil
import subprocess
from pathlib import Path

try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

def do_merge_chunks(
    client_id: str,
    position_id: str,
    candidate_id: str,
    file_prefix: str = "part_",
    output_prefix: str = "recording",
) -> dict:
    """
    Merge webm chunk files into a single MP4.
    file_prefix   : glob prefix for chunk files (e.g. 'part_' or 'camera_part_')
    output_prefix : prefix for the final merged file (e.g. 'recording' or 'camera_recording')
    """
    chunks_base = config.local_chunks_dir(client_id, position_id, candidate_id)
    if not chunks_base.exists():
        return {"success": False, "error": f"Chunks directory not found: {chunks_base}"}

    chunk_files = sorted(chunks_base.glob(f"{file_prefix}*.webm"), key=lambda p: p.name)
    if not chunk_files:
        return {"success": False, "error": f"No chunk files found for prefix '{file_prefix}' in {chunks_base}"}

    out_path = config.local_merged_dir(client_id, position_id, candidate_id)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = out_path / f"details_{file_prefix}{timestamp}.mp4"

    list_file = chunks_base / f"concat_list_{file_prefix}{timestamp}.txt"
    try:
        with open(list_file, "w") as f:
            for p in chunk_files:
                path_str = p.absolute().as_posix().replace("'", "'\\''")
                f.write(f"file '{path_str}'\n")
    except Exception as e:
        return {"success": False, "error": f"Failed to create concat list: {str(e)}"}

    success = _ffmpeg_concat(list_file, session_file)
    if list_file.exists():
        try:
            os.remove(list_file)
        except Exception:
            pass

    if not success:
        return {"success": False, "error": "Failed to merge chunks into session file via FFmpeg"}

    # Cumulative recording logic
    existing_recordings = sorted(
        out_path.glob(f"{output_prefix}_*.mp4"), key=lambda p: p.stat().st_mtime
    )
    new_recording_file = out_path / f"{output_prefix}_{timestamp}.mp4"

    if existing_recordings:
        latest_recording = existing_recordings[-1]
        temp_merged = out_path / f"temp_merge_{output_prefix}_{timestamp}.mp4"
        merge_list = out_path / f"merge_list_{output_prefix}_{timestamp}.txt"
        try:
            with open(merge_list, "w") as f:
                rec_path = latest_recording.absolute().as_posix().replace("'", "'\\''")
                sess_path = session_file.absolute().as_posix().replace("'", "'\\''")
                f.write(f"file '{rec_path}'\n")
                f.write(f"file '{sess_path}'\n")
            
            merge_success = _ffmpeg_concat(merge_list, temp_merged)
            if merge_list.exists():
                os.remove(merge_list)
            
            if merge_success:
                os.replace(temp_merged, new_recording_file)
                # Remove old recordings
                for old_rec in existing_recordings:
                    try:
                        os.remove(old_rec)
                    except Exception:
                        pass
            else:
                return {"success": False, "error": "Failed to merge session into cumulative recording"}
        except Exception as e:
            return {"success": False, "error": f"Cumulative merge failed: {str(e)}"}
    else:
        try:
            shutil.copy2(session_file, new_recording_file)
        except Exception as e:
            return {"success": False, "error": f"Failed to save first recording: {e}"}

    # Final cleanup
    try:
        if session_file.exists():
            os.remove(session_file)
    except Exception:
        pass
    
    # Cleanup chunks
    for chunk in chunk_files:
        try:
            chunk.unlink(missing_ok=True)
        except Exception:
            pass
    
    # Cleanup dir if empty
    try:
        if chunks_base.exists() and not any(chunks_base.iterdir()):
            chunks_base.rmdir()
    except Exception:
        pass

    return {
        "success": True, 
        "merged": str(new_recording_file), 
        "session": str(session_file),
        "type": output_prefix
    }

def _ffmpeg_concat(list_file: Path, out_file: Path) -> bool:
    """Run FFmpeg concat demuxer. Try -c copy first, then re-encode on failure."""
    cmd_copy = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_file),
    ]
    try:
        result = subprocess.run(cmd_copy, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return True
        
        logger.warning("FFmpeg concat -c copy failed (stderr): %s", (result.stderr or "").strip())
        
        # Fallback: re-encode
        cmd_reencode = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-c:a", "aac", "-movflags", "+faststart",
            str(out_file),
        ]
        result2 = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=900)
        if result2.returncode != 0:
            logger.error("FFmpeg concat re-encode also failed: %s", (result2.stderr or "").strip())
        return result2.returncode == 0
    except Exception as e:
        logger.error("FFmpeg concat exception: %s", e)
        return False
