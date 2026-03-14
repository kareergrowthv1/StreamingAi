import datetime
import os
import shutil
import subprocess
import time
from pathlib import Path

import config


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
    chunks_base = config.CHUNKS_DIR / client_id / position_id / candidate_id
    if not chunks_base.exists():
        return {"success": False, "error": "Chunks directory not found"}

    # Sort by name (timestamp is in name: part_123456.webm / camera_part_123456.webm)
    chunk_files = sorted(chunks_base.glob(f"{file_prefix}*.webm"), key=lambda p: p.name)
    if not chunk_files:
        return {"success": False, "error": f"No chunk files found for prefix '{file_prefix}'"}

    out_path = config.MERGED_DIR / client_id / position_id / candidate_id
    out_path.mkdir(parents=True, exist_ok=True)
    

    # 1. Create unique session file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = out_path / f"details_{timestamp}.mp4"
    
    list_file = chunks_base / "concat_list.txt"
    try:
        with open(list_file, "w") as f:
            for p in chunk_files:
                path_str = p.absolute().as_posix().replace("'", "'\\''")
                f.write(f"file '{path_str}'\n")
    except Exception as e:
        return {"success": False, "error": str(e)}

    # Merge chunks to session_file
    success = _ffmpeg_concat(list_file, session_file)
    if list_file.exists():
        os.remove(list_file)

    if not success:
        return {"success": False, "error": "Failed to merge chunks into session file"}

    # 2. Update/Create cumulative recording with timestamp
    # Find existing recording_*.mp4
    existing_recordings = sorted(out_path.glob(f"{output_prefix}_*.mp4"), key=lambda p: p.stat().st_mtime)
    
    new_recording_file = out_path / f"{output_prefix}_{timestamp}.mp4"
    
    if existing_recordings:
        # Merge most recent recording_*.mp4 + new session_file
        latest_recording = existing_recordings[-1]
        temp_merged = out_path / f"temp_merge_{timestamp}.mp4"
        
        merge_list = out_path / f"merge_list_{timestamp}.txt"
        with open(merge_list, "w") as f:
            rec_path = latest_recording.absolute().as_posix().replace("'", "'\\''")
            sess_path = session_file.absolute().as_posix().replace("'", "'\\''")
            f.write(f"file '{rec_path}'\n")
            f.write(f"file '{sess_path}'\n")
        
        merge_success = _ffmpeg_concat(merge_list, temp_merged)
        if merge_list.exists():
            os.remove(merge_list)
            
        if merge_success:
            # Move temp to final timestamped name
            os.replace(temp_merged, new_recording_file)
            # Remove previous recordings to keep only the latest merged one
            for old_rec in existing_recordings:
                try:
                    os.remove(old_rec)
                except Exception:
                    pass
        else:
            return {"success": False, "error": "Failed to merge session into cumulative recording"}
    else:
        # First recording, just move session_file to new timestamped name
        # We use shutil.copy2 then remove to be safe, but os.rename is better if on same disk
        try:
            shutil.copy2(session_file, new_recording_file)
        except Exception as e:
            return {"success": False, "error": f"Failed to save first recording: {e}"}
        
    # --- Cleanup ---
    # 1. Delete the current session file ALWAYS (it's either merged or copied)
    try:
        if session_file.exists():
            os.remove(session_file)
    except Exception:
        pass

    # 2. Delete any other details_*.mp4 files in the directory
    for detail_file in out_path.glob("details_*.mp4"):
        try:
            os.remove(detail_file)
        except Exception:
            pass

    # 3. Handle legacy recording.mp4 if it exists
    legacy_file = out_path / "recording.mp4"
    if legacy_file.exists():
        try:
            os.remove(legacy_file)
        except Exception:
            pass

    # 4. Cleanup chunks with this prefix only (leave other-prefix chunks intact)
    for chunk in chunk_files:
        try:
            chunk.unlink(missing_ok=True)
        except Exception:
            pass
    # Remove concat_list.txt if it remains
    leftover_list = chunks_base / "concat_list.txt"
    leftover_list.unlink(missing_ok=True)
    # Remove chunks dir only if now empty
    try:
        if chunks_base.exists() and not any(chunks_base.iterdir()):
            chunks_base.rmdir()
    except Exception:
        pass

    return {"success": True, "merged": str(new_recording_file), "session": str(session_file)}


def _ffmpeg_concat(list_file: Path, out_file: Path) -> bool:
    """Helper to run FFmpeg concat demuxer."""
    # Try -c copy first (fastest, no quality loss)
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
            
        # Fallback: re-encode
        cmd_reencode = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-movflags", "+faststart",
            str(out_file),
        ]
        result2 = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=900)
        return result2.returncode == 0
    except Exception:
        return False
