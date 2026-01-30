"""Configuration for Streaming backend."""
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
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

# Ensure dirs exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)
