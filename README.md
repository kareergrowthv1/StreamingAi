# Streaming — Proctoring & Screen Recording

Self-contained module for **browser-based proctoring** and **screen recording**: camera screenshots, 10-second video chunks, AI proctoring (face / head pose) in the browser, and server-side merge into a single playable video.

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Backend setup](#backend-setup)
- [Frontend setup](#frontend-setup)
- [Running the app](#running-the-app)
- [Optional: Celery + Redis](#optional-celery--redis)
- [Environment variables](#environment-variables)
- [API overview](#api-overview)
- [Project structure](#project-structure)

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| **Python 3.10+** | Backend (FastAPI) |
| **Node.js 18+** | Frontend (React + Vite) |
| **FFmpeg** | Merging video chunks into one MP4 |
| **Redis** (optional) | Message broker for Celery; backend can run without it |

### Install FFmpeg

FFmpeg is **required** for merging recorded chunks. It is not a Python package; install it on your system:

| OS | Command |
|----|--------|
| **macOS** | `brew install ffmpeg` |
| **Ubuntu / Debian** | `sudo apt update && sudo apt install ffmpeg` |
| **Windows** | Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH |

Verify:

```bash
ffmpeg -version
```

---

## Backend setup

All steps below assume you are in the **KareerGrowth** repo root (e.g. `~/Downloads/KareerGrowth`). Do not run `cd` from inside `Streaming/backend` when the path is given as `Streaming/backend`.

### 1. Go to the backend directory

```bash
cd Streaming/backend
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
```

Activate it:

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows (Cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

You should see `(.venv)` in your prompt.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, Pydantic, python-multipart, python-dotenv, and optional Celery/Redis/DB packages. See `requirements.txt` for the full list.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` if needed (see [Environment variables](#environment-variables)). Defaults are fine for local development:

- Server: `HOST=0.0.0.0`, `PORT=9000`
- Storage: `./uploads`, `./chunks`, `./merged` (created automatically)
- CORS: `http://localhost:5174`, `http://127.0.0.1:5174` (Vite dev server)

### 5. Run the backend

```bash
uvicorn main:app --host 0.0.0.0 --port 9000
```

Or use the port from your `.env`:

```bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-9000}
```

You should see something like:

```
INFO:     Uvicorn running on http://0.0.0.0:9000
INFO:     Application startup complete.
```

- **Health check:** [http://127.0.0.1:9000/health](http://127.0.0.1:9000/health) → `{"status":"ok"}`

**Note:** If you do not run Redis/Celery, the backend still works. When the user clicks **End Test**, the merge runs **in-process** (FFmpeg is still required).

---

## Frontend setup

From the **KareerGrowth** repo root:

### 1. Go to the frontend directory

```bash
cd Streaming/frontend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Run the dev server

```bash
npm run dev
```

The app will be at **http://localhost:5174**. Vite proxies `/api` and `/ws` to the backend at `http://127.0.0.1:9000`, so the backend must be running (see [Backend setup](#backend-setup)).

### Build for production

```bash
npm run build
npm run preview   # optional: preview the build
```

---

## Running the app

1. **Start the backend** (from `Streaming/backend` with venv active):

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 9000
   ```

2. **Start the frontend** (from `Streaming/frontend`):

   ```bash
   npm run dev
   ```

3. Open **http://localhost:5174**, enter **Client ID**, **Position ID**, **Candidate ID**, then:

   - **Capture Screenshot** — camera frame → PNG → uploaded to backend  
   - **Start Streaming** — screen recording in 10s chunks, each chunk playable  
   - **End Test** — stops recording, merges chunks with FFmpeg, shows **Play recording** link  

---

## Optional: Celery + Redis

If you install Redis and run a Celery worker, the merge job runs in the background (e.g. user can close the tab and the merge still completes).

### Install and start Redis

- **macOS:** `brew install redis && brew services start redis` (or run `redis-server` in a terminal)
- **Ubuntu:** `sudo apt install redis-server && sudo systemctl start redis`

### Run Celery worker

From `Streaming/backend` (with the same venv active):

```bash
celery -A tasks worker -l info
```

With Redis running, **End Test** will queue the merge task to Celery instead of running it in-process. If Redis/Celery is not available, the backend falls back to in-process merge.

---

## Environment variables

### Backend (`.env` in `Streaming/backend`)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address for Uvicorn |
| `PORT` | `9000` | Port for the API |
| `UPLOAD_DIR` | `./uploads` | Directory for screenshots |
| `CHUNKS_DIR` | `./chunks` | Directory for video chunks before merge |
| `MERGED_DIR` | `./merged` | Directory for final merged MP4 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL (for Celery) |
| `CORS_ORIGINS` | `http://localhost:5174,...` | Allowed frontend origins (comma-separated) |
| `DATABASE_URL` | *(empty)* | Optional PostgreSQL URL for proctoring events |

### Frontend

Optional; leave unset when using Vite dev proxy (default):

| Variable | Description |
|----------|-------------|
| `VITE_API_BASE` | Backend URL (e.g. `http://127.0.0.1:9000`) if not using proxy |
| `VITE_WS_BASE` | WebSocket URL (e.g. `ws://127.0.0.1:9000`) if not using proxy |

---

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/screenshot` | Upload camera screenshot (multipart: `client_id`, `position_id`, `candidate_id`, `file`) |
| `POST` | `/api/chunk` | Upload one video chunk (multipart: `client_id`, `position_id`, `candidate_id`, `index`, `file`) |
| `POST` | `/api/end-test` | End test and trigger merge (form: `client_id`, `position_id`, `candidate_id`) |
| `GET` | `/api/merged/{client_id}/{position_id}/{candidate_id}` | Stream the merged recording (MP4) |
| `WS` | `/ws/proctoring` | WebSocket for proctoring events (no_face, multiple_faces, head_turned) |
| `GET` | `/health` | Health check |

---

## Project structure

```
Streaming/
├── README.md                 # This file
├── .gitignore
├── backend/
│   ├── main.py               # FastAPI app, routes, WebSocket
│   ├── config.py             # Loads .env, paths, CORS
│   ├── merge_utils.py       # FFmpeg merge + delete chunks
│   ├── tasks.py              # Celery task (optional)
│   ├── requirements.txt      # Python dependencies + FFmpeg note
│   └── .env.example          # Example env; copy to .env
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js        # Dev server port 5174, proxy to backend
    └── src/
        ├── App.jsx           # UI: inputs, Capture / Start / End
        ├── config.js         # API_BASE, WS_BASE
        ├── main.jsx
        └── hooks/
            ├── useScreenshot.js    # getUserMedia + Canvas + upload
            ├── useScreenRecording.js # getDisplayMedia + MediaRecorder (10s chunks)
            └── useProctoring.js    # MediaPipe Face Landmarker + WebSocket
```

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **Port 9000 in use** | Use another port, e.g. `uvicorn main:app --port 9001`, and set the same in `Streaming/frontend/vite.config.js` proxy. |
| **"FFmpeg not installed"** | Install FFmpeg (see [Prerequisites](#prerequisites)) and ensure `ffmpeg` is on your PATH. |
| **"No module named 'celery'"** | Backend works without Celery; merge runs in-process. To use Celery: `pip install celery[redis]` and run Redis + worker. |
| **CORS errors** | Ensure `CORS_ORIGINS` in backend `.env` includes your frontend origin (e.g. `http://localhost:5174`). |
| **Merge fails / video not playable** | Ensure FFmpeg is installed and chunks are being uploaded (check `CHUNKS_DIR`). Backend tries copy then re-encode; check server logs for FFmpeg errors. |
# StreamingAi
