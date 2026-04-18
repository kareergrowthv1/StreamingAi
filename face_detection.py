"""
Proctoring Service using MediaPipe FaceLandmarker.
Based on reference logic from Test/ats_new_ai.
Detects: no_face, multiple_faces, looking_left, looking_right, looking_up, looking_down, ok.
"""
from __future__ import annotations

import base64
import logging
import threading
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

# Thresholds for head movement
YAW_THRESHOLD = 0.20
PITCH_THRESHOLD = 0.20

# Face landmark indices
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_EDGE = 234
RIGHT_EDGE = 454

class ProctoringService:
    _landmarker = None
    _lock = threading.Lock()

    @classmethod
    def get_landmarker(cls):
        """Initialize and return the Mediapipe FaceLandmarker singleton."""
        if cls._landmarker is None:
            # Try to get MODEL_PATH from config, fallback to local path
            model_path = getattr(config, "MODEL_PATH", Path(__file__).parent / "models" / "face_landmarker.task")
            if not isinstance(model_path, Path):
                model_path = Path(model_path)
                
            if not model_path.exists():
                logger.error(f"Mediapipe model not found at {model_path}")
                return None

            try:
                base_options = python.BaseOptions(model_asset_path=str(model_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=5  # detect up to 5 faces so we can flag multiple_faces violation
                )
                cls._landmarker = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                logger.error(f"Failed to initialize FaceLandmarker: {e}")
                return None
        return cls._landmarker

    @classmethod
    def process_frame(cls, frame_b64: str) -> dict:
        """Decode base64 frame and analyze for proctoring violations."""
        try:
            landmarker = cls.get_landmarker()
            if not landmarker:
                return {"event": "ok", "confidence": 0.0}

            # Decode base64 frame
            header, encoded = frame_b64.split(",", 1) if "," in frame_b64 else ("", frame_b64)
            data = base64.b64decode(encoded)
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {"event": "no_face", "confidence": 0.0}

            # Mediapipe detection
            with cls._lock:
                # Convert BGR (OpenCV) to RGB (MediaPipe)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = landmarker.detect(mp_image)

            if not results or not results.face_landmarks:
                return {"event": "no_face", "confidence": 1.0}

            if len(results.face_landmarks) > 1:
                return {"event": "multiple_faces", "confidence": 0.9}

            # Pose estimation using landmarks
            landmarks = results.face_landmarks[0]
            nose = landmarks[NOSE_TIP]
            forehead = landmarks[FOREHEAD]
            chin = landmarks[CHIN]
            left_edge = landmarks[LEFT_EDGE]
            right_edge = landmarks[RIGHT_EDGE]

            # YAW (Left/Right)
            face_width = abs(right_edge.x - left_edge.x)
            horizontal_center = (left_edge.x + right_edge.x) / 2
            yaw = (nose.x - horizontal_center) / face_width if face_width > 0 else 0

            # PITCH (Up/Down)
            face_height = abs(chin.y - forehead.y)
            vertical_center = (forehead.y + chin.y) / 2
            pitch = (nose.y - vertical_center) / face_height if face_height > 0 else 0

            if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
                if yaw > YAW_THRESHOLD: event = "looking_right"
                elif yaw < -YAW_THRESHOLD: event = "looking_left"
                elif pitch > PITCH_THRESHOLD: event = "looking_down"
                elif pitch < -PITCH_THRESHOLD: event = "looking_up"
                else: event = "head_turned"
                return {"event": event, "confidence": 0.9, "yaw": yaw, "pitch": pitch}

            return {"event": "ok", "confidence": 1.0}
        except Exception as e:
            logger.error(f"Proctoring error: {e}")
            return {"event": "ok", "confidence": 0.0}

def analyze(frame_b64: str) -> dict:
    """Legacy wrapper for main.py"""
    return ProctoringService.process_frame(frame_b64)
