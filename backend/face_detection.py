"""
Proctoring: exact face detection and head pose.
Events: no_face | multiple_faces | looking_left | looking_right | looking_up | looking_down | ok.
Uses OpenCV Haar cascade for face count; MediaPipe Face Mesh for head pose (yaw/pitch).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Exact event names returned to frontend
EVENT_NO_FACE = "no_face"
EVENT_MULTIPLE_FACES = "multiple_faces"
EVENT_LOOKING_LEFT = "looking_left"
EVENT_LOOKING_RIGHT = "looking_right"
EVENT_LOOKING_UP = "looking_up"
EVENT_LOOKING_DOWN = "looking_down"
EVENT_OK = "ok"

# Head pose thresholds (normalized). Tune for stricter/looser detection.
YAW_THRESHOLD = 0.12   # Nose left/right of eye-center: beyond this = looking_left / looking_right
PITCH_THRESHOLD = 0.15 # Nose up/down of mid face: beyond this = looking_up / looking_down
CONFIDENCE_CAP = 0.35  # Max offset used to cap confidence (so confidence in 0..1)

_cv2 = None
_mp_face_mesh = None
_haar_cascade = None


def _load_deps() -> bool:
    global _cv2, _mp_face_mesh, _haar_cascade
    if _cv2 is not None:
        return True
    try:
        import cv2 as cv
        _cv2 = cv
        _haar_cascade = cv.CascadeClassifier(
            cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    except ImportError:
        logger.warning(
            "opencv-python not installed; face detection disabled. pip install opencv-python-headless"
        )
        return False
    try:
        import mediapipe as mp
        _mp_face_mesh = mp.solutions.face_mesh
    except ImportError:
        logger.warning("mediapipe not installed; head pose disabled. pip install mediapipe")
    return True


def _count_faces(image_bgr) -> int:
    """Count faces using OpenCV Haar cascade. Returns number of faces (0, 1, or more)."""
    if _cv2 is None or _haar_cascade is None:
        return 0
    try:
        gray = _cv2.cvtColor(image_bgr, _cv2.COLOR_BGR2GRAY)
        faces = _haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return len(faces) if faces is not None else 0
    except Exception as e:
        logger.debug("Face count failed: %s", e)
        return 0


def _head_pose_mediapipe(image_rgb) -> tuple[str, float]:
    """
    Infer head pose from MediaPipe Face Mesh.
    Returns (event, confidence).
    event: ok | looking_left | looking_right | looking_up | looking_down
    - looking_left:  head turned left  (nose moves right in image -> nose_x > center)
    - looking_right: head turned right (nose moves left in image -> nose_x < center)
    - looking_up:    head tilted up    (nose moves up in image -> nose_y < center)
    - looking_down:  head tilted down  (nose moves down in image -> nose_y > center)
    """
    if _mp_face_mesh is None:
        return EVENT_OK, 0.0
    try:
        with _mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
            if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
                return EVENT_OK, 0.0
            landmarks = results.multi_face_landmarks[0]
            h, w = image_rgb.shape[:2]
            # Landmarks: 4=nose tip, 33=left eye inner, 263=right eye inner
            nose_x = landmarks.landmark[4].x * w
            nose_y = landmarks.landmark[4].y * h
            left_x = landmarks.landmark[33].x * w
            right_x = landmarks.landmark[263].x * w
            center_x = (left_x + right_x) / 2
            center_y = h * 0.5
            # Normalized offsets in [-1, 1] range
            yaw_norm = (nose_x - center_x) / max(w * 0.5, 1)
            pitch_norm = (nose_y - center_y) / max(h * 0.5, 1)
            # Yaw: positive = nose right of center = head turned left = looking_left
            if yaw_norm > YAW_THRESHOLD:
                conf = min(1.0, (yaw_norm - YAW_THRESHOLD) / CONFIDENCE_CAP)
                return EVENT_LOOKING_LEFT, round(conf, 2)
            if yaw_norm < -YAW_THRESHOLD:
                conf = min(1.0, (abs(yaw_norm) - YAW_THRESHOLD) / CONFIDENCE_CAP)
                return EVENT_LOOKING_RIGHT, round(conf, 2)
            # Pitch: positive = nose below center = head tilted down = looking_down
            if pitch_norm > PITCH_THRESHOLD:
                conf = min(1.0, (pitch_norm - PITCH_THRESHOLD) / CONFIDENCE_CAP)
                return EVENT_LOOKING_DOWN, round(conf, 2)
            if pitch_norm < -PITCH_THRESHOLD:
                conf = min(1.0, (abs(pitch_norm) - PITCH_THRESHOLD) / CONFIDENCE_CAP)
                return EVENT_LOOKING_UP, round(conf, 2)
            return EVENT_OK, 0.9
    except Exception as e:
        logger.debug("MediaPipe head pose failed: %s", e)
        return EVENT_OK, 0.0


def analyze(image_bytes: bytes) -> dict[str, Any]:
    """
    Run proctoring analysis on one frame (JPEG/PNG bytes).
    Returns {"event": str, "confidence": float}.
    Exact events: no_face | multiple_faces | looking_left | looking_right | looking_up | looking_down | ok
    """
    if not image_bytes or len(image_bytes) < 100:
        return {"event": EVENT_NO_FACE, "confidence": 1.0}
    if not _load_deps():
        return {"event": EVENT_OK, "confidence": 0.0}
    try:
        import numpy as np
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if img is None:
            return {"event": EVENT_NO_FACE, "confidence": 1.0}
        num_faces = _count_faces(img)
        if num_faces == 0:
            return {"event": EVENT_NO_FACE, "confidence": 1.0}
        if num_faces > 1:
            return {"event": EVENT_MULTIPLE_FACES, "confidence": 1.0}
        img_rgb = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        pose_event, pose_conf = _head_pose_mediapipe(img_rgb)
        if pose_event != EVENT_OK:
            return {"event": pose_event, "confidence": pose_conf}
        return {"event": EVENT_OK, "confidence": 0.9}
    except Exception as e:
        logger.exception("face_detection.analyze failed: %s", e)
        return {"event": EVENT_OK, "confidence": 0.0}
