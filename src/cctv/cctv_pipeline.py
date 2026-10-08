"""
FusionPresence — CCTV Processing Pipeline
==========================================
Handles camera I/O, per-frame face processing, and temporal multi-frame
confirmation — all without touching any existing pipeline.

Architecture
------------
Camera (USB/RTSP)
    └─► capture_cctv_frame()
            └─► process_cctv_frame()
                    └─► get_face_embeddings()   ← existing face_pipeline
                            └─► cosine_similarity vs. enrolled candidates
                                    └─► update_confirmation_tracker()
                                            └─► confirmed? → mark attendance

Temporal Confirmation
---------------------
A student must be consistently recognised across at least
CCTV_CONFIRMATION_FRAMES frames within a CCTV_CONFIRMATION_WINDOW-second
sliding window before attendance is recorded.  This prevents single-frame
false positives.

NOTE: This module never writes to the database.
      All DB writes are handled in cctv_attendance.py.

IMPORTANT: Does NOT modify RECOGNITION_THRESHOLD or any existing pipeline.
           Uses its own CCTV_FACE_THRESHOLD exclusively.
"""

import time
import cv2
import numpy as np

# Reuse the existing ArcFace pipeline — read-only, no modifications
from src.pipline.face_pipeline import (
    get_face_embeddings,
    cosine_similarity,
    _l2_normalise,
    is_old_dlib_embedding,
)


# ===========================================================================
# Camera Management
# ===========================================================================

def get_cctv_camera(source):
    """
    Open a cv2.VideoCapture for *source*.

    Parameters
    ----------
    source : int | str
        0 (or any int) for a USB webcam index.
        An RTSP URL string for an IP camera.

    Returns
    -------
    cv2.VideoCapture | None
        Opened capture object, or None if the connection failed.
    """
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None
        # Minimise RTSP buffer size to reduce stream latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    except Exception:
        return None


def release_cctv_camera(cap) -> None:
    """
    Safely release a cv2.VideoCapture object.
    Never raises — safe to call even if cap is None or already released.
    """
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def capture_cctv_frame(cap) -> tuple:
    """
    Read one frame from an open VideoCapture.

    Returns
    -------
    tuple (success: bool, frame_rgb: np.ndarray | None)
        frame_rgb is in RGB format (H×W×3, uint8), ready for InsightFace.
        Returns (False, None) on any error.
    """
    if cap is None or not cap.isOpened():
        return False, None
    try:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            return False, None
        # OpenCV reads in BGR — convert to RGB for the InsightFace pipeline
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return True, frame_rgb
    except Exception:
        return False, None


# ===========================================================================
# Per-Frame Recognition
# ===========================================================================

def process_cctv_frame(
    frame_rgb: np.ndarray,
    candidate_embeddings: dict,
    threshold: float,
) -> list:
    """
    Detect and recognise faces in a single CCTV frame.

    Uses the existing InsightFace model (already cached by @st.cache_resource).
    Does NOT use the global RECOGNITION_THRESHOLD — uses *threshold* (CCTV-specific).

    Parameters
    ----------
    frame_rgb : np.ndarray
        RGB frame from the camera (H×W×3, uint8).
    candidate_embeddings : dict
        { student_id (int): face_embedding_list } — subject-enrolled students only.
    threshold : float
        CCTV-specific cosine similarity threshold.

    Returns
    -------
    list of dict
        One entry per *recognised* face (above threshold):
        { "student_id": int, "similarity": float }
        Faces below threshold are silently ignored (unknown persons).
    """
    face_data = get_face_embeddings(frame_rgb)
    if not face_data:
        return []

    # Pre-process candidates: normalise + skip old 128-D Dlib embeddings
    valid_candidates: dict = {}
    for sid, emb_list in candidate_embeddings.items():
        if not emb_list:
            continue
        if is_old_dlib_embedding(emb_list):
            continue  # Student needs re-enrollment; skip silently
        valid_candidates[int(sid)] = _l2_normalise(
            np.array(emb_list, dtype=np.float32)
        )

    if not valid_candidates:
        return []

    recognized: list = []

    for face in face_data:
        query_emb = face["embedding"]
        best_sid   = None
        best_score = -1.0

        for sid, stored_emb in valid_candidates.items():
            score = cosine_similarity(query_emb, stored_emb)
            if score > best_score:
                best_score = score
                best_sid   = sid

        if best_score >= threshold and best_sid is not None:
            recognized.append({
                "student_id": best_sid,
                "similarity": round(best_score, 4),
            })
        # else: unknown face — do nothing (could be a visitor, teacher, etc.)

    return recognized


# ===========================================================================
# Temporal Multi-Frame Confirmation
# ===========================================================================

def update_confirmation_tracker(
    student_id: int,
    frame_tracker: dict,
    confirmation_frames: int,
    confirmation_window: float,
) -> bool:
    """
    Update the sliding-window recognition counter for *student_id*.

    A student is *confirmed* when they have been recognised in at least
    ``confirmation_frames`` frames within the last ``confirmation_window``
    seconds.  This prevents single-frame false positives from triggering
    attendance records.

    Parameters
    ----------
    student_id : int
    frame_tracker : dict
        Mutable dict { student_id: [unix_timestamp, ...] }.
        Stored in st.session_state.cctv_frame_tracker and passed by reference.
    confirmation_frames : int
        Minimum number of recognitions required (CCTV_CONFIRMATION_FRAMES).
    confirmation_window : float
        Sliding window duration in seconds (CCTV_CONFIRMATION_WINDOW).

    Returns
    -------
    bool
        True  → student has now accumulated enough confirmed frames.
        False → not yet confirmed; keep collecting frames.
    """
    now    = time.time()
    cutoff = now - confirmation_window

    # Retrieve existing timestamps (or start fresh)
    timestamps: list = frame_tracker.get(student_id, [])

    # Record this recognition
    timestamps.append(now)

    # Evict timestamps outside the sliding window
    timestamps = [t for t in timestamps if t >= cutoff]

    # Write back
    frame_tracker[student_id] = timestamps

    return len(timestamps) >= confirmation_frames


# ===========================================================================
# Utility
# ===========================================================================

def build_candidate_embeddings(enrolled_students: list) -> dict:
    """
    Build the candidate embedding dict expected by process_cctv_frame().

    Parameters
    ----------
    enrolled_students : list
        Output of get_subject_students_with_embeddings(subject_id).
        Each entry: { student_id, name, face_embedding, voice_embedding }.

    Returns
    -------
    dict { student_id (int): face_embedding_list }
        Only students who have a face_embedding are included.
        Students without a face profile are silently skipped.
    """
    candidates: dict = {}
    for s in enrolled_students:
        emb = s.get("face_embedding")
        if emb:
            candidates[int(s["student_id"])] = emb
    return candidates
