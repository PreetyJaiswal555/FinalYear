"""
FusionPresence — Face Recognition Pipeline
==========================================
Modern ArcFace / InsightFace-based face recognition.

Architecture:
    Image
      └─► InsightFace (RetinaFace detection + face alignment)
            └─► ArcFace 512-D embedding (L2-normalised)
                  └─► Cosine similarity vs. stored embeddings
                        └─► Threshold → student_id + similarity score

Author: FusionPresence Team
"""

import numpy as np
import streamlit as st
import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Cosine-similarity threshold for a face to be considered "recognised".
# Values range from -1 (opposite) to 1 (identical).
# 0.65 is a reasonable starting point; tune upward for stricter matching.
RECOGNITION_THRESHOLD: float = 0.65

# InsightFace model pack.  "buffalo_sc" is lightweight (MobileNet backbone).
# Switch to "buffalo_l" for higher accuracy at the cost of speed.
INSIGHTFACE_MODEL_NAME: str = "buffalo_sc"


# ---------------------------------------------------------------------------
# Model Loading  (cached — loaded once per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading face recognition model…")
def load_insightface_model():
    """
    Load and cache the InsightFace FaceAnalysis application.

    Returns
    -------
    insightface.app.FaceAnalysis
        Ready-to-use analysis app with detection + recognition modules.
    """
    import os, ssl

    # ── Workaround: InsightFace downloads models via HTTPS from GitHub.
    # On some networks the TLS handshake fails with UNEXPECTED_EOF_WHILE_READING.
    # Monkey-patch ssl to create an unverified context so the download succeeds.
    _orig_create_default_https_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context

    try:
        import insightface
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=INSIGHTFACE_MODEL_NAME,
            providers=["CPUExecutionProvider"],   # change to CUDAExecutionProvider if GPU available
        )
        # ctx_id=-1 → CPU inference; det_size controls detection resolution
        app.prepare(ctx_id=-1, det_size=(640, 640))
        return app
    except Exception as exc:
        err_msg = str(exc)
        if "SSLError" in err_msg or "HTTPS" in err_msg or "EOF" in err_msg:
            st.error(
                "❌ **Face model download failed (SSL/Network error)**\n\n"
                "InsightFace could not download the `buffalo_sc` model pack from GitHub "
                "due to a network/TLS issue.\n\n"
                "**How to fix:**\n"
                "1. Check your internet connection and try again.\n"
                "2. If behind a corporate proxy/firewall, configure it to allow GitHub release assets.\n"
                "3. As a last resort, manually download `buffalo_sc.zip` from "
                "[InsightFace releases](https://github.com/deepinsight/insightface/releases) "
                "and place it in `~/.insightface/models/buffalo_sc/`.\n\n"
                f"Raw error: `{exc}`"
            )
        else:
            st.error(
                f"❌ Failed to load face recognition model: {exc}\n\n"
                "Please ensure `insightface` and `onnxruntime` are installed."
            )
        return None
    finally:
        # Restore the original SSL context
        ssl._create_default_https_context = _orig_create_default_https_context



# ---------------------------------------------------------------------------
# Embedding Utilities
# ---------------------------------------------------------------------------

def _to_bgr(image_rgb_np: np.ndarray) -> np.ndarray:
    """Convert an RGB numpy array to BGR (required by InsightFace / OpenCV)."""
    return cv2.cvtColor(image_rgb_np, cv2.COLOR_RGB2BGR)


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    """Return the L2-normalised version of a vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalised embeddings.

    Because both vectors are unit-normalised, this is simply the dot product,
    yielding a value in [-1, 1].  Higher → more similar.
    """
    return float(np.dot(emb_a, emb_b))


def is_old_dlib_embedding(embedding: list) -> bool:
    """
    Return True if the stored embedding is a legacy Dlib 128-D vector.
    ArcFace embeddings produced by this pipeline are 512-D.
    """
    return isinstance(embedding, list) and len(embedding) == 128


# ---------------------------------------------------------------------------
# Face Embedding Extraction
# ---------------------------------------------------------------------------

def get_face_embeddings(image_rgb_np: np.ndarray) -> list:
    """
    Detect all faces in *image_rgb_np* and return their ArcFace embeddings.

    Parameters
    ----------
    image_rgb_np : np.ndarray
        Input image in RGB format (H × W × 3, uint8).

    Returns
    -------
    list of dict
        Each element:
        {
            "embedding": np.ndarray (512-D, L2-normalised),
            "bbox":      (x1, y1, x2, y2)   ← face bounding box in pixels
        }
        Returns an empty list when no faces are detected.
    """
    app = load_insightface_model()
    if app is None:
        return []

    image_bgr = _to_bgr(image_rgb_np)
    faces = app.get(image_bgr)

    results = []
    for face in faces:
        embedding = _l2_normalise(np.array(face.normed_embedding, dtype=np.float32))
        bbox = tuple(int(v) for v in face.bbox)   # (x1, y1, x2, y2)
        results.append({"embedding": embedding, "bbox": bbox})

    return results


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def recognize_faces(
    image_rgb_np: np.ndarray,
    candidate_embeddings: dict,
    threshold: float = RECOGNITION_THRESHOLD,
) -> list:
    """
    Detect all faces in the image and match each against *candidate_embeddings*.

    Parameters
    ----------
    image_rgb_np : np.ndarray
        RGB image (H × W × 3).
    candidate_embeddings : dict
        Mapping of  ``student_id  →  embedding_list``  for candidate students.
        ``embedding_list`` should be a Python list (as stored in Supabase).
        Old 128-D Dlib embeddings are automatically skipped.
    threshold : float, optional
        Minimum cosine similarity to accept a recognition.  Default 0.65.

    Returns
    -------
    list of dict
        One entry per detected face:
        {
            "student_id":       int | None,
            "similarity":       float  (best cosine score found),
            "status":           "recognized" | "unknown",
            "bbox":             (x1, y1, x2, y2)
        }
    """
    face_data = get_face_embeddings(image_rgb_np)

    if not face_data:
        return []

    # Pre-process candidate embeddings (normalise + filter old Dlib vectors)
    valid_candidates = {}
    for sid, emb_list in candidate_embeddings.items():
        if not emb_list:
            continue
        if is_old_dlib_embedding(emb_list):
            # Old 128-D Dlib embedding — skip, student needs re-enrollment
            continue
        valid_candidates[sid] = _l2_normalise(np.array(emb_list, dtype=np.float32))

    results = []

    for face in face_data:
        query_emb = face["embedding"]
        best_sid = None
        best_score = -1.0

        for sid, stored_emb in valid_candidates.items():
            score = cosine_similarity(query_emb, stored_emb)
            if score > best_score:
                best_score = score
                best_sid = sid

        if best_score >= threshold:
            results.append({
                "student_id": best_sid,
                "similarity": round(best_score, 4),
                "status": "recognized",
                "bbox": face["bbox"],
            })
        else:
            results.append({
                "student_id": None,
                "similarity": round(best_score, 4) if best_score >= 0 else 0.0,
                "status": "unknown",
                "bbox": face["bbox"],
            })

    return results


# ---------------------------------------------------------------------------
# Enrollment Helper
# ---------------------------------------------------------------------------

def enroll_face_embedding(image_rgb_np: np.ndarray) -> tuple:
    """
    Extract an ArcFace embedding suitable for storing as a student's face profile.

    Enforces exactly one face in the image.

    Parameters
    ----------
    image_rgb_np : np.ndarray
        RGB image of the student's face.

    Returns
    -------
    tuple  (embedding_list | None, error_message | None)
        On success : (list[float], None)   ← 512-D list, ready for Supabase
        On failure : (None, str)           ← human-readable error message
    """
    face_data = get_face_embeddings(image_rgb_np)

    if len(face_data) == 0:
        return None, "No face detected. Please ensure your face is clearly visible and well-lit."

    if len(face_data) > 1:
        return None, (
            f"{len(face_data)} faces detected in the image. "
            "Please ensure only your face is visible during enrollment."
        )

    embedding = face_data[0]["embedding"]
    return embedding.tolist(), None   # Convert numpy → Python list for JSON storage


# ---------------------------------------------------------------------------
# Compatibility shim (used in student login — global match across all students)
# ---------------------------------------------------------------------------

def predict_student_login(
    image_rgb_np: np.ndarray,
    all_students: list,
    threshold: float = RECOGNITION_THRESHOLD,
) -> tuple:
    """
    Match a single-face login image against ALL students in the database.

    Used exclusively for the student self-service login (not subject-specific).

    Parameters
    ----------
    image_rgb_np : np.ndarray
        RGB camera capture from the student login page.
    all_students : list
        List of student dicts from ``get_all_students()``.
        Each dict must contain ``student_id`` and ``face_embedding``.
    threshold : float, optional
        Cosine similarity threshold.  Default: RECOGNITION_THRESHOLD.

    Returns
    -------
    tuple  (num_faces_detected, recognition_result | None)
        num_faces_detected : int
        recognition_result : dict | None
            On match: {"student_id": int, "similarity": float}
            None when unrecognised or multiple/zero faces.
    """
    face_data = get_face_embeddings(image_rgb_np)
    num_faces = len(face_data)

    if num_faces != 1:
        return num_faces, None

    query_emb = face_data[0]["embedding"]
    best_sid = None
    best_score = -1.0

    for student in all_students:
        emb_list = student.get("face_embedding")
        if not emb_list or is_old_dlib_embedding(emb_list):
            continue
        stored_emb = _l2_normalise(np.array(emb_list, dtype=np.float32))
        score = cosine_similarity(query_emb, stored_emb)
        if score > best_score:
            best_score = score
            best_sid = student["student_id"]

    if best_score >= threshold:
        return num_faces, {"student_id": best_sid, "similarity": round(best_score, 4)}

    return num_faces, None
