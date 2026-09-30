"""
Latecomer Attendance Dialog
============================
Teacher-controlled latecomer verification workflow:

    Live Camera Snapshot
        → Face Verification (existing ArcFace pipeline)
        → Attendance Check (already present today?)
        → Subject Enrollment Check
        → Duplicate Latecomer Check
        → Voice Verification (existing Resemblyzer pipeline)
        → Face ID == Voice ID match
        → Mark Present + Late (in attendance_logs + late_attendance)

Reuses:
  - src.pipline.face_pipeline  (get_face_embeddings, cosine_similarity, RECOGNITION_THRESHOLD)
  - src.pipline.voice_pipline  (get_voice_embedding, identify_speaker)
  - src.database.db            (check_student_present_today, check_existing_late_record,
                                insert_late_attendance, get_subject_students_with_embeddings)
"""

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime

from src.pipline.face_pipeline import (
    get_face_embeddings,
    cosine_similarity,
    _l2_normalise,
    is_old_dlib_embedding,
    RECOGNITION_THRESHOLD,
)
from src.pipline.voice_pipline import get_voice_embedding, identify_speaker
from src.database.db import (
    get_subject_students_with_embeddings,
    check_student_present_today,
    check_existing_late_record,
    insert_late_attendance,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_candidate_face_embeddings(enrolled_students: list) -> dict:
    """Return {student_id: normalised_np_embedding} from enrolled student list."""
    result = {}
    for s in enrolled_students:
        emb = s.get('face_embedding')
        if emb and not is_old_dlib_embedding(emb):
            result[int(s['student_id'])] = _l2_normalise(np.array(emb, dtype=np.float32))
    return result


def _build_candidate_voice_embeddings(enrolled_students: list) -> dict:
    """Return {student_id: voice_embedding_list} from enrolled student list."""
    return {
        int(s['student_id']): s['voice_embedding']
        for s in enrolled_students
        if s.get('voice_embedding')
    }


def _student_name(enrolled_students: list, student_id: int) -> str:
    for s in enrolled_students:
        if int(s['student_id']) == student_id:
            return s['name']
    return str(student_id)


# ---------------------------------------------------------------------------
# Face step
# ---------------------------------------------------------------------------

def _latecomer_face_step(subject_id: int):
    """
    Render the camera capture UI for face verification.

    Stores into session_state:
        latecomer_face_sid      — verified student_id (int) or None
        latecomer_face_score    — cosine similarity (float)
        latecomer_face_done     — True once face step is resolved
        latecomer_face_error    — error string or None
    """
    st.subheader("Step 1 — Face Verification")
    st.write("Ask the student to look directly at the camera, then take a snapshot.")

    cam_img = st.camera_input("📸 Live Camera — Face Snapshot", key="latecomer_cam")

    if cam_img is None:
        return  # waiting for snapshot

    if st.button("Verify Face", type="primary", width="stretch", key="latecomer_verify_face_btn"):
        with st.spinner("Running face recognition…"):
            img_np = np.array(Image.open(cam_img).convert("RGB"))
            face_data = get_face_embeddings(img_np)

            if not face_data:
                st.error("❌ No face detected. Please ensure the student is clearly visible and well-lit.")
                return

            if len(face_data) > 1:
                st.error("❌ Multiple faces detected. Only the latecomer student should be in frame.")
                return

            # Load enrolled students and their face embeddings
            enrolled_students = get_subject_students_with_embeddings(subject_id)
            candidate_face = _build_candidate_face_embeddings(enrolled_students)

            if not candidate_face:
                st.error("❌ No enrolled students have face profiles yet.")
                return

            query_emb = face_data[0]["embedding"]
            best_sid = None
            best_score = -1.0

            for sid, stored_emb in candidate_face.items():
                score = cosine_similarity(query_emb, stored_emb)
                if score > best_score:
                    best_score = score
                    best_sid = sid

            if best_score < RECOGNITION_THRESHOLD:
                st.error(
                    f"❌ Student could not be identified. "
                    f"(Best similarity: {best_score:.2f}, threshold: {RECOGNITION_THRESHOLD:.2f})"
                )
                return

            # Found a match — now run security checks
            student_name = _student_name(enrolled_students, best_sid)

            # 1. Check subject enrollment (already guaranteed since we only matched against enrolled students)
            # (get_subject_students_with_embeddings already filters by subject_id)

            # 2. Check if already present today
            if check_student_present_today(best_sid, subject_id):
                st.error(
                    f"⚠️ **{student_name}** is already marked **Present** today for this subject. "
                    "Latecomer attendance is not required."
                )
                return

            # 3. Check for duplicate late record
            if check_existing_late_record(best_sid, subject_id):
                st.error(
                    f"⚠️ **{student_name}** already has a latecomer record for today. "
                    "Duplicate late attendance is not allowed."
                )
                return

            # All checks passed — proceed to voice step
            st.session_state.latecomer_face_sid = best_sid
            st.session_state.latecomer_face_score = round(best_score, 4)
            st.session_state.latecomer_face_done = True
            st.session_state.latecomer_face_error = None
            st.session_state.latecomer_enrolled_students = enrolled_students
            st.rerun()


# ---------------------------------------------------------------------------
# Voice step
# ---------------------------------------------------------------------------

def _latecomer_voice_step(subject_id: int):
    """
    Render the audio capture UI for voice verification.

    Checks that the voice-identified student_id matches the face-identified student_id.

    Stores into session_state:
        latecomer_voice_sid     — voice-verified student_id (int) or None
        latecomer_voice_score   — voice similarity score (float)
        latecomer_voice_done    — True once voice step is resolved
    """
    face_sid = st.session_state.latecomer_face_sid
    enrolled_students = st.session_state.get("latecomer_enrolled_students", [])
    student_name = _student_name(enrolled_students, face_sid)

    st.success(f"✅ Face verified: **{student_name}** (similarity: {st.session_state.latecomer_face_score:.2f})")
    st.subheader("Step 2 — Voice Verification")
    st.write(
        f"Ask **{student_name}** to say: *\"I am present\"* and record their voice below."
    )

    audio_data = st.audio_input("🎤 Record Student Voice", key="latecomer_audio")

    if audio_data is None:
        return

    if st.button("Verify Voice", type="primary", width="stretch", key="latecomer_verify_voice_btn"):
        with st.spinner("Running voice verification…"):
            audio_bytes = audio_data.read()
            voice_emb = get_voice_embedding(audio_bytes)

            if voice_emb is None:
                st.error("❌ Could not process voice recording. Please try again.")
                return

            # Build candidate voice embeddings (enrolled students only)
            candidate_voice = _build_candidate_voice_embeddings(enrolled_students)

            if not candidate_voice:
                st.error("❌ No enrolled students have voice profiles yet.")
                return

            voice_sid, voice_score = identify_speaker(voice_emb, candidate_voice, threshold=0.65)

            if voice_sid is None:
                st.error(
                    f"❌ Voice verification failed. "
                    f"The voice could not be matched to any enrolled student "
                    f"(best score: {voice_score:.2f})."
                )
                return

            # CRITICAL: face ID must match voice ID
            if int(voice_sid) != int(face_sid):
                mismatch_name = _student_name(enrolled_students, voice_sid)
                st.error(
                    f"❌ Identity mismatch! "
                    f"Face identified **{student_name}** but voice matched **{mismatch_name}**. "
                    "Latecomer attendance rejected."
                )
                return

            # Both match — save record
            st.session_state.latecomer_voice_sid = int(voice_sid)
            st.session_state.latecomer_voice_score = round(float(voice_score), 4)
            st.session_state.latecomer_voice_done = True
            st.rerun()


# ---------------------------------------------------------------------------
# Save step
# ---------------------------------------------------------------------------

def _latecomer_save_step(subject_id: int):
    """Confirm and save the verified latecomer record."""
    enrolled_students = st.session_state.get("latecomer_enrolled_students", [])
    face_sid = st.session_state.latecomer_face_sid
    face_score = st.session_state.latecomer_face_score
    voice_score = st.session_state.latecomer_voice_score
    student_name = _student_name(enrolled_students, face_sid)
    arrival_time = datetime.now().strftime("%I:%M %p")

    st.success(
        f"✅ Face verified: **{student_name}** (score: {face_score:.2f})\n\n"
        f"✅ Voice verified: **{student_name}** (score: {voice_score:.2f})"
    )
    st.subheader("Step 3 — Confirm Late Attendance")

    st.markdown(
        f"""
| Field | Value |
|---|---|
| **Student** | {student_name} |
| **Status** | 🟡 Present + Late |
| **Arrival Time** | {arrival_time} |
| **Face Score** | {face_score:.2f} |
| **Voice Score** | {voice_score:.2f} |
"""
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Discard", width="stretch", type="secondary", key="latecomer_discard_btn"):
            _reset_latecomer_state()
            st.rerun()

    with col2:
        if st.button("✅ Confirm Late Attendance", width="stretch", type="primary", key="latecomer_confirm_btn"):
            try:
                insert_late_attendance(
                    student_id=face_sid,
                    subject_id=subject_id,
                    face_score=face_score,
                    voice_score=voice_score,
                    liveness_passed=False,
                )
                st.toast(f"✅ {student_name} marked Present + Late!", icon="🟡")
                _reset_latecomer_state()
                st.session_state.latecomer_last_success = student_name
                st.rerun()
            except Exception as e:
                st.error(
                    f"❌ Database save failed: {e}\n\n"
                    "Attendance has NOT been marked. Please try again."
                )


# ---------------------------------------------------------------------------
# State reset helper
# ---------------------------------------------------------------------------

def _reset_latecomer_state():
    """Clear all latecomer verification session state (does not affect any other state)."""
    for key in [
        'latecomer_face_sid', 'latecomer_face_score', 'latecomer_face_done',
        'latecomer_face_error', 'latecomer_voice_sid', 'latecomer_voice_score',
        'latecomer_voice_done', 'latecomer_enrolled_students',
    ]:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Public entry point — inline section (not a @st.dialog to keep camera working)
# ---------------------------------------------------------------------------

def latecomer_attendance_section(subject_id: int, subject_label: str):
    """
    Render the inline Latecomer Attendance section.

    This is rendered directly inside the Take Attendance tab when the teacher
    has enabled Latecomer Mode (st.session_state.latecomer_mode_active == True).

    Parameters
    ----------
    subject_id : int
        The currently selected subject.
    subject_label : str
        Display name for UI feedback.
    """
    # ── Banner ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#5865F2;border-radius:1rem;padding:1rem 1.5rem;margin-bottom:1rem;">
            <h3 style="color:white;margin:0;font-family:'Climate Crisis',sans-serif;">
                🟡 Latecomer Mode ACTIVE
            </h3>
            <p style="color:#E0E3FF;margin:0.4rem 0 0;">
                Face and voice verification are required.
                Only the teacher should operate this screen.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"Subject: **{subject_label}**")

    # ── Success toast from previous successful verification ────────────────
    if st.session_state.get('latecomer_last_success'):
        name = st.session_state.latecomer_last_success
        st.success(
            f"✅ **{name}** has been successfully marked **Present + Late**. "
            "You may verify the next latecomer or stop Latecomer Mode."
        )
        st.session_state.latecomer_last_success = None

    st.divider()

    # ── State machine ──────────────────────────────────────────────────────
    face_done  = st.session_state.get('latecomer_face_done', False)
    voice_done = st.session_state.get('latecomer_voice_done', False)

    if not face_done:
        _latecomer_face_step(subject_id)

    elif face_done and not voice_done:
        _latecomer_voice_step(subject_id)

    else:
        _latecomer_save_step(subject_id)

    # ── Stop Latecomer Mode button ─────────────────────────────────────────
    st.divider()
    if st.button(
        "🛑 Stop Latecomer Mode",
        type="secondary",
        width="stretch",
        key="stop_latecomer_btn",
    ):
        st.session_state.latecomer_mode_active = False
        _reset_latecomer_state()
        st.rerun()
