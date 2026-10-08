"""
FusionPresence — CCTV Attendance Section
==========================================
Full CCTV attendance workflow: UI + DB writes.

State Machine
-------------
    IDLE
      ↓  [▶ Start CCTV Attendance]
    RUNNING  (normal attendance window)
      │   Recognised student → temporal confirmation → mark Present
      ↓  [🔒 Close Normal Attendance]
    LATE WINDOW  (CCTV still running)
      │   Recognised student (not already present) → mark Present + Late
      ↓  [⏹ Stop CCTV]
    COMPLETED
      │   Enrolled students never recognised → mark Absent
      ↓
    IDLE

Key design decisions
--------------------
- Subject-filtered:  only enrolled students' embeddings are compared.
- Duplicate-safe:    check_student_present_today() prevents double-records.
- No liveness:       liveness_passed is explicitly set to False — we never
                     claim anti-spoofing capability that hasn't been implemented.
- Rerun-safe:        session is created once; no duplicate DB session on rerun.
- Camera-safe:       camera is released in all stop/error paths.

Reuses without modification
---------------------------
  src.pipline.face_pipeline     get_face_embeddings, existing cached model
  src.database.db               get_subject_students_with_embeddings,
                                check_student_present_today,
                                check_existing_late_record,
                                insert_late_attendance, create_attendance,
                                get_teacher_subjects
  src.database.config           supabase client
  src.cctv.cctv_config          CCTV settings
  src.cctv.cctv_pipeline        frame capture + temporal confirmation
"""

import streamlit as st
from datetime import datetime

from src.database.config import supabase
from src.database.db import (
    get_subject_students_with_embeddings,
    check_student_present_today,
    check_existing_late_record,
    insert_late_attendance,
    create_attendance,
    get_teacher_subjects,
)
from src.cctv.cctv_config import get_cctv_config, is_cctv_configured
from src.cctv.cctv_pipeline import (
    get_cctv_camera,
    release_cctv_camera,
    capture_cctv_frame,
    process_cctv_frame,
    update_confirmation_tracker,
    build_candidate_embeddings,
)


# ===========================================================================
# Database helpers — attendance_sessions table
# ===========================================================================

def start_cctv_session(subject_id: int, teacher_id: int) -> int | None:
    """
    Insert a new row into attendance_sessions and return its id.

    Returns None if the insert fails (error already shown to user).
    """
    try:
        ts   = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        resp = supabase.table("attendance_sessions").insert({
            "subject_id": subject_id,
            "teacher_id": teacher_id,
            "start_time": ts,
            "status":     "running",
        }).execute()
        if resp.data:
            return resp.data[0]["id"]
    except Exception as exc:
        st.error(f"❌ Could not create CCTV session record: {exc}")
    return None


def _update_session_normal_close(session_id: int) -> None:
    """Set normal_close_time and status = 'normal_closed'."""
    try:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        supabase.table("attendance_sessions").update({
            "normal_close_time": ts,
            "status":            "normal_closed",
        }).eq("id", session_id).execute()
    except Exception as exc:
        st.error(f"❌ DB update failed (normal close): {exc}")


def _finalize_session_db(session_id: int) -> None:
    """Set end_time and status = 'completed'."""
    try:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        supabase.table("attendance_sessions").update({
            "end_time": ts,
            "status":   "completed",
        }).eq("id", session_id).execute()
    except Exception as exc:
        st.error(f"❌ DB update failed (session finalize): {exc}")


# ===========================================================================
# Attendance write helpers — CCTV-specific
# ===========================================================================

def mark_cctv_present(student_id: int, subject_id: int) -> bool:
    """
    Mark student as Present in attendance_logs — CCTV path.

    Guards against duplicates using check_student_present_today().
    Returns True if a new record was written, False if already present.
    """
    if check_student_present_today(student_id, subject_id):
        return False   # Already present via any method — nothing to do

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    try:
        create_attendance([{
            "student_id": student_id,
            "subject_id": subject_id,
            "timestamp":  ts,
            "is_present": True,
        }])
        return True
    except Exception as exc:
        st.error(f"❌ Could not mark student present: {exc}")
        return False


def mark_cctv_late(student_id: int, subject_id: int, face_score: float) -> bool:
    """
    Mark student as Present + Late in attendance_logs and late_attendance.

    Reuses insert_late_attendance() from db.py which:
      1. Inserts into late_attendance (liveness_passed=False — no liveness detector)
      2. Inserts into attendance_logs  (is_present=True)

    Returns True if a new late record was written, False if already late.
    """
    if check_existing_late_record(student_id, subject_id):
        return False   # Duplicate late — skip

    try:
        insert_late_attendance(
            student_id=student_id,
            subject_id=subject_id,
            face_score=face_score,
            voice_score=None,        # No voice verification in CCTV mode
            liveness_passed=False,   # No liveness detection implemented
        )
        return True
    except Exception as exc:
        st.error(f"❌ Could not mark student late: {exc}")
        return False


def finalize_cctv_attendance(subject_id: int, confirmed_students: set) -> int:
    """
    After Stop CCTV — mark all enrolled-but-unrecognised students as Absent.

    Only marks Absent if the student is not already recorded as Present today
    (handles edge case where they were marked via another attendance method).

    Parameters
    ----------
    subject_id        : int
    confirmed_students: set of student_ids recognised during the CCTV session.

    Returns
    -------
    int  Number of students newly marked Absent.
    """
    enrolled   = get_subject_students_with_embeddings(subject_id)
    ts         = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    absent_logs: list = []

    for student in enrolled:
        sid = int(student["student_id"])
        if sid in confirmed_students:
            continue   # Was seen — skip
        # Not recognised — mark absent only if no existing present record
        if not check_student_present_today(sid, subject_id):
            absent_logs.append({
                "student_id": sid,
                "subject_id": subject_id,
                "timestamp":  ts,
                "is_present": False,
            })

    if absent_logs:
        try:
            create_attendance(absent_logs)
        except Exception as exc:
            st.error(f"❌ Could not record absent students: {exc}")

    return len(absent_logs)


# ===========================================================================
# Session State Management
# ===========================================================================

def _init_cctv_state() -> None:
    """Ensure all CCTV session-state keys exist with safe defaults."""
    defaults = {
        "cctv_running":              False,
        "cctv_session_id":           None,
        "cctv_start_time":           None,
        "cctv_normal_closed":        False,
        "cctv_normal_close_time":    None,
        "cctv_subject_id":           None,
        "cctv_confirmed_students":   set(),
        "cctv_late_students":        set(),
        "cctv_frame_tracker":        {},
        "cctv_camera":               None,
        "cctv_frame_counter":        0,
        "cctv_last_recognized":      "None",
        "cctv_last_frame":           None,
        "cctv_enrolled_students":    [],
        "cctv_candidate_embeddings": {},
        "cctv_error":                None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_cctv_state() -> None:
    """
    Release camera and clear all CCTV session-state keys.
    Safe to call even if CCTV was never started.
    Does NOT touch any non-CCTV session state.
    """
    cap = st.session_state.get("cctv_camera")
    release_cctv_camera(cap)

    for key in [
        "cctv_running", "cctv_session_id", "cctv_start_time",
        "cctv_normal_closed", "cctv_normal_close_time",
        "cctv_subject_id", "cctv_confirmed_students", "cctv_late_students",
        "cctv_frame_tracker", "cctv_camera", "cctv_frame_counter",
        "cctv_last_recognized", "cctv_last_frame",
        "cctv_enrolled_students", "cctv_candidate_embeddings", "cctv_error",
    ]:
        st.session_state.pop(key, None)


# ===========================================================================
# Student name lookup
# ===========================================================================

def _student_name(student_id: int) -> str:
    """Look up student name from the cached enrolled list in session state."""
    enrolled = st.session_state.get("cctv_enrolled_students", [])
    for s in enrolled:
        if int(s["student_id"]) == student_id:
            return s["name"]
    return str(student_id)


# ===========================================================================
# Frame Processing (called each rerun while CCTV is active)
# ===========================================================================

def _process_one_frame(config: dict) -> None:
    """
    Capture and process a single CCTV frame.
    Updates session state with recognition results.
    Called once per fragment rerun.
    """
    cap = st.session_state.get("cctv_camera")
    if cap is None:
        st.session_state.cctv_error = "Camera not available."
        return

    # Increment frame counter
    st.session_state.cctv_frame_counter += 1
    n = config.get("process_every_n_frames", 3)

    # Capture frame regardless (for display)
    success, frame_rgb = capture_cctv_frame(cap)

    if not success:
        st.session_state.cctv_error = (
            "⚠️ Could not read camera frame. Connection may be lost. "
            "Previously recorded attendance has been preserved."
        )
        return

    # Store for display
    st.session_state.cctv_last_frame = frame_rgb
    st.session_state.cctv_error = None

    # Only run face recognition every N-th frame
    if st.session_state.cctv_frame_counter % n != 0:
        return

    candidate_embeddings = st.session_state.get("cctv_candidate_embeddings", {})
    if not candidate_embeddings:
        return

    threshold  = config.get("face_threshold", 0.60)
    recognized = process_cctv_frame(frame_rgb, candidate_embeddings, threshold)

    for rec in recognized:
        student_id = int(rec["student_id"])
        similarity = rec["similarity"]

        # Already confirmed — nothing more to do
        if student_id in st.session_state.cctv_confirmed_students:
            continue

        # Temporal confirmation check
        confirmed = update_confirmation_tracker(
            student_id=student_id,
            frame_tracker=st.session_state.cctv_frame_tracker,
            confirmation_frames=config.get("confirmation_frames", 5),
            confirmation_window=config.get("confirmation_window", 10.0),
        )

        if not confirmed:
            continue   # Accumulating frames — not yet reliable enough

        # ── Confirmed — decide Present or Present+Late ─────────────────
        subject_id     = st.session_state.cctv_subject_id
        normal_closed  = st.session_state.cctv_normal_closed

        if not normal_closed:
            # Normal attendance window → Present
            written = mark_cctv_present(student_id, subject_id)
            # Add to confirmed set whether newly written or already present
            if written or check_student_present_today(student_id, subject_id):
                st.session_state.cctv_confirmed_students.add(student_id)
                name = _student_name(student_id)
                st.session_state.cctv_last_recognized = f"✅ {name} (Present)"
        else:
            # Late window → Present + Late (only if not already present)
            if check_student_present_today(student_id, subject_id):
                # Already marked present — add to confirmed, don't duplicate
                st.session_state.cctv_confirmed_students.add(student_id)
            else:
                late_written = mark_cctv_late(student_id, subject_id, similarity)
                if late_written:
                    st.session_state.cctv_confirmed_students.add(student_id)
                    st.session_state.cctv_late_students.add(student_id)
                    name = _student_name(student_id)
                    st.session_state.cctv_last_recognized = (
                        f"🟡 {name} (Present + Late)"
                    )


# ===========================================================================
# Streamlit Fragment — auto-rerunning live feed
# ===========================================================================

@st.fragment(run_every=2)
def _live_cctv_fragment(config: dict) -> None:
    """
    Auto-rerunning fragment (every 2 seconds) that processes CCTV frames
    and displays the live feed without blocking the main page.
    """
    if not st.session_state.get("cctv_running", False):
        return

    cap = st.session_state.get("cctv_camera")
    if cap is None:
        st.warning("Camera not available.")
        return

    # ── Capture frame ─────────────────────────────────────────────────────
    success, frame_rgb = capture_cctv_frame(cap)
    if not success:
        st.warning("⚠️ Could not read frame. Check camera connection.")
        return

    st.session_state.cctv_last_frame = frame_rgb
    st.session_state.cctv_error = None

    # ── Face recognition on this frame ────────────────────────────────────
    candidate_embeddings = st.session_state.get("cctv_candidate_embeddings", {})
    threshold  = config.get("face_threshold", 0.40)
    subject_id = st.session_state.cctv_subject_id
    normal_closed = st.session_state.cctv_normal_closed

    # Run recognition and collect raw scores for debug display
    raw_results = []
    if candidate_embeddings:
        raw_results = process_cctv_frame(frame_rgb, candidate_embeddings, threshold=0.0)
        # ↑ threshold=0.0 to see ALL scores regardless of pass/fail

    # ── Debug panel — similarity scores ───────────────────────────────────
    with st.expander("🔍 Recognition Debug (live scores)", expanded=True):
        if not raw_results:
            st.caption("No face detected in this frame.")
        else:
            for rec in raw_results:
                sid   = rec["student_id"]
                score = rec["similarity"]
                name  = _student_name(sid)
                bar_color = "🟢" if score >= threshold else "🔴"
                frames_so_far = len(st.session_state.cctv_frame_tracker.get(sid, []))
                needed = config.get("confirmation_frames", 2)
                st.caption(
                    f"{bar_color} **{name}** — score: `{score:.3f}` "
                    f"(threshold: `{threshold}`) | "
                    f"confirmation: {frames_so_far}/{needed}"
                )

    # ── Process recognitions above threshold ──────────────────────────────
    for rec in raw_results:
        student_id = int(rec["student_id"])
        similarity = rec["similarity"]

        if similarity < threshold:
            continue   # below threshold — skip

        if student_id in st.session_state.cctv_confirmed_students:
            continue   # already confirmed

        confirmed = update_confirmation_tracker(
            student_id=student_id,
            frame_tracker=st.session_state.cctv_frame_tracker,
            confirmation_frames=config.get("confirmation_frames", 2),
            confirmation_window=config.get("confirmation_window", 20.0),
        )

        if not confirmed:
            continue

        # ── Confirmed — mark attendance ───────────────────────────────────
        if not normal_closed:
            written = mark_cctv_present(student_id, subject_id)
            if written or check_student_present_today(student_id, subject_id):
                st.session_state.cctv_confirmed_students.add(student_id)
                name = _student_name(student_id)
                st.session_state.cctv_last_recognized = f"✅ {name} (Present)"
                st.toast(f"✅ {name} marked Present!", icon="✅")
        else:
            if check_student_present_today(student_id, subject_id):
                st.session_state.cctv_confirmed_students.add(student_id)
            else:
                late_written = mark_cctv_late(student_id, subject_id, similarity)
                if late_written:
                    st.session_state.cctv_confirmed_students.add(student_id)
                    st.session_state.cctv_late_students.add(student_id)
                    name = _student_name(student_id)
                    st.session_state.cctv_last_recognized = f"🟡 {name} (Present + Late)"
                    st.toast(f"🟡 {name} marked Present + Late!", icon="🟡")

    # ── Live camera display ───────────────────────────────────────────────
    st.image(frame_rgb, caption="📹 Live CCTV Feed", use_container_width=True)

    # ── Confirmed students list ───────────────────────────────────────────
    confirmed = st.session_state.get("cctv_confirmed_students", set())
    late      = st.session_state.get("cctv_late_students", set())
    if confirmed:
        st.markdown("**Confirmed this session:**")
        for sid in sorted(confirmed):
            name = _student_name(sid)
            tag  = " 🟡 Late" if sid in late else " ✅ Present"
            st.markdown(f"- {name}{tag}")



# ===========================================================================
# Status Panel
# ===========================================================================

def _render_status_panel() -> None:
    """Render the CCTV status banner and metrics."""
    cctv_running   = st.session_state.get("cctv_running", False)
    normal_closed  = st.session_state.get("cctv_normal_closed", False)
    confirmed      = st.session_state.get("cctv_confirmed_students", set())
    late           = st.session_state.get("cctv_late_students", set())
    enrolled       = st.session_state.get("cctv_enrolled_students", [])

    # ── State banner ─────────────────────────────────────────────────────
    if not cctv_running:
        st.markdown(
            """<div style="background:#95a5a6;border-radius:1rem;
                          padding:0.8rem 1.2rem;margin-bottom:0.5rem;">
                <h3 style="color:white;margin:0;font-family:'Outfit',sans-serif;">
                    🔴 CCTV Not Running
                </h3></div>""",
            unsafe_allow_html=True,
        )
        normal_status = "Not Started"
    elif not normal_closed:
        st.markdown(
            """<div style="background:#2ecc71;border-radius:1rem;
                          padding:0.8rem 1.2rem;margin-bottom:0.5rem;">
                <h3 style="color:white;margin:0;font-family:'Outfit',sans-serif;">
                    📹 CCTV Attendance ACTIVE — Normal Window
                </h3></div>""",
            unsafe_allow_html=True,
        )
        normal_status = "Open"
    else:
        st.markdown(
            """<div style="background:#e67e22;border-radius:1rem;
                          padding:0.8rem 1.2rem;margin-bottom:0.5rem;">
                <h3 style="color:white;margin:0;font-family:'Outfit',sans-serif;">
                    📹 CCTV Active — Late Window
                </h3></div>""",
            unsafe_allow_html=True,
        )
        normal_status = "Closed"

    # ── Metrics ──────────────────────────────────────────────────────────
    total_enrolled  = len(enrolled)
    late_count      = len(late)
    present_count   = len(confirmed) - late_count
    absent_estimate = max(total_enrolled - len(confirmed), 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Present",          present_count)
    m2.metric("Late",             late_count)
    m3.metric("Absent (so far)",  absent_estimate)
    m4.metric("Enrolled",         total_enrolled)

    # ── Timing ───────────────────────────────────────────────────────────
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(f"**Normal Attendance:** {normal_status}")
        if cctv_running:
            start = st.session_state.get("cctv_start_time", "—")
            st.caption(f"⏱ Started: {start}")
    with info_col2:
        last = st.session_state.get("cctv_last_recognized", "None")
        st.markdown(f"**Last Recognized:** {last}")
        if normal_closed:
            close_t = st.session_state.get("cctv_normal_close_time", "—")
            st.caption(f"🔒 Normal closed: {close_t}")


# ===========================================================================
# Button Handlers
# ===========================================================================

def _handle_start_cctv(subject_id: int, teacher_id: int, config: dict) -> None:
    """Start CCTV Attendance — opens camera, loads embeddings, creates DB session."""
    if st.session_state.cctv_running:
        return   # Rerun guard: already running

    source = config["source"]
    with st.spinner("🔌 Connecting to camera…"):
        cap = get_cctv_camera(source)

    if cap is None:
        source_display = "webcam (0)" if source == 0 else f"RTSP stream"
        st.error(
            f"❌ **Camera connection failed** ({source_display}).\n\n"
            "Please check:\n"
            "- The camera is plugged in and not used by another app.\n"
            "- For IP cameras: verify the RTSP URL and network."
        )
        return

    # Load subject-enrolled students and their face embeddings
    enrolled = get_subject_students_with_embeddings(subject_id)
    if not enrolled:
        release_cctv_camera(cap)
        st.error("❌ No students are enrolled in this subject. Cannot start CCTV.")
        return

    candidate_embeddings = build_candidate_embeddings(enrolled)
    if not candidate_embeddings:
        release_cctv_camera(cap)
        st.warning(
            "⚠️ None of the enrolled students have a face profile yet. "
            "Ask students to enroll via the Student Portal first."
        )
        return

    # Create DB session — guard against rerun creating duplicate sessions
    session_id = start_cctv_session(subject_id, teacher_id)
    if session_id is None:
        release_cctv_camera(cap)
        return   # Error already shown by start_cctv_session()

    # Initialise session state for this CCTV run
    st.session_state.cctv_running              = True
    st.session_state.cctv_session_id           = session_id
    st.session_state.cctv_start_time           = datetime.now().strftime("%I:%M %p")
    st.session_state.cctv_normal_closed        = False
    st.session_state.cctv_normal_close_time    = None
    st.session_state.cctv_subject_id           = subject_id
    st.session_state.cctv_confirmed_students   = set()
    st.session_state.cctv_late_students        = set()
    st.session_state.cctv_frame_tracker        = {}
    st.session_state.cctv_camera               = cap
    st.session_state.cctv_frame_counter        = 0
    st.session_state.cctv_last_recognized      = "None"
    st.session_state.cctv_last_frame           = None
    st.session_state.cctv_enrolled_students    = enrolled
    st.session_state.cctv_candidate_embeddings = candidate_embeddings
    st.session_state.cctv_error                = None

    st.toast(
        f"📹 CCTV started! Monitoring {len(enrolled)} enrolled students.",
        icon="✅",
    )
    st.rerun()


def _handle_close_normal_attendance() -> None:
    """Close normal attendance window — CCTV keeps running."""
    if not st.session_state.cctv_running or st.session_state.cctv_normal_closed:
        return

    session_id = st.session_state.get("cctv_session_id")
    if session_id:
        _update_session_normal_close(session_id)

    st.session_state.cctv_normal_closed     = True
    st.session_state.cctv_normal_close_time = datetime.now().strftime("%I:%M %p")

    # CCTV continues running — only the attendance window changes
    st.toast("🔒 Normal attendance closed. CCTV continues for latecomers.", icon="🔒")
    st.rerun()


def _handle_stop_cctv(subject_id: int) -> None:
    """Stop CCTV — finalize absent records, update DB, release camera."""
    if not st.session_state.cctv_running:
        return

    confirmed  = st.session_state.cctv_confirmed_students.copy()
    late       = st.session_state.cctv_late_students.copy()
    session_id = st.session_state.get("cctv_session_id")

    with st.spinner("⏹ Finalizing attendance…"):
        # 1. Update session record in DB
        if session_id:
            _finalize_session_db(session_id)

        # 2. Mark absent for all enrolled students not confirmed
        absent_count = finalize_cctv_attendance(subject_id, confirmed)

    # 3. Release camera + clear state
    _reset_cctv_state()

    # 4. Show final summary
    present_only = len(confirmed) - len(late)
    late_count   = len(late)
    st.success(
        f"✅ **CCTV Session Complete!**\n\n"
        f"- 🟢 Present (on time): **{present_only}**\n"
        f"- 🟡 Present + Late:    **{late_count}**\n"
        f"- 🔴 Absent:            **{absent_count}**\n\n"
        "Records saved. View in **Attendance Records** or **Latecomers** tabs."
    )
    st.rerun()


# ===========================================================================
# Public Entry Point
# ===========================================================================

def cctv_attendance_section(teacher_id: int) -> None:
    """
    Render the complete CCTV Attendance section inside the Teacher Portal.

    Called from teacher_tab_cctv_attendance() in teacher_screen.py.
    Uses an inline section (not @st.dialog) so that the camera and session
    state persist across Streamlit reruns — same pattern as
    latecomer_attendance_section() in dialog_latecomer.py.

    Parameters
    ----------
    teacher_id : int
        The authenticated teacher's ID (from st.session_state.teacher_data).
    """
    # ── Guard: CCTV not configured ────────────────────────────────────────
    if not is_cctv_configured():
        with st.container(border=True):
            st.info(
                "📷 **CCTV camera is not configured.**\n\n"
                "Add the following optional keys to `.streamlit/secrets.toml` "
                "to enable CCTV Attendance:\n\n"
                "```toml\n"
                'CCTV_SOURCE = "0"                  # 0 = webcam, or RTSP URL\n'
                'CCTV_FACE_THRESHOLD = "0.60"\n'
                'CCTV_CONFIRMATION_FRAMES = "5"\n'
                'CCTV_CONFIRMATION_WINDOW = "10"\n'
                'CCTV_PROCESS_EVERY_N_FRAMES = "3"\n'
                "```\n\n"
                "All other existing features continue to work normally."
            )
        return

    config = get_cctv_config()
    _init_cctv_state()

    cctv_running  = st.session_state.cctv_running
    normal_closed = st.session_state.cctv_normal_closed

    # ── Subject Selector ──────────────────────────────────────────────────
    subjects = get_teacher_subjects(teacher_id)
    if not subjects:
        st.warning("No subjects found. Create one in Manage Subjects first.")
        return

    subject_options = {
        f"{s['name']} — {s['subject_code']}": s["subject_id"]
        for s in subjects
    }

    selected_label = st.selectbox(
        "Select Subject",
        options=list(subject_options.keys()),
        key="cctv_subject_select",
        disabled=cctv_running,   # Lock subject while session is in progress
    )
    selected_subject_id = subject_options[selected_label]

    # Camera source info (no credentials displayed)
    source = config["source"]
    if source == 0 or source == "0":
        source_label = "Webcam (built-in / USB)"
    else:
        source_label = "IP Camera (RTSP)"
    st.caption(f"📷 Camera: **{source_label}**")

    st.divider()

    # ── Three Action Buttons ──────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        if st.button(
            "▶ Start CCTV Attendance",
            type="primary",
            width="stretch",
            disabled=cctv_running,
            key="cctv_start_btn",
        ):
            _handle_start_cctv(selected_subject_id, teacher_id, config)

    with btn2:
        if st.button(
            "🔒 Close Normal Attendance",
            type="secondary",
            width="stretch",
            disabled=(not cctv_running or normal_closed),
            key="cctv_close_normal_btn",
        ):
            _handle_close_normal_attendance()

    with btn3:
        if st.button(
            "⏹ Stop CCTV",
            type="tertiary",
            width="stretch",
            disabled=not cctv_running,
            key="cctv_stop_btn",
        ):
            _handle_stop_cctv(selected_subject_id)

    st.divider()

    # ── Status Panel ──────────────────────────────────────────────────────
    _render_status_panel()

    # ── Live Feed Fragment (auto-reruns while camera is running) ──────────
    if cctv_running:
        st.divider()
        _live_cctv_fragment(config)
