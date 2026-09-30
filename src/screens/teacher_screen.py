"""
FusionPresence — Teacher Portal Screen
========================================
Handles:
  - Teacher login / registration
  - Take Attendance  (Face Recognition + Voice Attendance)
  - Manage Subjects  (Create / Share)
  - Attendance Records  (Daily View + Weekly View)
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from src.components.subject_card import subject_card
from src.database.db import (
    check_teacher_exists,
    create_teacher,
    teacher_login,
    get_teacher_subjects,
    get_attendance_for_teacher,
    get_subject_students_with_embeddings,
    get_attendance_for_subject,
)
from src.components.dialog_create_subject import create_subject_dialog
from src.components.dialog_share_subject import share_subject_dialog
from src.components.dialog_add_photo import add_photos_dialog
from src.components.dialog_attendance_results import attendance_result_dialog
from src.components.dialog_voice_attendance import voice_attendance_dialog
from src.database.config import supabase

# New ArcFace pipeline
from src.pipline.face_pipeline import recognize_faces, RECOGNITION_THRESHOLD


# ===========================================================================
# Teacher Screen Router
# ===========================================================================

def teacher_screen():
    style_background_dashboard()
    style_base_layout()

    if "teacher_data" in st.session_state:
        teacher_dashboard()
    elif 'teacher_login_type' not in st.session_state or st.session_state.teacher_login_type == "login":
        teacher_screen_login()
    elif st.session_state.teacher_login_type == "register":
        teacher_screen_register()


# ===========================================================================
# Teacher Dashboard (tab router)
# ===========================================================================

def teacher_dashboard():
    teacher_data = st.session_state.teacher_data

    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"Welcome, {teacher_data['name']}")
        if st.button("Logout", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['is_logged_in'] = False
            del st.session_state.teacher_data
            st.rerun()

    st.space()

    if "current_teacher_tab" not in st.session_state:
        st.session_state.current_teacher_tab = 'take_attendance'

    tab1, tab2, tab3 = st.columns(3)

    with tab1:
        t1_type = "primary" if st.session_state.current_teacher_tab == 'take_attendance' else "tertiary"
        if st.button('Take Attendance', type=t1_type, width='stretch', icon=':material/ar_on_you:'):
            st.session_state.current_teacher_tab = 'take_attendance'
            st.rerun()

    with tab2:
        t2_type = "primary" if st.session_state.current_teacher_tab == 'manage_subjects' else "tertiary"
        if st.button('Manage Subjects', type=t2_type, width='stretch', icon=':material/book_ribbon:'):
            st.session_state.current_teacher_tab = 'manage_subjects'
            st.rerun()

    with tab3:
        t3_type = "primary" if st.session_state.current_teacher_tab == 'attendance_records' else "tertiary"
        if st.button('Attendance Records', type=t3_type, width='stretch', icon=':material/cards_stack:'):
            st.session_state.current_teacher_tab = 'attendance_records'
            st.rerun()

    st.divider()

    if st.session_state.current_teacher_tab == "take_attendance":
        teacher_tab_take_attendance()
    if st.session_state.current_teacher_tab == "manage_subjects":
        teacher_tab_manage_subjects()
    if st.session_state.current_teacher_tab == "attendance_records":
        teacher_tab_attendance_records()

    footer_dashboard()


# ===========================================================================
# Tab 1 — Take Attendance
# ===========================================================================

def teacher_tab_take_attendance():
    teacher_id = st.session_state.teacher_data['teacher_id']
    st.header('Take Attendance')

    if 'attendance_images' not in st.session_state:
        st.session_state.attendance_images = []

    subjects = get_teacher_subjects(teacher_id)
    if not subjects:
        st.warning('No subjects found. Create one to get started.')
        return

    subject_options = {f"{s['name']} — {s['subject_code']}": s['subject_id'] for s in subjects}

    col1, col2 = st.columns([3, 1], vertical_alignment='bottom')
    with col1:
        selected_subject_label = st.selectbox('Select Subject', options=list(subject_options.keys()))
    with col2:
        if st.button('Add Photos', type='primary', icon=':material/photo_prints:', width='stretch'):
            add_photos_dialog()

    selected_subject_id = subject_options[selected_subject_label]

    st.divider()

    # Photo gallery preview
    if st.session_state.attendance_images:
        st.subheader('Added Photos')
        gallery_cols = st.columns(4)
        for idx, img in enumerate(st.session_state.attendance_images):
            with gallery_cols[idx % 4]:
                st.image(img, width='stretch', caption=f'Photo {idx + 1}')

    has_photos = bool(st.session_state.attendance_images)
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button(
            'Clear All Photos', width='stretch', type='tertiary',
            icon=':material/delete:', disabled=not has_photos
        ):
            st.session_state.attendance_images = []
            st.rerun()

    with c2:
        if st.button(
            'Run Face Recognition', width='stretch', type='secondary',
            icon=':material/face_2:', disabled=not has_photos
        ):
            _run_face_recognition(selected_subject_id)

    with c3:
        if st.button(
            'Use Voice Attendance', type='primary', width='stretch',
            icon=':material/mic:'
        ):
            voice_attendance_dialog(selected_subject_id)


def _run_face_recognition(selected_subject_id: int):
    """
    Run ArcFace recognition on all uploaded classroom photos.

    - Loads only enrolled students for the selected subject (candidate filtering).
    - Calls recognize_faces() for each photo.
    - Aggregates results across all photos.
    - Shows attendance results dialog for teacher confirmation.
    """
    with st.spinner('Running face recognition on classroom photos…'):

        # ── 1. Load subject-specific candidates ──────────────────────────
        enrolled_students = get_subject_students_with_embeddings(selected_subject_id)

        if not enrolled_students:
            st.warning('No students are enrolled in this subject.')
            return

        # Build candidate dict: student_id → face_embedding list
        candidate_embeddings = {
            s['student_id']: s.get('face_embedding')
            for s in enrolled_students
            if s.get('face_embedding')
        }

        if not candidate_embeddings:
            st.warning(
                'None of the enrolled students have a face profile yet.  '
                'Ask students to enroll their face via the Student Portal.'
            )
            return

        # ── 2. Run recognition on each photo ─────────────────────────────
        # all_detected: student_id → list of photo labels where detected
        all_detected: dict[int, list[str]] = {}
        # all_scores:   student_id → best similarity score seen
        all_scores: dict[int, float] = {}

        for idx, img in enumerate(st.session_state.attendance_images):
            img_np = np.array(img.convert('RGB'))
            results = recognize_faces(img_np, candidate_embeddings, RECOGNITION_THRESHOLD)

            for r in results:
                if r['status'] == 'recognized' and r['student_id'] is not None:
                    sid   = int(r['student_id'])
                    score = r['similarity']
                    label = f"Photo {idx + 1}"
                    all_detected.setdefault(sid, []).append(label)
                    if score > all_scores.get(sid, -1):
                        all_scores[sid] = score

        # ── 3. Build results table ────────────────────────────────────────
        results_rows  = []
        attendance_to_log = []
        current_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        for student in enrolled_students:
            sid     = int(student['student_id'])
            sources = all_detected.get(sid, [])
            score   = all_scores.get(sid)
            is_present = len(sources) > 0

            results_rows.append({
                "Name":       student['name'],
                "ID":         sid,
                "Detected In": ", ".join(sources) if is_present else "—",
                "Similarity":  f"{score:.2f}" if score is not None and is_present else "—",
                "Status":      "✅ Present" if is_present else "❌ Absent",
            })

            attendance_to_log.append({
                'student_id': sid,
                'subject_id': selected_subject_id,
                'timestamp':  current_timestamp,
                'is_present': bool(is_present),
            })

        attendance_result_dialog(pd.DataFrame(results_rows), attendance_to_log)


# ===========================================================================
# Tab 2 — Manage Subjects
# ===========================================================================

def teacher_tab_manage_subjects():
    teacher_id = st.session_state.teacher_data['teacher_id']

    col1, col2 = st.columns(2)
    with col1:
        st.header('Manage Subjects')
    with col2:
        if st.button('Create New Subject', type="primary", width='stretch'):
            create_subject_dialog(teacher_id)

    subjects = get_teacher_subjects(teacher_id)
    if subjects:
        for sub in subjects:
            stats = [
                ("🫂", "Students", sub['total_students']),
                ("🕰️", "Classes",  sub['total_classes']),
            ]

            def share_btn(sub=sub):
                if st.button(
                    f"Share — {sub['name']}",
                    type="primary",
                    key=f"share_{sub['subject_code']}",
                    icon=":material/share:"
                ):
                    share_subject_dialog(sub['name'], sub['subject_code'])
                st.space()

            subject_card(
                name=sub['name'],
                code=sub['subject_code'],
                section=sub['section'],
                stats=stats,
                footer_callback=share_btn,
            )
    else:
        st.info("No subjects found. Create one using the button above.")


# ===========================================================================
# Tab 3 — Attendance Records  (Daily + Weekly)
# ===========================================================================

def teacher_tab_attendance_records():
    st.header('Attendance Records')
    teacher_id = st.session_state.teacher_data['teacher_id']

    # ── View toggle ───────────────────────────────────────────────────────
    if 'attendance_view' not in st.session_state:
        st.session_state.attendance_view = 'daily'

    v1, v2 = st.columns(2)
    with v1:
        t_daily = "primary" if st.session_state.attendance_view == 'daily' else "tertiary"
        if st.button('Daily View', type=t_daily, width='stretch', icon=':material/calendar_today:'):
            st.session_state.attendance_view = 'daily'
            st.rerun()
    with v2:
        t_weekly = "primary" if st.session_state.attendance_view == 'weekly' else "tertiary"
        if st.button('Weekly View', type=t_weekly, width='stretch', icon=':material/calendar_view_week:'):
            st.session_state.attendance_view = 'weekly'
            st.rerun()

    st.divider()

    if st.session_state.attendance_view == 'daily':
        _daily_attendance_view(teacher_id)
    else:
        _weekly_attendance_view(teacher_id)


# ---------------------------------------------------------------------------
# Daily View  (original behaviour, preserved exactly)
# ---------------------------------------------------------------------------

def _daily_attendance_view(teacher_id: int):
    records = get_attendance_for_teacher(teacher_id)

    if not records:
        st.info("No attendance records found.")
        return

    data = []
    for r in records:
        ts = r.get('timestamp')
        data.append({
            "ts_group":    ts.split(".")[0] if ts else None,
            "Time":        datetime.fromisoformat(ts).strftime("%Y-%m-%d  %I:%M %p") if ts else "N/A",
            "Subject":     r['subjects']['name'],
            "Subject Code": r['subjects']['subject_code'],
            "is_present":  bool(r.get('is_present', False)),
        })

    df = pd.DataFrame(data)

    summary = (
        df.groupby(['ts_group', 'Time', 'Subject', 'Subject Code'])
        .agg(
            Present_Count=('is_present', 'sum'),
            Total_Count=('is_present', 'count'),
        )
        .reset_index()
    )

    summary['Attendance'] = (
        "✅ " + summary['Present_Count'].astype(str) +
        " / " + summary['Total_Count'].astype(str) + " students"
    )

    display_df = (
        summary.sort_values(by='ts_group', ascending=False)
        [['Time', 'Subject', 'Subject Code', 'Attendance']]
    )

    st.dataframe(display_df, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Weekly View  (new functionality)
# ---------------------------------------------------------------------------

def _get_week_options() -> list[tuple[str, datetime, datetime]]:
    """
    Generate a list of the last 12 calendar weeks (Mon–Sun) ending today.

    Returns
    -------
    list of (label, week_start, week_end)
    """
    today = datetime.now().date()
    # Find the Monday of the current week
    current_monday = today - timedelta(days=today.weekday())

    weeks = []
    for i in range(12):
        wstart = current_monday - timedelta(weeks=i)
        wend   = wstart + timedelta(days=6)   # Sunday
        label  = f"{wstart.strftime('%d %b')} – {wend.strftime('%d %b %Y')}"
        weeks.append((label, datetime.combine(wstart, datetime.min.time()),
                      datetime.combine(wend, datetime.max.time())))
    return weeks


def _weekly_attendance_view(teacher_id: int):
    """
    Display a weekly attendance grid for a selected subject and week.

    Rows    = enrolled students
    Columns = Mon / Tue / Wed / Thu / Fri  (dates that have sessions)
    Cell    = ✓ present  |  ✗ absent  |  — no session that day
    """
    # ── Subject selector ─────────────────────────────────────────────────
    subjects = get_teacher_subjects(teacher_id)
    if not subjects:
        st.info("No subjects found.")
        return

    subject_options = {f"{s['name']} — {s['subject_code']}": s for s in subjects}
    selected_label  = st.selectbox('Select Subject', options=list(subject_options.keys()),
                                   key='weekly_subject_select')
    selected_subject = subject_options[selected_label]
    subject_id       = selected_subject['subject_id']

    # ── Week selector ─────────────────────────────────────────────────────
    week_options   = _get_week_options()
    week_labels    = [w[0] for w in week_options]
    selected_week_label = st.selectbox('Select Week', options=week_labels,
                                       key='weekly_week_select')
    _, week_start, week_end = next(w for w in week_options if w[0] == selected_week_label)

    st.markdown(
        f"### 📅 Week: **{week_start.strftime('%d %b')} – {week_end.strftime('%d %b %Y')}**"
    )

    # ── Fetch data ────────────────────────────────────────────────────────
    logs            = get_attendance_for_subject(subject_id)
    enrolled        = get_subject_students_with_embeddings(subject_id)

    if not enrolled:
        st.info("No students are enrolled in this subject.")
        return

    if not logs:
        st.info("No attendance records found for this subject.")
        return

    # Filter logs to selected week
    week_logs = []
    for log in logs:
        ts = log.get('timestamp')
        if not ts:
            continue
        try:
            log_dt = datetime.fromisoformat(ts)
            # Supabase timestamps may be timezone-aware; strip tz to compare
            # with the naive week_start / week_end datetimes from _get_week_options()
            if log_dt.tzinfo is not None:
                log_dt = log_dt.replace(tzinfo=None)
        except ValueError:
            continue
        if week_start <= log_dt <= week_end:
            week_logs.append(log)

    if not week_logs:
        st.info(f"No attendance sessions recorded during {selected_week_label}.")
        return

    # ── Identify unique session timestamps (de-duplicated per-subject) ───
    # A "session" is a unique timestamp string (as saved by attendance taking)
    unique_sessions = sorted(set(log['timestamp'] for log in week_logs))

    # Map session → weekday label (Mon 22 Sep, etc.)
    session_day_labels: dict[str, str] = {}
    for ts in unique_sessions:
        dt = datetime.fromisoformat(ts)
        session_day_labels[ts] = dt.strftime("%a\n%d %b")

    # ── Build attendance pivot ────────────────────────────────────────────
    # present_map[student_id][session_ts] = True/False
    present_map: dict[int, dict[str, bool]] = {}
    for log in week_logs:
        sid   = int(log['student_id'])
        ts    = log['timestamp']
        pres  = bool(log.get('is_present', False))
        if sid not in present_map:
            present_map[sid] = {}
        # If the same student appears multiple times in one session (shouldn't happen), take True if any
        present_map[sid][ts] = present_map[sid].get(ts, False) or pres

    # ── Build display table ───────────────────────────────────────────────
    col_labels = [session_day_labels[ts] for ts in unique_sessions]

    rows = []
    for student in enrolled:
        sid   = int(student['student_id'])
        name  = student['name']
        row   = {"Student": name}

        present_count = 0
        for ts in unique_sessions:
            day_label = session_day_labels[ts]
            is_pres   = present_map.get(sid, {}).get(ts)
            if is_pres is None:
                row[day_label] = "—"
            elif is_pres:
                row[day_label] = "✓"
                present_count += 1
            else:
                row[day_label] = "✗"

        total_sessions = len(unique_sessions)
        pct            = round(present_count / total_sessions * 100) if total_sessions > 0 else 0
        row["Sessions"] = f"{present_count}/{total_sessions}"
        row["Attendance %"] = f"{pct}%"
        rows.append(row)

    weekly_df = pd.DataFrame(rows)

    # ── Render grid ───────────────────────────────────────────────────────
    st.dataframe(
        weekly_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Student":       st.column_config.TextColumn("Student", width="medium"),
            "Sessions":      st.column_config.TextColumn("Sessions"),
            "Attendance %":  st.column_config.TextColumn("Attendance %"),
        },
    )

    # ── Summary stats ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Attendance Summary")

    total_students  = len(enrolled)
    avg_pct = (
        sum(
            (sum(1 for ts in unique_sessions if present_map.get(int(s['student_id']), {}).get(ts)) /
             len(unique_sessions) * 100)
            for s in enrolled
        ) / total_students
        if total_students > 0 and unique_sessions
        else 0
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Students",  total_students)
    m2.metric("Sessions This Week", len(unique_sessions))
    m3.metric("Average Attendance", f"{avg_pct:.0f}%")


# ===========================================================================
# Login helpers
# ===========================================================================

def login_teacher(username: str, password: str) -> bool:
    if not username or not password:
        return False
    teacher = teacher_login(username, password)
    if teacher:
        st.session_state.user_role   = 'teacher'
        st.session_state.teacher_data = teacher
        st.session_state.is_logged_in = True
        return True
    return False


def teacher_screen_login():
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go Back to Home", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            st.rerun()

    st.header('Sign in to Teacher Portal', text_alignment='center')
    st.space()
    st.space()

    teacher_username = st.text_input("Username", placeholder='e.g. ananyaroy')
    teacher_pass     = st.text_input("Password", type='password', placeholder='Enter password')
    st.divider()

    btnc1, btnc2 = st.columns(2)
    with btnc1:
        if st.button('Login', type='primary', icon=':material/passkey:', shortcut='control+enter', width='stretch'):
            if login_teacher(teacher_username, teacher_pass):
                st.toast("Welcome back!", icon="👋")
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.error("Incorrect username or password. Please try again.")

    with btnc2:
        if st.button('Register Instead', type="primary", icon=':material/passkey:', width='stretch'):
            st.session_state.teacher_login_type = 'register'

    footer_dashboard()


def register_teacher(teacher_username, teacher_name, teacher_pass, teacher_pass_confirm):
    if not teacher_username or not teacher_name or not teacher_pass:
        return False, "All fields are required."
    if check_teacher_exists(teacher_username):
        return False, "That username is already taken. Please choose another."
    if teacher_pass != teacher_pass_confirm:
        return False, "Passwords do not match."
    try:
        create_teacher(teacher_username, teacher_pass, teacher_name)
        return True, "Account created successfully! You can now log in."
    except Exception:
        return False, "An unexpected error occurred. Please try again."


def teacher_screen_register():
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go Back to Home", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            st.rerun()

    st.header('Register Teacher Account')
    st.space()
    st.space()

    teacher_username     = st.text_input("Username",          placeholder='e.g. ananyaroy')
    teacher_name         = st.text_input("Full Name",         placeholder='e.g. Ananya Roy')
    teacher_pass         = st.text_input("Password",          type='password', placeholder='Enter password')
    teacher_pass_confirm = st.text_input("Confirm Password",  type='password', placeholder='Re-enter password')

    st.divider()

    btnc1, btnc2 = st.columns(2)
    with btnc1:
        if st.button('Register Now', type='primary', icon=':material/passkey:', shortcut='control+enter', width='stretch'):
            success, message = register_teacher(
                teacher_username, teacher_name, teacher_pass, teacher_pass_confirm
            )
            if success:
                st.success(message)
                import time
                time.sleep(2)
                st.session_state.teacher_login_type = "login"
                st.rerun()
            else:
                st.error(message)

    with btnc2:
        if st.button('Login Instead', type="primary", icon=':material/passkey:', width='stretch'):
            st.session_state.teacher_login_type = 'login'

    footer_dashboard()