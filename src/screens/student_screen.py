"""
FusionPresence — Student Portal Screen
=======================================
Handles:
  - Student face-based login (ArcFace recognition)
  - Student registration (name + face enrollment + optional voice)
  - Student dashboard (enrolled subjects + attendance stats)
  - Face re-enrollment for students with outdated Dlib embeddings
"""

import streamlit as st
import numpy as np
from PIL import Image
import time

from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard

# New ArcFace pipeline
from src.pipline.face_pipeline import (
    predict_student_login,
    enroll_face_embedding,
    is_old_dlib_embedding,
    RECOGNITION_THRESHOLD,
)
from src.pipline.voice_pipline import get_voice_embedding
from src.database.db import (
    get_all_students,
    create_student,
    get_student_subjects,
    get_student_attendance,
    unenroll_student_to_subject,
    update_student_face_embedding,
)
from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def student_dashboard():
    student_data = st.session_state.student_data
    student_id   = student_data['student_id']

    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"Welcome, {student_data['name']}")
        if st.button("Logout", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['is_logged_in'] = False
            del st.session_state.student_data
            st.rerun()

    st.space()

    # ── Re-enrollment banner ──────────────────────────────────────────────
    face_emb = student_data.get('face_embedding')
    if not face_emb or is_old_dlib_embedding(face_emb):
        st.warning(
            "⚠️ **Your face data needs to be updated.**  "
            "Your existing face profile uses an older format and cannot be used "
            "for face recognition.  Please re-enroll your face below.",
            icon="⚠️"
        )
        _re_enrollment_section(student_id)
        st.divider()

    # ── Enrolled Subjects ─────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.header('Enrolled Subjects')
    with c2:
        if st.button('Enroll in Subject', type='primary', width='stretch'):
            enroll_dialog()

    st.divider()

    with st.spinner('Loading your enrolled subjects…'):
        subjects = get_student_subjects(student_id)
        logs     = get_student_attendance(student_id)

    # Build per-subject attendance stats
    stats_map = {}
    for log in logs:
        sid = log['subject_id']
        if sid not in stats_map:
            stats_map[sid] = {"total": 0, "attended": 0}
        stats_map[sid]['total'] += 1
        if log.get('is_present'):
            stats_map[sid]['attended'] += 1

    if not subjects:
        st.info("You have not enrolled in any subjects yet. Use the button above to join one.")
    else:
        cols = st.columns(2)
        for i, sub_node in enumerate(subjects):
            sub = sub_node['subjects']
            sid = sub['subject_id']
            stats = stats_map.get(sid, {"total": 0, "attended": 0})

            def unenroll_button(sub=sub, sid=sid):
                if st.button(
                    "Unenroll from this subject",
                    type='tertiary',
                    width='stretch',
                    icon=':material/delete_forever:',
                    key=f"unenroll_{sid}"
                ):
                    unenroll_student_to_subject(student_id, sid)
                    st.toast(f"Unenrolled from {sub['name']} successfully!")
                    st.rerun()

            with cols[i % 2]:
                subject_card(
                    name=sub['name'],
                    code=sub['subject_code'],
                    section=sub['section'],
                    stats=[
                        ('📅', 'Total Sessions', stats['total']),
                        ('✅', 'Attended',        stats['attended']),
                    ],
                    footer_callback=unenroll_button
                )

    footer_dashboard()


def _re_enrollment_section(student_id: int):
    """Inline re-enrollment panel shown when a student's embedding is outdated."""
    with st.container(border=True):
        st.subheader("🔄 Re-enroll Your Face")
        st.write(
            "Take a clear photo to generate a new face profile.  "
            "Make sure only your face is visible and the image is well-lit."
        )
        photo = st.camera_input("Take a photo to re-enroll", key="reenroll_camera")
        if photo:
            img_np = np.array(Image.open(photo).convert('RGB'))
            if st.button("Save New Face Profile", type='primary', key="reenroll_save"):
                with st.spinner('Processing your face…'):
                    embedding, err = enroll_face_embedding(img_np)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        update_student_face_embedding(student_id, embedding)
                        # Refresh session data
                        all_students = get_all_students()
                        updated = next(
                            (s for s in all_students if s['student_id'] == student_id), None
                        )
                        if updated:
                            st.session_state.student_data = updated
                        st.success("✅ Face profile updated successfully! You are now ready for face recognition.")
                        time.sleep(1.5)
                        st.rerun()


# ---------------------------------------------------------------------------
# Student Screen (login + registration)
# ---------------------------------------------------------------------------

def student_screen():
    style_background_dashboard()
    style_base_layout()

    if "student_data" in st.session_state:
        student_dashboard()
        return

    # ── Header + back button ──────────────────────────────────────────────
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go Back to Home", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            st.rerun()

    st.header('Sign in with Face ID', text_alignment='center')
    st.space()
    st.space()

    show_registration = False

    photo_source = st.camera_input("Position your face clearly in the center")

    if photo_source:
        img_np = np.array(Image.open(photo_source).convert('RGB'))

        with st.spinner('Scanning your face…'):
            num_faces, match = predict_student_login(img_np, get_all_students(), RECOGNITION_THRESHOLD)

        if num_faces == 0:
            st.warning("⚠️ No face detected. Please ensure your face is clearly visible and well-lit.")
        elif num_faces > 1:
            st.warning(f"⚠️ {num_faces} faces detected. Only one face should be in frame for login.")
        else:
            if match:
                student_id = match['student_id']
                similarity = match['similarity']
                all_students = get_all_students()
                student = next((s for s in all_students if s['student_id'] == student_id), None)
                if student:
                    st.session_state.is_logged_in = True
                    st.session_state.user_role    = 'student'
                    st.session_state.student_data = student
                    st.toast(f"Welcome back, {student['name']}! (Similarity: {similarity:.2f})")
                    time.sleep(1)
                    st.rerun()
            else:
                st.info(
                    "Face not recognised. If you are a new student, please register below."
                )
                show_registration = True

    # ── Registration panel ────────────────────────────────────────────────
    if show_registration:
        with st.container(border=True):
            st.header('Register New Profile')
            new_name = st.text_input("Enter your full name", placeholder='E.g. Preety Jaiswal')

            st.subheader('Voice Enrollment (Optional)')
            st.info(
                "Record a short phrase like I am present to enable voice-based attendance."
                "to enable voice-based attendance."
            )

            audio_data = None
            try:
                audio_data = st.audio_input('Record a short phrase')
            except Exception:
                st.error('Audio recording is not available in this environment.')

            if st.button('Create Account', type='primary'):
                if not new_name:
                    st.warning('Please enter your name before creating an account.')
                else:
                    with st.spinner('Creating your profile…'):
                        img_np = np.array(Image.open(photo_source).convert('RGB'))
                        embedding, err = enroll_face_embedding(img_np)

                        if err:
                            st.error(f"❌ {err}")
                        else:
                            voice_emb = None
                            if audio_data:
                                voice_emb = get_voice_embedding(audio_data.read())

                            response_data = create_student(
                                new_name,
                                face_embedding=embedding,
                                voice_embedding=voice_emb,
                            )

                            if response_data:
                                st.session_state.is_logged_in = True
                                st.session_state.user_role    = 'student'
                                st.session_state.student_data = response_data[0]
                                st.toast(f"Profile created! Welcome, {new_name}!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error('Could not create your profile. Please try again.')

    footer_dashboard()