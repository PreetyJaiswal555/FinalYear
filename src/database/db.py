from src.database.config import supabase
import bcrypt




def hash_pass(pwd):
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()

def check_pass(pwd, hashed):
    return bcrypt.checkpw(pwd.encode(), hashed.encode())




def check_teacher_exists(username):
    # Check for unique username, returns false when username is already taken
    response = supabase.table("teachers").select("username").eq("username", username).execute()
    return len(response.data) > 0 



def create_teacher(username, password, name):

    data = { "username" : username, "password": hash_pass(password), "name": name}
    response = supabase.table("teachers").insert(data).execute()
    return response.data   




def teacher_login(username, password):
    response = supabase.table("teachers").select("*").eq("username", username).execute()
    if response.data:
        teacher = response.data[0]
        if check_pass(password, teacher['password']):
            return teacher
    return None


def get_all_students():
    response = supabase.table('students').select("*").execute()
    return response.data  



def create_student(new_name, face_embedding=None, voice_embedding=None):
    data = {'name': new_name, 'face_embedding':face_embedding, "voice_embedding": voice_embedding}
    response = supabase.table('students').insert(data).execute()
    return response.data  


def create_subject(subject_code, name, section, teacher_id):
    data = {"subject_code": subject_code, "name": name, "section": section, "teacher_id": teacher_id}
    response = supabase.table("subjects").insert(data).execute()
    return response.data   


def get_teacher_subjects(teacher_id):
    response = supabase.table('subjects').select("*, subject_students(count), attendance_logs(timestamp)").eq("teacher_id", teacher_id).execute()
    subjects = response.data


    for sub in subjects:
        sub['total_students'] = sub.get("subject_students", [{}])[0].get('count', 0) if sub.get('subject_students') else 0
        attendance = sub.get('attendance_logs', [])
        unique_sessions = len(set(log['timestamp'] for log in attendance))
        sub['total_classes'] = unique_sessions


        sub.pop('subject_student', None)
        sub.pop('attendance_logs', None)

    return subjects


def  enroll_student_to_subject(student_id, subject_id):
    data = {'student_id': student_id, "subject_id": subject_id}
    response= supabase.table('subject_students').insert(data).execute()
    return response.data


def  unenroll_student_to_subject(student_id, subject_id):
    response= supabase.table('subject_students').delete().eq('student_id', student_id).eq('subject_id', subject_id).execute()
    return response.data



def get_student_subjects(student_id):
    response = supabase.table('subject_students').select('*, subjects(*)').eq('student_id', student_id).execute()
    return response.data


def get_student_attendance(student_id):
    response = supabase.table('attendance_logs').select('*, subjects(*)').eq('student_id', student_id).execute()
    return response.data


def create_attendance(logs):
    response = supabase.table('attendance_logs').insert(logs).execute()
    return response.data

def get_attendance_for_teacher(teacher_id):
    response = supabase.table('attendance_logs').select("*, subjects!inner(*)").eq('subjects.teacher_id', teacher_id).execute()
    return response.data


# ---------------------------------------------------------------------------
# New functions for ArcFace pipeline and weekly attendance
# ---------------------------------------------------------------------------

def get_subject_students_with_embeddings(subject_id):
    """
    Return all students enrolled in *subject_id* together with their
    face_embedding and voice_embedding.

    Used by face attendance to restrict candidate matching to only those
    students who are enrolled in the selected subject.

    Returns
    -------
    list of dict
        Each dict: { student_id, name, face_embedding, voice_embedding }
    """
    response = (
        supabase.table('subject_students')
        .select('*, students(student_id, name, face_embedding, voice_embedding)')
        .eq('subject_id', subject_id)
        .execute()
    )
    students = []
    for node in (response.data or []):
        s = node.get('students')
        if s:
            students.append(s)
    return students


def update_student_face_embedding(student_id, embedding_list):
    """
    Overwrite the face_embedding for *student_id* with a new ArcFace
    512-D embedding provided as a Python list.

    Parameters
    ----------
    student_id : int
    embedding_list : list[float]  (512-D ArcFace embedding)

    Returns
    -------
    list  Supabase response data
    """
    response = (
        supabase.table('students')
        .update({'face_embedding': embedding_list})
        .eq('student_id', student_id)
        .execute()
    )
    return response.data


def update_student_voice_embedding(student_id: int, embedding_list: list):
    """
    Overwrite the voice_embedding for *student_id* with a new Resemblyzer
    embedding provided as a Python list.

    Parameters
    ----------
    student_id : int
    embedding_list : list[float]

    Returns
    -------
    list  Supabase response data
    """
    response = (
        supabase.table('students')
        .update({'voice_embedding': embedding_list})
        .eq('student_id', student_id)
        .execute()
    )
    return response.data


# ---------------------------------------------------------------------------
# Latecomer Attendance Functions
# ---------------------------------------------------------------------------

def check_student_present_today(student_id: int, subject_id: int) -> bool:
    """
    Return True if the student is already marked present in attendance_logs
    for the given subject_id today (any time on the current calendar date).
    """
    from datetime import date
    today_str = date.today().isoformat()
    response = (
        supabase.table('attendance_logs')
        .select('id, is_present')
        .eq('student_id', student_id)
        .eq('subject_id', subject_id)
        .gte('timestamp', f'{today_str}T00:00:00')
        .lte('timestamp', f'{today_str}T23:59:59')
        .execute()
    )
    rows = response.data or []
    return any(bool(r.get('is_present')) for r in rows)


def check_existing_late_record(student_id: int, subject_id: int) -> bool:
    """
    Return True if a late_attendance record already exists for this student
    and subject today (prevents duplicate late records).
    """
    from datetime import date
    today_str = date.today().isoformat()
    response = (
        supabase.table('late_attendance')
        .select('id')
        .eq('student_id', student_id)
        .eq('subject_id', subject_id)
        .gte('timestamp', f'{today_str}T00:00:00')
        .lte('timestamp', f'{today_str}T23:59:59')
        .execute()
    )
    return len(response.data or []) > 0


def insert_late_attendance(student_id: int, subject_id: int,
                           face_score: float = None, voice_score: float = None,
                           liveness_passed: bool = False) -> list:
    """
    Insert a single row into late_attendance and mark the student as present
    in attendance_logs for today (if not already present).

    Returns
    -------
    list  Supabase insert response data for the late_attendance row.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 1. Insert late record
    late_data = {
        'student_id': student_id,
        'subject_id': subject_id,
        'timestamp': ts,
        'face_score': face_score,
        'voice_score': voice_score,
        'liveness_passed': liveness_passed,
        'status': 'late',
    }
    late_resp = supabase.table('late_attendance').insert(late_data).execute()

    # 2. Also mark student as present in the main attendance_logs
    # so that existing attendance percentage calculations count this student
    attendance_data = {
        'student_id': student_id,
        'subject_id': subject_id,
        'timestamp': ts,
        'is_present': True,
    }
    supabase.table('attendance_logs').insert(attendance_data).execute()

    return late_resp.data


def get_late_attendance_for_teacher(teacher_id: int) -> list:
    """
    Return all late_attendance rows for subjects that belong to *teacher_id*,
    joined with student name and subject info.

    Returns
    -------
    list of dict
    """
    response = (
        supabase.table('late_attendance')
        .select('*, students(name), subjects!inner(name, subject_code, teacher_id)')
        .eq('subjects.teacher_id', teacher_id)
        .order('timestamp', desc=True)
        .execute()
    )
    return response.data or []


def get_attendance_for_subject(subject_id):
    """
    Return all attendance_logs rows for *subject_id* together with the
    student name.  Used for the weekly attendance aggregation view.

    Returns
    -------
    list of dict
        Each dict: { id, student_id, subject_id, timestamp, is_present,
                     students: { name } }
    """
    response = (
        supabase.table('attendance_logs')
        .select('*, students(name)')
        .eq('subject_id', subject_id)
        .execute()
    )
    return response.data or []
