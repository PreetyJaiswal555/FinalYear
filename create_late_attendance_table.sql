-- ============================================================
-- FINALYEAR — Latecomer Attendance Feature: DB Setup
-- ============================================================
-- Run this SQL in the Supabase SQL Editor for project:
--   https://stalptqgticoasphanxo.supabase.co
--
-- Steps:
--   1. Go to: https://supabase.com/dashboard/project/stalptqgticoasphanxo
--   2. Left sidebar → SQL Editor
--   3. Paste this entire file and click Run
-- ============================================================

CREATE TABLE IF NOT EXISTS late_attendance (
    id          BIGSERIAL    PRIMARY KEY,
    subject_id  BIGINT       REFERENCES subjects(subject_id)  ON DELETE CASCADE,
    student_id  BIGINT       REFERENCES students(student_id)  ON DELETE CASCADE,
    timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    face_score  FLOAT,
    voice_score FLOAT,
    liveness_passed BOOLEAN  DEFAULT FALSE,
    status      TEXT         DEFAULT 'late'
);

-- Index for fast per-subject-per-student-per-day queries
CREATE INDEX IF NOT EXISTS idx_late_attendance_student_subject
    ON late_attendance (student_id, subject_id, timestamp);

-- Optional: enable Row Level Security (RLS) with a permissive policy
-- so the service key can read/write (matches existing tables' setup)
ALTER TABLE late_attendance ENABLE ROW LEVEL SECURITY;

CREATE POLICY IF NOT EXISTS "Allow all for service key"
    ON late_attendance
    FOR ALL
    USING (true)
    WITH CHECK (true);
