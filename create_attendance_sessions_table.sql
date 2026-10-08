-- ============================================================
-- FINALYEAR — CCTV Attendance Feature: Database Migration
-- ============================================================
-- Run this SQL in the Supabase SQL Editor BEFORE enabling CCTV.
--
-- Project URL: https://stalptqgticoasphanxo.supabase.co
--
-- Steps:
--   1. Go to: https://supabase.com/dashboard/project/stalptqgticoasphanxo
--   2. Left sidebar → SQL Editor
--   3. Paste this entire file and click Run
--
-- SAFETY GUARANTEES:
--   ✅  Uses IF NOT EXISTS — safe to run multiple times
--   ✅  No existing tables are modified
--   ✅  No existing data is touched
--   ✅  No columns dropped, no tables truncated
--   ✅  Purely additive migration
-- ============================================================

-- New table: one row per CCTV attendance session
-- Tracks the session lifecycle: running → normal_closed → completed
CREATE TABLE IF NOT EXISTS attendance_sessions (
    id                BIGSERIAL    PRIMARY KEY,

    -- Foreign keys to existing tables (ON DELETE SET NULL keeps session history
    -- even if a subject or teacher is later deleted)
    subject_id        BIGINT       REFERENCES subjects(subject_id)  ON DELETE SET NULL,
    teacher_id        BIGINT       REFERENCES teachers(teacher_id)  ON DELETE SET NULL,

    -- Lifecycle timestamps
    start_time        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    normal_close_time TIMESTAMPTZ,   -- NULL until teacher clicks Close Normal Attendance
    end_time          TIMESTAMPTZ,   -- NULL until teacher clicks Stop CCTV

    -- Session status
    -- Allowed values: 'running' | 'normal_closed' | 'completed' | 'cancelled'
    status            TEXT         NOT NULL DEFAULT 'running',

    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Index for quick look-ups by teacher (e.g. "all active sessions for this teacher")
CREATE INDEX IF NOT EXISTS idx_attendance_sessions_teacher
    ON attendance_sessions (teacher_id, status);

-- Index for quick look-ups by subject (e.g. weekly attendance aggregation)
CREATE INDEX IF NOT EXISTS idx_attendance_sessions_subject
    ON attendance_sessions (subject_id, created_at);

-- Enable Row Level Security to match the existing tables' setup
ALTER TABLE attendance_sessions ENABLE ROW LEVEL SECURITY;

-- Permissive policy so the Supabase service key can read/write
-- (matches the policy pattern used for late_attendance)
CREATE POLICY IF NOT EXISTS "Allow all for service key"
    ON attendance_sessions
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- ============================================================
-- Verification query (optional — run after migration to confirm)
-- ============================================================
-- SELECT table_name FROM information_schema.tables
-- WHERE table_schema = 'public'
-- ORDER BY table_name;
-- ============================================================
