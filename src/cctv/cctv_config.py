"""
FusionPresence — CCTV Configuration
=====================================
Reads CCTV-specific settings from Streamlit secrets.
Falls back to safe defaults if any key is absent.

Supported secrets (all optional):
    CCTV_SOURCE                Camera source: "0" = webcam, or RTSP URL
    CCTV_FACE_THRESHOLD        Cosine-similarity threshold (CCTV only)
    CCTV_CONFIRMATION_FRAMES   Min frames to confirm a recognition
    CCTV_CONFIRMATION_WINDOW   Sliding-window duration in seconds
    CCTV_PROCESS_EVERY_N_FRAMES  Process every Nth frame (reduces CPU)

If CCTV_SOURCE is absent the feature shows a "not configured" message and
every other existing feature continues to work normally.
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Built-in defaults (used when a secret key is absent)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "CCTV_FACE_THRESHOLD":        0.60,
    "CCTV_CONFIRMATION_FRAMES":   5,
    "CCTV_CONFIRMATION_WINDOW":   10.0,   # seconds
    "CCTV_PROCESS_EVERY_N_FRAMES": 3,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_cctv_configured() -> bool:
    """Return True only when CCTV_SOURCE has been set in secrets.toml."""
    return st.secrets.get("CCTV_SOURCE", None) is not None


def get_cctv_config() -> dict:
    """
    Read all CCTV settings from Streamlit secrets.

    Returns
    -------
    dict with keys:
        source                 int (0) or str (RTSP URL) or None
        face_threshold         float
        confirmation_frames    int
        confirmation_window    float  (seconds)
        process_every_n_frames int
    """
    config: dict = {}

    # ── Camera source ─────────────────────────────────────────────────────
    raw_source = st.secrets.get("CCTV_SOURCE", None)
    if raw_source is not None:
        s = str(raw_source).strip()
        try:
            config["source"] = int(s)        # "0" → 0 (webcam index)
        except ValueError:
            config["source"] = s             # keep as RTSP URL string
    else:
        config["source"] = None              # not configured

    # ── Numeric settings ──────────────────────────────────────────────────
    config["face_threshold"] = _safe_float(
        st.secrets.get("CCTV_FACE_THRESHOLD"),
        _DEFAULTS["CCTV_FACE_THRESHOLD"],
    )
    config["confirmation_frames"] = _safe_int(
        st.secrets.get("CCTV_CONFIRMATION_FRAMES"),
        _DEFAULTS["CCTV_CONFIRMATION_FRAMES"],
    )
    config["confirmation_window"] = _safe_float(
        st.secrets.get("CCTV_CONFIRMATION_WINDOW"),
        _DEFAULTS["CCTV_CONFIRMATION_WINDOW"],
    )
    config["process_every_n_frames"] = _safe_int(
        st.secrets.get("CCTV_PROCESS_EVERY_N_FRAMES"),
        _DEFAULTS["CCTV_PROCESS_EVERY_N_FRAMES"],
    )

    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
