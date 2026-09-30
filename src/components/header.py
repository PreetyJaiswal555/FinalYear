import streamlit as st
import base64
import os


def _get_logo_b64() -> str:
    """Read assets/finallogo.png and return a base64 data URI for use in HTML."""
    logo_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "finallogo.png")
    logo_path = os.path.normpath(logo_path)
    with open(logo_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def header_home():
    logo_src = _get_logo_b64()

    st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; margin-bottom:30px; margin-top:30px">
            <img src='{logo_src}' style='height:100px;' />
            <h1 style='text-align:center; line-height:1.1;'>
                <span style='color:#1a1a4e;'>FUSION</span><br/>
                <span style='color:#5865F2;'>PRESENCE</span>
            </h1>
        </div>   
                
                """, unsafe_allow_html=True)


def header_dashboard():
    logo_src = _get_logo_b64()

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&display=swap');
        .fp-brand {{ line-height: 1.0; }}
        .fp-fusion {{
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 800 !important;
            font-size: 2.8rem !important;
            color: #11164F !important;
            letter-spacing: -1px !important;
            text-transform: uppercase !important;
            -webkit-text-stroke: 0.6px #11164F !important;
        }}
        .fp-presence {{
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 700 !important;
            font-size: 2.1rem !important;
            color: #5865F2 !important;
            letter-spacing: 0px !important;
            text-transform: uppercase !important;
        }}
        </style>
        <div style="display:flex; align-items:center; gap:14px; padding:4px 0;">
            <img src='{logo_src}' style='height:72px; flex-shrink:0;' />
            <div class="fp-brand">
                <div class="fp-fusion">FUSION</div>
                <div class="fp-presence">PRESENCE</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
