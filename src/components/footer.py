import streamlit as st

def footer_home():

    st.markdown("""
        <div style="
            margin-top: 2rem;
            padding: 18px 0 10px;
            text-align: center;
            color: rgba(255,255,255,0.75);
            font-size: 13px;
        ">
            © 2026 &nbsp; • &nbsp;
            Intelligent Multimodal Classroom Attendance System
        </div>
    """, unsafe_allow_html=True)


def footer_dashboard():

    st.markdown("""
        <div style="
            margin-top: 2rem;
            padding: 18px 0 10px;
            text-align: center;
            color: rgba(0,0,0,0.60);
            font-size: 13px;
        ">
            © 2026 &nbsp; • &nbsp;
            Intelligent Multimodal Classroom Attendance System
        </div>
    """, unsafe_allow_html=True)