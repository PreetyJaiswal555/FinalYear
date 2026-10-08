import streamlit as st
import base64
import os


def _get_bg_b64() -> str:
    """Read assets/backgroundcolour.png and return a CSS-ready base64 data URI."""
    bg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "assets", "backgroundcolour.png"
    )
    bg_path = os.path.normpath(bg_path)
    with open(bg_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def style_background_home():

    bg_src = _get_bg_b64()

    st.markdown(f""" 
         <style>

            .stApp{{
              background: url("{bg_src}") no-repeat center center fixed !important;
              background-size: cover !important;
            }}
            .stApp div[data-testid="stColumn"]{{
                    background-color:#E0E3FF !important;
                    padding:2.5rem !important;
                    border-radius: 5rem !important;
                    }}

            /* Home page column headings — dark so visible on #E0E3FF */
            .stApp div[data-testid="stColumn"] h1,
            .stApp div[data-testid="stColumn"] h2,
            .stApp div[data-testid="stColumn"] h3 {{
                    color: #333333 !important;
                    }}


         </style>
    """, unsafe_allow_html=True)




def style_background_dashboard():
    """
    Applied only in teacher_screen and student_screen (never on the home page).
    Includes all dark-text fixes scoped to the dashboard background.
    """
    st.markdown(""" 
         <style>

            .stApp{
              background:#ECE9FF !important;
            }

            /* Headings — dark navy on dashboard only */
            h1 { color:#1a1a4e !important; }
            h2 { color:#1a1a4e !important; }
            h3, h4 { color:#1e293b !important; }

            /* ═══════════════════════════════════════════════════════════════
               DASHBOARD TEXT VISIBILITY
               Dark text on the light purple (#ECE9FF) dashboard background.
               Buttons are explicitly kept white at the end.
               Home page is NOT affected — this block is dashboard-only.
               ═══════════════════════════════════════════════════════════════ */

            /* Paragraph, caption, small, markdown content */
            p, small, .stCaption,
            [data-testid="stCaptionContainer"] p,
            div[data-testid="stMarkdownContainer"] p,
            div[data-testid="stMarkdownContainer"] li,
            div[data-testid="stMarkdownContainer"] strong,
            div[data-testid="stMarkdownContainer"] em,
            div[data-testid="stMarkdownContainer"] span {
            color:#1e293b !important;
            }

            /* All widget / input labels (Username, Password, Select Subject…) */
            label, .stTextInput label, .stSelectbox label,
            .stTextArea label, .stNumberInput label,
            .stFileUploader label, .stCameraInput label,
            .stAudioInput label,
            [data-testid="stWidgetLabel"],
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] label,
            [data-testid="stWidgetLabel"] span {
            color:#1e293b !important;
            }

            /* Metric labels, values, delta */
            [data-testid="stMetricLabel"],
            [data-testid="stMetricLabel"] p,
            [data-testid="stMetricLabel"] div,
            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] *,
            [data-testid="stMetricDelta"],
            [data-testid="stMetricDelta"] * {
            color:#1e293b !important;
            }

            /* Bold / strong text */
            strong, b { color:#1e293b !important; }

            /* Selectbox selected value text */
            [data-testid="stSelectbox"] span,
            [data-testid="stSelectbox"] div[class*="singleValue"] {
            color:#1e293b !important;
            }

            /* ── Keep button text WHITE — always last, highest priority ── */
            button, button p, button span, button div,
            button strong, button em, button small,
            [data-testid="stBaseButton-primary"],
            [data-testid="stBaseButton-primary"] *,
            [data-testid="stBaseButton-secondary"],
            [data-testid="stBaseButton-secondary"] *,
            [data-testid="stBaseButton-tertiary"],
            [data-testid="stBaseButton-tertiary"] * {
            color:white !important;
            }

            /* Stronger override using Streamlit's rendered kind attribute */
            .stApp button[kind="primary"],
            .stApp button[kind="primary"] *,
            .stApp button[kind="secondary"],
            .stApp button[kind="secondary"] *,
            .stApp button[kind="tertiary"],
            .stApp button[kind="tertiary"] * {
            color: white !important;
            opacity: 1 !important;
            }

            /* Shortcut badges inside dashboard buttons */
            .stApp button[kind="primary"] small,
            .stApp button[kind="primary"] kbd,
            .stApp button[kind="primary"] span,
            .stApp button[kind="secondary"] small,
            .stApp button[kind="secondary"] kbd,
            .stApp button[kind="secondary"] span,
            .stApp button[kind="tertiary"] small,
            .stApp button[kind="tertiary"] kbd,
            .stApp button[kind="tertiary"] span {
            color: white !important;
            opacity: 1 !important;
            }

            /* ── Popups / Dialogs — restore white text (dark background) ── */
            [data-testid="stModal"] p,
            [data-testid="stModal"] span,
            [data-testid="stModal"] label,
            [data-testid="stModal"] small,
            [data-testid="stModal"] strong,
            [data-testid="stModal"] b,
            [data-testid="stModal"] li,
            [data-testid="stModal"] [data-testid="stWidgetLabel"],
            [data-testid="stModal"] [data-testid="stWidgetLabel"] p,
            [data-testid="stModal"] [data-testid="stWidgetLabel"] span,
            [data-baseweb="modal"] p,
            [data-baseweb="modal"] span,
            [data-baseweb="modal"] label,
            [data-baseweb="modal"] small,
            [data-baseweb="modal"] strong,
            [role="dialog"] p,
            [role="dialog"] span,
            [role="dialog"] label,
            [role="dialog"] small,
            [role="dialog"] strong,
            [role="dialog"] b,
            [role="dialog"] [data-testid="stWidgetLabel"] p {
            color:white !important;
            }

            /* ── Buttons inside dialogs — also white ── */
            [data-testid="stModal"] button,
            [data-testid="stModal"] button *,
            [role="dialog"] button,
            [role="dialog"] button * {
            color:white !important;
            }


         </style>


        


    
    """
    ,unsafe_allow_html=True)




def style_base_layout():
    """
    Shared layout — called by ALL screens including home page.
    Contains only: fonts, hide Streamlit chrome, button styles.
    NO text-color rules here (those are in style_background_dashboard only).
    """
    st.markdown(""" 
         <style>
            @import url('https://fonts.googleapis.com/css2?family=Climate+Crisis:YEAR@1979&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap');
            

            /*hide top bar of streamlit */

           #MainMenu,footer,header{
               visibility:hidden;
            } 

            .block-container{
               padding-top:1.5rem !important
               
               }

            h1{
            font-family:'Climate Crisis',sans-serif !important;
            font-size:3.5rem !important;
            line-height:1.1 !important;
            margin-bottom:0rem !important;

            } 
            h2{
            font-family:'Climate Crisis',sans-serif !important;
            font-size:2rem !important;
            line-height:0.9 !important;
            margin-bottom:0rem !important;

            } 

            h3,h4,p{
            font-family:'Outfit',sans-serif;


              
            }




             button{
               border-radius:1.5rem !important;
               background-color:#5865F2 !important;
               color:white !important;
               padding:10px 20px !important;
               border:none !important;
               transition:transform 0.25s ease-in-out !important;


            
            }
            button[kind="secondary"]{
               border-radius:1.5rem !important;
               background-color:#EB459E !important;
               color:white !important;
               padding:10px 20px !important;
               border:none !important;
               transition:transform 0.25s ease-in-out !important;


            
            }

             button[kind="tertiary"]{
               border-radius:1.5rem !important;
               background-color:black !important;
               color:white !important;
               padding:10px 20px !important;
               border:none !important;
               transition:transform 0.25s ease-in-out !important;


            
            }
            button:hover{
            transform:scale(1.05)
            }



         </style>



        


    
    """
    ,unsafe_allow_html=True)
