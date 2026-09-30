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


         </style>
    """, unsafe_allow_html=True)




def style_background_dashboard():

    st.markdown(""" 
         <style>

            .stApp{
              background:#ECE9FF !important;

            }


         </style>



        

    
    
    """
    ,unsafe_allow_html=True)




def style_base_layout():

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
