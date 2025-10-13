# ui_theme.py
import streamlit as st

st.set_page_config(page_title="NMF Tool", layout="wide")

def apply_custom_theme():
    st.set_page_config(
        page_title="NMF Exploration Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
