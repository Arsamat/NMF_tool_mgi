# ui_theme.py
import streamlit as st

def apply_custom_theme():
    st.set_page_config(layout="wide")

    st.markdown("""
    <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #9D00FF;
        }

        /* Sidebar text (labels, titles, etc.) */
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }

        /* Main area text */
        .block-container {
            color: #000000;
        }

        /* Sidebar buttons */
        [data-testid="stSidebar"] button {
            background-color: #DE811D;
            color: white !important;
        }
        [data-testid="stSidebar"] button:hover {
            background-color: #6BFF00;
            color: black !important;
        }
    </style>
    """, unsafe_allow_html=True)
