import streamlit as st
import requests
from streamlit_autorefresh import st_autorefresh

from ui_theme import apply_custom_theme
apply_custom_theme()



# --- Session state ---
st.session_state.setdefault("preprocessed_feather", None)
st.session_state.setdefault("results", {})
st.session_state.setdefault("spearman_img", None)
st.session_state.setdefault("gene_column", None)
st.session_state.setdefault("hvg", None)
st.session_state.setdefault("metadata_index", None)
st.session_state.setdefault("design_factor", None)
st.session_state.setdefault("gene_symbols", None)
st.session_state.setdefault("k_range", [])
st.session_state.setdefault("max_iter", 0)
st.session_state.setdefault("heatmap_pdf_bytes", [])
st.session_state.setdefault("counts", None)
st.session_state.setdefault("meta", None)
# For Spearman plots
st.session_state.setdefault("spearman_boxplot_png", None)
st.session_state.setdefault("spearman_max_png", None)
st.session_state.setdefault("pairs_by_k", {})

st.session_state.setdefault("API_URL", "")
st.session_state.setdefault("LAMBDA_URL", "")
st.session_state.setdefault("HEALTH_URL", "")


#st.session_state["API_URL"] = "http://3.141.231.76:8000/"
st.session_state["API_URL"] = "http://52.14.223.10:8000/"

st.write("# Welcome to NMF exploration tool!")
st.markdown("""
Start by uploading data for preprocessing or uploading already preprocessed data.
Then explore different values of *k* and analyze the results.
Finally run NMF algorithm with a chosen value of *k* and download the results.
"""
)

st.page_link("pages/1_Metadata_Upload.py", label="Continue")