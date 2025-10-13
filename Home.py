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

st.session_state["LAMBDA_URL"] = "https://hc5ycktqbvxywpf4f4xhxfvm2e0dpozl.lambda-url.us-east-2.on.aws/"   # Function URL or API GW route               # EC2 FastAPI base

HEALTH_URL = st.session_state["API_URL"] + "healthz"
#---- session flags ----
st.session_state.setdefault("ec2_start_triggered", False)
st.session_state.setdefault("fastapi_ready", False)

def start_ec2_once():
    if st.session_state["ec2_start_triggered"]:
        return
    try:
        # fire-and-forget: short timeout, don't block
        requests.post(st.session_state["LAMBDA_URL"], json={}, timeout=10)
        st.session_state["ec2_start_triggered"] = True
    except Exception as e:
        st.warning(f"Could not call Lambda: {e}")

def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        st.session_state["fastapi_ready"] = r.ok
    except Exception:
        st.session_state["fastapi_ready"] = False

# 1) Kick off EC2 start the first time the user lands on the app
start_ec2_once()
check_health()
# 2) Poll readiness
if not st.session_state["fastapi_ready"]:
    st.info("Waking up the compute node… this usually takes 1–4 minutes.")
    st_autorefresh(interval=5000, key="preproc_refresh")
else:
    st.success("Compute node is ready.")

st.write("# Welcome to NMF exploration tool!")
st.markdown("""
Start by uploading data for preprocessing or uploading already preprocessed data.
Then explore different values of *k* and analyze the results.
Finally run NMF algorithm with a chosen value of *k* and download the results.
"""
)

st.page_link("pages/1_Metadata_Upload.py", label="Continue")