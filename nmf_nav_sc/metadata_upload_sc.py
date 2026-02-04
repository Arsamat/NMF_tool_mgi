import streamlit as st
import pandas as pd
from ui_theme import apply_custom_theme
import requests 
from streamlit_autorefresh import st_autorefresh


def run_metadata_upload_sc():
    apply_custom_theme()

    if "meta_sc" not in st.session_state:
        st.session_state["meta_sc"] = None
    if "metadata_index_sc" not in st.session_state:
        st.session_state["metadata_index_sc"] = None
    if "design_factor_sc" not in st.session_state:
        st.session_state["design_factor_sc"] = None

    st.session_state["LAMBDA_URL_sc"] = "https://hc5ycktqbvxywpf4f4xhxfvm2e0dpozl.lambda-url.us-east-2.on.aws/"
    HEALTH_URL = st.session_state["API_URL_sc"] + "healthz"

    # ---- session flags ----
    st.session_state.setdefault("ec2_start_triggered_sc", False)
    st.session_state.setdefault("fastapi_ready_sc", False)

    st.subheader("Start by uploading your metadata file. You won't be able to use further features without it.")

    st.markdown('''
            Your metadata file should at minimum contain a column with **sample names** and a column indicating **sample groups**. Additional metadata variables can be overlayed on the usage score heatmap to help with visualization and interpretation.

    Samples in the module usage heatmaps will be ordered according to this metadata file. Order samples in this file in a way that reflects your experimental design, to help with visualization and interpretation of results.

    You can come back and upload a different file if you want to use a different data set later.
                ''')

    if st.button("View Metadata File Example"):
        example_data = {
            "SampleName": ["Sample1", "Sample2", 'Sample3'],
            "Group": ["Group1", "Group2", "Group3"]
        }
        df = pd.DataFrame(example_data)
        st.dataframe(df)

    def check_health():
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            st.session_state["fastapi_ready_sc"] = r.ok
        except Exception:
            st.session_state["fastapi_ready_sc"] = False

    if st.session_state["meta_sc"] is None:
        tmp_file = st.file_uploader("Upload metadata", type=["csv", "tsv", "txt"])
        if st.button("Click to Finish Upload") and tmp_file is not None:
            sep = "," if tmp_file.name.endswith(".csv") else "\t"
            meta = pd.read_csv(tmp_file, sep=sep)
            st.session_state["meta_sc"] = meta

        def start_ec2_once():
            if st.session_state["ec2_start_triggered_sc"]:
                return
            try:
                requests.post(st.session_state["LAMBDA_URL_sc"], json={}, timeout=10)
                st.session_state["ec2_start_triggered_sc"] = True
            except Exception as e:
                st.warning(f"Could not call Lambda: {e}")

        if not st.session_state["ec2_start_triggered_sc"]:
            start_ec2_once()

    if st.session_state["meta_sc"] is not None and not st.session_state["fastapi_ready_sc"]:
        check_health()
        if not st.session_state["fastapi_ready_sc"]:
            st.info("Waking up the compute node… this usually takes 1–4 minutes. Please, wait till it is ready to proceed.")
            st_autorefresh(interval=8000, key="preproc_refresh_sc")
        else:
            st.success("Compute node is ready.")

    if st.session_state["meta_sc"] is not None:
        st.subheader("Metadata Uploaded")

        if st.button("Remove Metadata"):
            st.session_state["meta_sc"] = None
            st.rerun()

        st.dataframe(st.session_state["meta_sc"])
        st.write("Please, provide the name of the column that stores sample names before proceeding. Use exactly the same name as in the file")

        sample_column = st.selectbox(
            "Select the column that contains sample names:",
            options=st.session_state["meta_sc"].columns.tolist(),
            index=0
        )

        st.session_state["design_factor_sc"] = st.selectbox(
            "Column name storing design group data",
            options=st.session_state["meta_sc"].columns.tolist(),
            index=1
        )

        if st.button("Save"):
            st.session_state["metadata_index_sc"] = sample_column

    if st.session_state["metadata_index_sc"] is not None:
        if st.button("Continue"):
            st.session_state["_go_to_sc"] = "Preprocess Data for NMF"
            st.rerun()


