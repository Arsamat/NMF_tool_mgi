import streamlit as st
import pandas as pd
from ui_theme import apply_custom_theme
import requests 
from streamlit_autorefresh import st_autorefresh


def run_metadata_upload():
    apply_custom_theme()
    if "meta" not in st.session_state:
        st.session_state["meta"] = None
    if "metadata_index" not in st.session_state:
        st.session_state["metadata_index"] = None
    if "design_factor" not in st.session_state:
        st.session_state["design_factor"] = None

    st.session_state["LAMBDA_URL"] = "https://hc5ycktqbvxywpf4f4xhxfvm2e0dpozl.lambda-url.us-east-2.on.aws/"   # Function URL or API GW route               # EC2 FastAPI base

    HEALTH_URL = st.session_state["API_URL"] + "healthz"
    #---- session flags ----
    st.session_state.setdefault("ec2_start_triggered", False)
    st.session_state.setdefault("fastapi_ready", False)

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
                st.session_state["fastapi_ready"] = r.ok
            except Exception:
                st.session_state["fastapi_ready"] = False

    if st.session_state["meta"] is None:
        tmp_file = st.file_uploader("Upload metadata", type=["csv", "tsv", "txt"])
        if st.button("Click to Finish Upload") and tmp_file is not None:
            sep = "," if tmp_file.name.endswith(".csv") else "\t"
            meta = pd.read_csv(tmp_file, sep=sep)
            st.session_state["meta"] = meta

        def start_ec2_once():
                if st.session_state["ec2_start_triggered"]:
                    return
                try:
                    # fire-and-forget: short timeout, don't block
                    requests.post(st.session_state["LAMBDA_URL"], json={}, timeout=10)
                    st.session_state["ec2_start_triggered"] = True
                except Exception as e:
                    st.warning(f"Could not call Lambda: {e}")

        # 1) Kick off EC2 start the first time the user lands on the app
        if not st.session_state["ec2_start_triggered"]:
            start_ec2_once()

    if st.session_state["meta"] is not None and not st.session_state["fastapi_ready"]:
        check_health()
        # 2) Poll readiness
        if not st.session_state["fastapi_ready"]:
            st.info("Waking up the compute node… this usually takes 1–4 minutes. Please, wait till it is ready to proceed.")
            st_autorefresh(interval=8000, key="preproc_refresh")
        else:
            st.success("Compute node is ready.")

    if st.session_state["meta"] is not None:
        st.subheader("Metadata Uploaded")

        if st.button("Remove Metadata"):
            st.session_state["meta"] = None
            st.rerun()
            
        st.dataframe(st.session_state["meta"])
        st.write("Please, provide the name of the column that stores sample names before proceeding. Use exactly the same name as in the file")
        sample_column = st.selectbox(
                "Select the column that contains sample names:",
                options=st.session_state["meta"].columns.tolist(),
                index=0
            )
        st.session_state["design_factor"] = st.selectbox(
                "Column name storing design group data",
                options=st.session_state["meta"].columns.tolist(),
                index=1
            )
            
        if st.button("Save"):
            st.session_state["metadata_index"] = sample_column
    
    if st.session_state["metadata_index"] is not None:
        if st.button("Continue"):
            st.session_state["_go_to"] = "Preprocess Data for NMF"
            st.rerun()

