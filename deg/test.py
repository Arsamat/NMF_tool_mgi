import streamlit as st
import requests
import pandas as pd
import io
from streamlit_autorefresh import st_autorefresh
from brb_data_pages.visualize_data import visualize_metadata


def deg_analysis():
    if "counts_tmp" not in st.session_state:
        st.session_state["counts_tmp"] = None
    if "metadata_tmp" not in st.session_state:
        st.session_state["metadata_tmp"] = None
    if "job_id_tmp" not in st.session_state:
        st.session_state["job_id_tmp"] = None
    if "saved_for_nmf" not in st.session_state:
        st.session_state["saved_for_nmf"] = False
    if "schema" not in st.session_state:
        st.session_state["schema"] = None

    #FASTAPI_URL = "http://18.218.84.81:8000/"
    FASTAPI_URL = "http://3.141.231.76:8000/"
    #st.session_state["LAMBDA_URL"] = "https://hc5ycktqbvxywpf4f4xhxfvm2e0dpozl.lambda-url.us-east-2.on.aws/"   # Function URL or API GW route               # EC2 FastAPI base
    st.session_state["LAMBDA_URL"] = "xxx"
    HEALTH_URL = FASTAPI_URL + "healthz"
    #---- session flags ----
    st.session_state.setdefault("ec2_start_triggered", False)
    st.session_state.setdefault("fastapi_ready", False)
    st.session_state.setdefault("visualize_data", False)

    def check_health():
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            st.session_state["fastapi_ready"] = r.ok
        except Exception:
            st.session_state["fastapi_ready"] = False
    

    def start_ec2_once():
            if st.session_state["ec2_start_triggered"]:
                return
            try:
                # fire-and-forget: short timeout, don't block
                requests.post(st.session_state["LAMBDA_URL"], json={}, timeout=10)
                st.session_state["ec2_start_triggered"] = True
            except Exception as e:
                st.warning(f"Could not make a call to the server: {e}")

    if not st.session_state["ec2_start_triggered"]:
        start_ec2_once()

    if not st.session_state["fastapi_ready"]:
        check_health()
        st.info("Waking up the compute node… this usually takes 1–4 minutes. Please, wait till it is ready to proceed.")
        st_autorefresh(interval=8000, key="preproc_refresh")
    else:
        st.success("Compute node is ready.")

    if st.session_state["fastapi_ready"]:
        # Step 1 — Load metadata schema from backend
        st.subheader("Step 1: Load Metadata Schema")
        if st.button("Load Schema"):
            st.session_state["schema"] = requests.get(f"{FASTAPI_URL}/get_metadata/").json()            
        
        if st.session_state["schema"] is not None: 
            schema = st.session_state["schema"]
            columns = schema["columns"]
            unique_vals = schema["unique_values"]
            filterable_columns = [c for c in columns if c != "SampleName"]

            st.title("🔍 Metadata Filtering UI")

            # Step 2 — Select metadata columns
            selected_cols = st.multiselect(
                "Choose metadata columns:",
                filterable_columns
            )

            filters = {}

            # Step 3 — For each selected column, choose unique values
            for col in selected_cols:
                st.markdown(f"**Select values for `{col}`:**")
                vals = st.multiselect(
                    f"{col}:",
                    options=unique_vals[col],
                    key=f"selector_widget_{col}"
                )

                if vals:
                    filters[col] = vals
            
            # Step 5 — Query backend for sample names
            if st.button("Find Matching Samples"):
                resp = requests.post(
                    f"{FASTAPI_URL}/get_samples",
                    json={"filters": filters}
                )
                if resp.status_code == 204:
                    st.warning("No matching samples were found.")
                else:
                    buf = io.BytesIO(resp.content)
                    st.session_state["metadata_tmp"] = pd.read_feather(buf)
                    
            if st.session_state["metadata_tmp"] is not None:

                @st.cache_data
                def convert_for_download(df):
                    return df.to_csv().encode("utf-8")

                st.subheader("Short version Metadata Table Preview:")
                st.dataframe(st.session_state["metadata_tmp"].head())

                meta_csv = convert_for_download(st.session_state["metadata_tmp"])

                st.download_button(
                    label="Download Full Metadata CSV",
                    data=meta_csv,
                    file_name="metadata_table_brb.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
                if st.button("Visualize Data"):
                    st.session_state["visualize_data"] = True

                if st.session_state["visualize_data"]:
                    if st.button("Close"):
                        st.session_state["visualize_data"] = False
                        st_autorefresh()
                    visualize_metadata()
                
                deg_code()

deg_analysis()

