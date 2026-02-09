import streamlit as st
import requests
import pandas as pd
import io
import zipfile
import json
import time
from streamlit_autorefresh import st_autorefresh

def authenticate():
    # placeholders variables for UI 
    title_placeholder = st.empty()
    help_placeholder = st.empty()
    password_input_placeholder = st.empty()
    button_placeholder = st.empty()
    success_placeholder = st.empty()
    
    # check if not authenticated 
    if not st.session_state['authenticated']:
        # UI for authentication
        with title_placeholder:
            st.title("If you are part of MGI you can access BRB-seq data for further analysis")
        with help_placeholder:
            with st.expander("**⚠️ Read if You Need Help With Password**"):
                st.write("To request or get an updated password contact developers.")
            
                st.write("**Azamat Khanbabaev** azamat@wustl.edu")
                st.write("**Aura Ferreiro** alferreiro@wustl.edu")
            # UI and get get user password
            with password_input_placeholder:
                user_password = st.text_input("Enter the application password:", type="password", key="pwd_input")
            check_password = True if user_password == st.secrets["PASSWORD"] else False
            # Check user password and correct password
            with button_placeholder:
                if st.button("Authenticate") or user_password:
                    # If password is correct
                    if check_password:
                        st.session_state['authenticated'] = True
                        password_input_placeholder.empty()
                        button_placeholder.empty()
                        success_placeholder.success("Authentication Successful!")
                        st.balloons()
                        time.sleep(1)
                        success_placeholder.empty()
                        title_placeholder.empty()
                        help_placeholder.empty()
                        st.rerun()
                    else:
                        st.error("❌ Incorrect Password. Please Try Agian.")

def extract_data():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
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

    if not st.session_state["authenticated"]:
        authenticate()
    else:
        FASTAPI_URL = "http://18.218.84.81:8000/"
        #FASTAPI_URL = "http://3.141.231.76:8000/"
        st.session_state["LAMBDA_URL"] = "https://hc5ycktqbvxywpf4f4xhxfvm2e0dpozl.lambda-url.us-east-2.on.aws/"   # Function URL or API GW route               # EC2 FastAPI base

        HEALTH_URL = FASTAPI_URL + "healthz"
        #---- session flags ----
        st.session_state.setdefault("ec2_start_triggered", False)
        st.session_state.setdefault("fastapi_ready", False)

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
                        zip_data = io.BytesIO(resp.content)

                        with zipfile.ZipFile(zip_data, "r") as z:
                            with z.open("counts") as f:
                                st.session_state["counts_tmp"] = pd.read_feather(f)
                            
                            with z.open("metadata") as f:
                                st.session_state["metadata_tmp"] = pd.read_feather(f)
                            
                            with z.open("job.json") as f:
                                st.session_state["job_id_tmp"] = json.load(f)["job_id"]
                    
                        
                
                if st.session_state["counts_tmp"] is not None and st.session_state["metadata_tmp"] is not None:

                    @st.cache_data
                    def convert_for_download(df):
                        return df.to_csv().encode("utf-8")

                    
                    st.subheader("Short version Counts Table Preview:")
                    st.dataframe(st.session_state["counts_tmp"].head())

                    csv_counts = convert_for_download(st.session_state["counts_tmp"])

                    st.download_button(
                        label="Download Full Counts CSV",
                        data=csv_counts,
                        file_name="counts_table_brb.csv",
                        mime="text/csv",
                        icon=":material/download:",
                    )


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


                    st.subheader("Using downloaded data")
                    st.markdown("You can use downloaded data inside the NMF tool. Just click Save data button and then move to the NMF tool."
                        "Downloaded data will be populated where required, so just follow the steps described in the tool.")
                    
                    if st.button("Save Data for NMF Tool"):
                        #if counts is not None and metadata is not None:

                        #st.session_state["counts"] = st.session_state["counts_tmp"]
                        st.session_state["meta"] = st.session_state["metadata_tmp"]
                        st.session_state["job_id"] = st.session_state["job_id_tmp"]
                        st.session_state["saved_for_nmf"] = True
                        st.session_state["gene_column"] = "Unnamed: 0"
                        st.success("Data Saved")
                        

                    if st.session_state["saved_for_nmf"]:
                        if st.button("Move to NMF Tool"):
                            st.session_state["_go_to_main"] = "NMF for Bulk RNA"
                            st.rerun()
    
