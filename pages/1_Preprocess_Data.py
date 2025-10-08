import streamlit as st
import requests
import pandas as pd
import io   
import threading
import queue
from streamlit_autorefresh import st_autorefresh

if "meta" not in st.session_state or st.session_state["meta"] is None:
    st.error("Upload metadata first on the previous page before proceeding here.")

st.header("You can either upload data for pre-processing or upload already pre-processed data")
if "preprocessed_feather" not in st.session_state:
    st.session_state["preprocessed_feather"] = None
if "process_status" not in st.session_state:
    st.session_state["process_status"] = None
if "result_queue" not in st.session_state:
    st.session_state["result_queue"] = queue.Queue()
if "API_URL" not in st.session_state:
    st.session_state["API_URL"] = "http://52.14.223.10:8000/"

def start_thread(files, url, data, result_queue):
    CONNECT_TIMEOUT = 10          # seconds to establish TCP connection
    READ_TIMEOUT    = 1800        # seconds to wait for server response bytes

    try:
        r = requests.post(
            url + "preprocess_df",
            files=files,
            data=data,
            #timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)   # <-- key change
        )
        r.raise_for_status()
        result_queue.put(("finished", r.content))
    except requests.exceptions.ReadTimeout:
        result_queue.put(("error", "The server took too long to respond. Try again or increase READ_TIMEOUT."))
    except requests.exceptions.ConnectionError as e:
        result_queue.put(("error", f"Connection error: {e}"))
    except requests.exceptions.HTTPError as e:
        # Try to extract the AI-generated summary from FastAPI JSON
        if e.response is not None:
            try:
                err_json = e.response.json()
                user_msg = err_json.get("message", "An unknown server error occurred.")
                error_id = err_json.get("error_id", "N/A")
                details = err_json.get("details", "")
                formatted = f"⚠️ **{user_msg}**\n\nError ID: `{error_id}`\n{details}"
                result_queue.put(("error", formatted))
            except ValueError:
                # Not JSON (plain text or HTML)
                result_queue.put(("error", f"Server error: {e.response.status_code}\n\n{e.response.text[:500]}"))
        else:
            result_queue.put(("error", f"Server error: {e}"))
    except Exception as e:
        result_queue.put(("error", f"Unexpected error: {str(e)}"))

def submit_preprocess():
    meta_buf = io.StringIO()
    st.session_state["meta"].to_csv(meta_buf, sep="\t", index=False)
    meta_bytes = meta_buf.getvalue().encode("utf-8")
    files = {
            "counts": (st.session_state["counts"].name, st.session_state["counts"].getvalue(), "text/plain"),
            "metadata": ("metadata.tsv", meta_bytes, "text/plain"),
        }
    if "batch" in st.session_state and st.session_state["batch"]: 
        data = {
        "gene_column": st.session_state["gene_column"],
        "metadata_index": st.session_state["metadata_index"],
        "design_factor": st.session_state["design_factor"],
        "hvg": st.session_state["hvg"],
        "symbols": st.session_state["gene_symbols"],
        "batch": st.session_state["batch"],
        "batch_include": ",".join(st.session_state["batch_covars"]),
        "batch_column": st.session_state["batch_column"]
    }
    elif "batch" in st.session_state and not st.session_state["batch"]:
        data = {
        "gene_column": st.session_state["gene_column"],
        "metadata_index": st.session_state["metadata_index"],
        "design_factor": st.session_state["design_factor"],
        "hvg": st.session_state["hvg"],
        "symbols": st.session_state["gene_symbols"],
        "batch": False,
        "batch_column": "",
        "batch_include": "",
    }
    st.session_state["process_status"] = "running"
    threading.Thread(target=start_thread, args=(files, st.session_state["API_URL"], data, st.session_state["result_queue"]), daemon=True).start()

@st.dialog("Preprocess Data")
def preprocess_window():
    st.session_state["counts"] = st.file_uploader("counts", type=["csv","tsv","txt"])
    st.session_state["hvg"] = st.number_input("Number of High Variable Genes", value=2000, key="hvg_preproc")
    #st.session_state["gene_column"] = st.text_input("Column name of the gene IDs in your count file", placeholder="Unnamed: 0")
    st.session_state["design_factor"] = st.text_input("Design factor", placeholder="Group", key="design_factor_preproc")
    st.session_state["gene_column"] = st.text_input("Enter the column name with gene names if no name enter Unknown: 0", placeholder="Unknown: 0") 
    st.session_state["batch"] = st.checkbox("Check the box if you would like to run batch correction", value=False)
    if st.session_state["batch"]:
        st.write("Please select the main effect(s) [metadata variables] that should be used for batch correction. The values of these main effects must be represented in at least two batches.")
        if st.session_state.get("meta") is not None:
            meta_cols = list(st.session_state["meta"].columns)
            st.session_state["batch_covars"] = st.multiselect(
                "Select metadata columns to keep as covariates",
                options=meta_cols,
                default=[]
            )
        else:
            st.warning("Upload metadata first to select batch covariates.")

        st.session_state["batch_column"] = st.text_input("Column name that holds batch data", placeholder="Batch")   
    st.session_state["gene_symbols"] = st.checkbox("Check the box if your count table is converted to symbol IDs", value=False)
    if st.button("Submit", key="btn_preproc") and st.session_state["counts"] is not None and st.session_state["meta"] is not None:
        submit_preprocess()
        st.rerun()

if st.button("Run preprocessing"):
    preprocess_window()

st.write("OR")

if not st.session_state["preprocessed_feather"]:
    uploaded = st.file_uploader("Upload preprocessed matrix", type=["csv", "feather"])
    if uploaded:
         if uploaded.name.endswith(".csv"):
              df = pd.read_csv(uploaded)
              st.session_state["preprocessed_feather"] = io.BytesIO()
              df.to_feather(st.session_state["preprocessed_feather"])
              st.session_state["preprocessed_feather"] = st.session_state["preprocessed_feather"].getvalue()
         elif uploaded.name.endswith(".feather"):
              st.session_state["preprocessed_feather"] = uploaded.getvalue()
         else:
              st.error("Unsupported file type. Please upload a CSV or Feather file.")

# Check for results first
if not st.session_state["result_queue"].empty():
    status, content = st.session_state["result_queue"].get()
    if status == "finished":
        st.session_state["preprocessed_feather"] = content
        st.session_state["process_status"] = "finished"
        st.rerun()  # Force refresh to show results immediately
    elif status == "error":
        st.session_state["process_status"] = "error"
        st.session_state["error_message"] = content
        st.rerun()

# Display status
if st.session_state["process_status"] == "running":
    st.info("Preprocessing is running... please wait.")
    st_autorefresh(interval=2000, key="preproc_refresh")
elif st.session_state["process_status"] == "finished":
    st.success("Preprocessing finished successfully!")
elif st.session_state["process_status"] == "error":
    st.error(st.session_state.get("error_message", "An unknown error occurred."))

if st.session_state["preprocessed_feather"]:
    st.subheader("Preprocessed preview")
    if st.button("Remove preprocessed data", type="secondary"):
        st.session_state["preprocessed_feather"] = None
        st.session_state["process_status"] = None
        st.rerun() 
    buf = io.BytesIO(st.session_state["preprocessed_feather"])
    df = pd.read_feather(buf)
    st.dataframe(df)

    st.download_button(
        "Download preprocessed feather",
        data=st.session_state["preprocessed_feather"],
        file_name="preprocessed.feather",
        mime="application/octet-stream",
        key="dl_preproc"
    )

    st.page_link("pages/2_Explore_K_Values.py", label="Continue")