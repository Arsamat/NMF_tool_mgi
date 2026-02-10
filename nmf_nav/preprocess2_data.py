import streamlit as st
import requests
import pandas as pd
import io   
import threading
import queue
from streamlit_autorefresh import st_autorefresh
from ui_theme import apply_custom_theme
from .upload_counts_helper import upload_counts, upload_preprocessed_counts
import urllib.parse


def run_preprocess_data():
    apply_custom_theme()
    if "meta" not in st.session_state or st.session_state["meta"] is None:
        st.error("Upload metadata first on the previous page before proceeding here.")


    if "preprocessed_feather" not in st.session_state:
        st.session_state["preprocessed_feather"] = None
    if "process_status" not in st.session_state:
        st.session_state["process_status"] = None
    if "result_queue" not in st.session_state:
        st.session_state["result_queue"] = queue.Queue()
    # if "API_URL" not in st.session_state:
    #     st.session_state["API_URL"] = "http://52.14.223.10:8000/"
    if "integer_format" not in st.session_state:
        st.session_state["integer_format"] = False
    if "display_menu" not in st.session_state:
        st.session_state["display_menu"] = False
    if "display_preprocessed_upload" not in st.session_state:
        st.session_state["display_preprocessed_upload"] = False
    if "job_id" not in st.session_state:
        st.session_state["job_id"] = None
    
    #st.session_state["job_id"] = "6e2b3cf2-ca58-406c-a8e6-e9b2e9341fca"
        
    st.subheader("You can either upload data for pre-processing or upload already pre-processed data")
    if st.button('Learn More about pre-processing'):
        st.markdown('''
        Please upload your data with transcripts as rows and samples as columns. Transcripts should be identified by either **Ensembl ID** or **gene symbols**.

        Raw counts will undergo the following preprocessing:

        1) Lowly expressed genes will be removed with **edgeR::filterByExpr()**

        2) TMM normalization of library sizes

        3) Transformation to log(counts per million)

        4) Optional batch correction with **limma::removeBatchEffect()**

        **You must know which factors you can and should run batch correction on!**

        5) Selection of the top X most variable genes (generally 2000 - 5000)

        6) The resulting normalized, transformed expression values will be forced positive by a simple linear transformation to enable NMF.

        The reason we apply this preprocessing (rather than, e.g., size factor normalization with DESeq2 which preserves integer counts) 
        is because we have determined **limma::removeBatchEffect()** to be the best method for batch correction, and it requires log-transformed data. 
        **If you submit your own pre-processed data, you are not required to follow this exact procedure.** However, removal of lowly expressed genes and some sort of normalization 
        for library size (e.g. size factor normalization with DESeq2) is **highly encouraged**. 
        The main restriction for NMF is that the data must be **non-negative**.
                    ''')
        st.button("Close info")

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
            "batch_column": st.session_state["batch_column"],
            "job_id": st.session_state["job_id"],
            "single_cell": False
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
            "job_id": st.session_state["job_id"],
            "single_cell": False
        }
        st.session_state["process_status"] = "running"
        threading.Thread(target=start_thread, args=(files, st.session_state["API_URL"], data, st.session_state["result_queue"]), daemon=True).start()

    def preprocess_window():
        if st.session_state["job_id"] is None:
            upload_counts()
        else:
            st.write("Your counts data was already uploaded. Choose from parameters below and submit for pre-processing")
            if st.button("Remove current counts data"):
                st.session_state["job_id"] = None
                st.session_state["brb_data"] = None
                st.rerun()

        if "job_id" in st.session_state:
            st.session_state["hvg"] = st.number_input("Number of High Variable Genes", value=2000, key="hvg_preproc")

            #st.session_state["gene_column"] = st.text_input("Column name of the gene IDs in your count file", placeholder="Unnamed: 0")
           
            if "brb_data" not in st.session_state or not st.session_state["brb_data"]:
                st.session_state["gene_column"] = st.text_input("Type in name of the column that stores gene names")

            #st.session_state["single_cell"] = st.checkbox("Check if your data is single-cell data", key="single_cell_check", value=False)
            #st.write(st.session_state["single_cell"])
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

                st.session_state["batch_column"] = st.selectbox(
                        "Select metadata column that holds batch information",
                        options=meta_cols,
                        index=0
                    )   
                
            st.session_state["gene_symbols"] = st.checkbox("Check the box if your transcript names are gene symbols (instead of Ensembl IDs).", value=False)
            if st.button("Submit", key="btn_preproc") and st.session_state["meta"] is not None:
                submit_preprocess()
                st.rerun()
    
    st.markdown("---")

    if not st.session_state["display_menu"]:
        if st.button("Start preprocessing"):
            st.session_state["display_menu"] = True
            st.rerun()
    
    if st.session_state["display_menu"]:
        if st.button("Close menu"):
            st.session_state["display_menu"] = False
            st.rerun()   
        preprocess_window()

    st.write("OR")

    if not st.session_state["preprocessed_feather"]:
        # uploaded = st.file_uploader("Upload preprocessed matrix", type=["csv", "feather"])
        # if uploaded:
        #     if uploaded.name.endswith(".csv"):
        #         df = pd.read_csv(uploaded)
        #         st.session_state["preprocessed_feather"] = io.BytesIO()
        #         df.to_feather(st.session_state["preprocessed_feather"])
        #         st.session_state["preprocessed_feather"] = st.session_state["preprocessed_feather"].getvalue()
        #     elif uploaded.name.endswith(".feather"):
        #         st.session_state["preprocessed_feather"] = uploaded.getvalue()
        #     else:
        #         st.error("Unsupported file type. Please upload a CSV or Feather file.")
        
        if not st.session_state["display_preprocessed_upload"]:
            if st.button("Upload Already Preprocessed Data"):
                st.session_state["display_preprocessed_upload"] = True
                st.rerun()
        
        if st.session_state["display_preprocessed_upload"]:
            if st.button("Close menu", key="preprocess_menu"):
                st.session_state["display_preprocessed_upload"] = False
                st.rerun()   
            upload_preprocessed_counts()
            

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
        st.subheader("Preprocessed short version preview")
        if st.button("Remove preprocessed data", type="secondary"):
            st.session_state["preprocessed_feather"] = None
            st.session_state["process_status"] = None
            st.rerun() 
        buf = io.BytesIO(st.session_state["preprocessed_feather"])
        df = pd.read_feather(buf)
        st.dataframe(df)

        st.subheader("Download full preprocessed data frame")


        #st.session_state["integer_format"] = st.checkbox("Select if uploaded data is in integer format", value=st.session_state["integer_format"])
        
        job_id = st.session_state["job_id"]
        api = st.session_state["API_URL"].rstrip("/")

        download_endpoint = f"{api}/download_preprocessed_data?job_id={urllib.parse.quote(job_id)}&data_type=preprocessed"

        st.link_button("Download Full Preprocessed Data", download_endpoint)
    
        st.markdown("---")

        if st.button("Continue"):
            st.session_state["_go_to"] = "Explore K Parameter"
            st.rerun()