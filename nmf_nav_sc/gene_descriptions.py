from openai import OpenAI
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import io
from ui_theme import apply_custom_theme

def run_gene_loadings():
    apply_custom_theme()

    if "API_URL" not in st.session_state:
        st.session_state["API_URL"] = "http://3.141.231.76:8000/"
    if "results_summary" not in st.session_state:
        st.session_state["results_summary"] = None
    df = None


    if "cnmf_gene_loadings" not in st.session_state and "gene_loadings" not in st.session_state:
        st.subheader("Please run one of the NMF algorithms to obtain gene loadings data")
    else:
        if "cnmf_gene_loadings" in st.session_state and st.session_state["cnmf_gene_loadings"] is not None:
            df = st.session_state["cnmf_gene_loadings"]
        elif st.session_state["gene_loadings"] is not None:
            df = st.session_state["gene_loadings"]
        else:
            st.subheader("Please, run any of the NMF algorithms to obtain gene loadings information before proceeding with this step")


        if df is not None:
            st.header("Choose the number of top genes from each module whose description you would like to get summarised")
            st.subheader("Gene loadings data preview")
            st.dataframe(df.head())


            st.subheader("Choose the top number of genes from each module")
            top_n = st.number_input("number of genes", 2, 5000, 20)
            buffer = io.BytesIO()
            df.to_feather(buffer)   # write DataFrame into memory
            buffer.seek(0)          # rewind to start
            files = {"file": ("gene_loadings.feather", buffer, "application/octet-stream")}
            data = {
                "top_n": top_n
            }

            if st.button("Summarise gene functions"):
                with st.spinner("Summarising gene functions..."):
                    response = requests.post(st.session_state["API_URL"] + "/process_gene_loadings/", files=files, data=data)

                    if response.status_code == 200:
                        st.session_state["results_summary"] = response.json()
                        st.success("Results received!")
                    else:
                        st.error(f"Backend error {response.status_code}: {response.text}")

        if st.session_state["results_summary"]:
            results = st.session_state["results_summary"]

            # Build tabs for each module
            tabs = st.tabs(list(results.keys()))
            for i, module in enumerate(results.keys()):
                with tabs[i]:
                    df_out = pd.DataFrame(results[module])
                    
                    # Show as table
                    st.dataframe(df_out)

                    # Optional: Download button
                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Download {module} as CSV",
                        data=csv_bytes,
                        file_name=f"{module}_top_genes.csv",
                        mime="text/csv"
                    )

    if st.button("Continue"):
        st.session_state["_go_to"] = "Spearman Correlation Analysis"
        st.rerun()