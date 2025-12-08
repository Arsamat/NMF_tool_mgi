import streamlit as st
import requests
import json
import io
import pandas as pd

if "expression_heatmap" not in st.session_state:
    st.session_state["expression_heatmap"] = None

def get_expression_heatmap(gene_loadings_df, default_values):

    st.title("Gene Expression Heatmap")

    # ----------------------------------------------------------
    # Load local session state data
    # ----------------------------------------------------------
    preprocessed_df_bytes = st.session_state["preprocessed_feather"]  # bytes
    metadata_df = st.session_state["meta"]                        # pandas DataFrame

    metadata_index = st.session_state.get("metadata_index", metadata_df.columns[0])

    with st.form("exprssion_heatmap_form"):

        # ----------------------------------------------------------
        # Multi-select annotation columns (only on FRONTEND)
        # ----------------------------------------------------------
        annotation_cols = st.multiselect(
            "Select metadata fields for annotation bars",
            options=metadata_df.columns.tolist(),
            key="annotation_cols",
            default=default_values
        )

        # ----------------------------------------------------------
        # Number of top genes per module
        # ----------------------------------------------------------
        X = st.number_input(
            "Top genes per module",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )

        # ----------------------------------------------------------
        # Convert DataFrames → Feather (in-memory)
        # ----------------------------------------------------------
        gene_loadings_buf = io.BytesIO()
        gene_loadings_df.reset_index(drop=False).to_feather(gene_loadings_buf)
        gene_loadings_buf.seek(0)

        metadata_buf = io.BytesIO()
        metadata_df.reset_index(drop=False).to_feather(metadata_buf)
        metadata_buf.seek(0)

        submit_expr_heatmap = st.form_submit_button("Generate Expression Heatmap")

    # preprocessed_df_bytes is already a bytes-like feather object from your pipeline

    # ----------------------------------------------------------
    # Send to FastAPI
    # ----------------------------------------------------------
    if submit_expr_heatmap:
        with st.spinner("Rendering heatmap on backend..."):

            files = {
                "gene_loadings": ("gene_loadings.feather", gene_loadings_buf, "application/octet-stream"),
                "preprocessed_df": ("preprocessed.feather", preprocessed_df_bytes, "application/octet-stream"),
                "metadata": ("metadata.feather", metadata_buf, "application/octet-stream"),
            }

            data = {
                "annotation_cols": json.dumps(annotation_cols),
                "metadata_index": metadata_index,
                "X": str(X),
            }

            try:
                response = requests.post(st.session_state["API_URL"] + "/plot_heatmap/", files=files, data=data)

                if response.status_code == 200:
                    return response.content

                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Request failed: {e}")