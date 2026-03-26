"""
Experiment-scoped metadata/counts download UI (embedded in Browse by experiment).

Filters always include the selected experiment; users may add more columns to narrow further.
Uses the same backend endpoints as group_selection / extract_counts_frontend.
"""

from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import requests
import streamlit as st

from brb_data_pages.backend_download import presigned_download_url
from brb_data_pages.visualize_data import visualize_metadata
from deg.group_helpers import SAMPLE_COL


def _ensure_extract_session():
    st.session_state.setdefault("deg_exp_extract_metadata_df", None)
    st.session_state.setdefault("deg_exp_extract_counts_tmp", None)
    st.session_state.setdefault("deg_exp_extract_job_id_tmp", None)
    st.session_state.setdefault("deg_exp_extract_visualize", False)


@st.cache_data(show_spinner=False)
def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_experiment_extract_ui(api_url: str, experiment_name: str, schema: dict):
    """
    Render visualize / download metadata / load & download counts for samples matching
    Experiment == experiment_name plus any optional extra filters.
    """
    _ensure_extract_session()
    api_url = api_url.rstrip("/") + "/"

    st.subheader("Download metadata and counts")
    st.info(
        f"Results are **scoped to experiment `{experiment_name}`**. "
        "Add optional filters below to narrow samples further."
    )

    columns = schema.get("columns", [])
    unique_vals = schema.get("unique_values", {})
    filterable = [c for c in columns if c != SAMPLE_COL and c != "Experiment"]

    st.markdown("##### Optional additional filters")
    extra_cols = st.multiselect(
        "Filter by additional columns:",
        filterable,
        key="deg_exp_extract_extra_cols",
    )

    merged: dict[str, list] = {"Experiment": [experiment_name]}
    for col in extra_cols:
        if col not in unique_vals:
            continue
        vals = st.multiselect(
            f"Values for `{col}`:",
            options=list(unique_vals[col]),
            key=f"deg_exp_extract_vals_{col}",
        )
        if vals:
            merged[col] = vals

    if st.button("Load matching samples", type="primary", key="deg_exp_extract_load_samples"):
        try:
            resp = requests.post(f"{api_url}get_samples/", json={"filters": merged}, timeout=120)
            resp.raise_for_status()
            meta = pd.read_feather(io.BytesIO(resp.content))
            if meta.empty:
                st.warning("No samples matched these filters.")
            else:
                st.session_state["deg_exp_extract_metadata_df"] = meta
                st.session_state["deg_exp_extract_counts_tmp"] = None
                st.session_state["deg_exp_extract_job_id_tmp"] = None
                st.success(f"Loaded **{len(meta)}** sample(s).")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to load samples: {e}")

    meta_df = st.session_state.get("deg_exp_extract_metadata_df")
    if meta_df is None or meta_df.empty:
        st.caption("Load matching samples to enable download, visualization, and counts retrieval.")
        return

    st.caption(f"**{len(meta_df)}** samples in the current selection.")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="Download metadata CSV",
            data=_csv_bytes(meta_df),
            file_name=f"metadata_{experiment_name}_filtered.csv",
            mime="text/csv",
            icon=":material/download:",
            key="deg_exp_extract_dl_meta",
        )
    with c2:
        if st.button("Retrieve counts for this selection", key="deg_exp_extract_load_counts"):
            try:
                buf = io.BytesIO()
                meta_df.to_feather(buf)
                buf.seek(0)
                resp = requests.post(
                    f"{api_url}get_counts/",
                    files={"metadata": ("metadata.feather", buf, "application/octet-stream")},
                    timeout=300,
                )
                resp.raise_for_status()
                zip_data = io.BytesIO(resp.content)
                with zipfile.ZipFile(zip_data, "r") as zf:
                    with zf.open("counts") as f:
                        st.session_state["deg_exp_extract_counts_tmp"] = pd.read_feather(f)
                    with zf.open("job.json") as f:
                        st.session_state["deg_exp_extract_job_id_tmp"] = json.load(f).get("job_id")
                st.success("Counts loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load counts: {e}")

    if not st.session_state.get("deg_exp_extract_visualize"):
        if st.button("Visualize metadata", key="deg_exp_extract_viz_on"):
            st.session_state["deg_exp_extract_visualize"] = True
            st.rerun()
    else:
        if st.button("Close visualization", key="deg_exp_extract_viz_off"):
            st.session_state["deg_exp_extract_visualize"] = False
            st.rerun()
        visualize_metadata(meta_df)

    counts_df = st.session_state.get("deg_exp_extract_counts_tmp")
    if counts_df is not None:
        st.markdown("**Counts preview**")
        st.dataframe(counts_df.head(), use_container_width=True)
        job_id = st.session_state.get("deg_exp_extract_job_id_tmp")
        if job_id:
            try:
                dl_url = presigned_download_url(api_url, job_id, "counts")
                st.link_button("Download full counts table", dl_url)
            except Exception as e:
                st.error(f"Could not get download link: {e}")

            if st.button("Import Data to NMF Tool"):
                st.session_state["job_id"] = job_id
                st.session_state["meta"] = meta_df
                st.session_state["metadata_index"] = "SampleName"
                st.session_state["design_factor"] = "Group"
                st.session_state["brb_data"] = True
                st.session_state["gene_column"] = "Geneid"

                st.session_state["active_page"] = "NMF for Bulk RNA"

                st.rerun()
