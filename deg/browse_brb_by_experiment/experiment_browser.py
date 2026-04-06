"""
DEG experiment browser page.

Flow:
1) Authenticate + wake the compute node (same pattern as group selection)
2) Fetch schema via `GET /get_metadata/`
3) Display curated list of experiments
4) On selection, show metadata summary + actions:
   - Run de novo analysis (hand off to browse-by-metadata flow)
   - Browse precomputed results
   - Download metadata + counts for the experiment
"""

import io

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ui_theme import apply_custom_theme
from deg.group_helpers import (
    authenticate,
    check_health,
    ensure_auth_session,
    ensure_ec2_wake_session,
    start_ec2_once,
)

DEG_API_URL = "http://18.218.84.81:8000/"
DEG_LAMBDA_URL = "https://hc5ycktqbvxywpf4f4xhxfvm2e0dpozl.lambda-url.us-east-2.on.aws/"


def _curated_experiment_table() -> pd.DataFrame:
    rows = [
        {
            "Experiment": "MFN2_MNP_Rotenone",
            "Date": "May 2025",
            "Description": "WTC11 MNPs (WT, MFN2 Het or Hom / Vehicle, 0.4 uM or 2 uM Rotenone / 24 hrs)",
        },
        {
            "Experiment": "MNP_WT_LATS2KO_Rot_TRULI_Timecourse",
            "Date": "Dec 2025",
            "Description": "WTC11 MNPs (WT, LATS2KO / Vehicle, 1 uM Rotenone, 10 uM TRULI, or Both / 6, 16, 24, 48 and 96 hrs)",
        },
        {
            "Experiment": "iPSC_WT_LATS2KO_Rot_Tun_Timecourse",
            "Date": "",
            "Description": "WTC11 iPSCs (WT, LATS2KO / Vehicle, EC25 or EC75 Rotenone or Tunicamycin / 6 and 24 hrs)",
        },
        {
            "Experiment": "WT_MNP_Rotenone_Timecourse",
            "Date": "",
            "Description": "WTC11 MNPs (WT / Vehicle, 2 uM Rotenone / 6, 12, 24, and 48 hrs)",
        },
        {
            "Experiment": "Pilot",
            "Date": "Sep – Oct 2025",
            "Description": "KOLF2.1J iMNs (WT, TDP43 / Vehicle, EC10, EC50, or EC90 Rotenone or Thapsigargin / 6, 24, 72, or 96 hrs)",
        },
        {
            "Experiment": "Protocol_Dev1",
            "Date": "Sep 2025",
            "Description": "Seeding density and day of maturation tests for AN1 or KOLF2.1J cells, Anatomic iSN/iMN protocols; iMN C9orf72, DNMT1, FUS, MFN2, TBK1 not treated",
        },
        {
            "Experiment": "Protocol_Dev2",
            "Date": "Oct 2025",
            "Description": "KOLF2.1J iPSC, Day 5 or Day 21 iCNs (Stem Cell Technologies protocol), not treated",
        },
        {
            "Experiment": "Production",
            "Date": "Dec 2025 – March 2026",
            "Description": "KOLF2.1J iMNs or iSNs (WT, TDP43, TBK1, PINK1, LRRK2, DNMT1 / Vehicle, EC25 or EC75 of Rotenone, Thapsigargin, Tunicamycin, BrefeldinA, Vps34-IN-1 / 6, 24, and 72 hrs)",
        },
    ]
    return pd.DataFrame(rows, columns=["Experiment", "Date", "Description"])


def _render_metadata_summary(meta: pd.DataFrame):
    st.subheader("Metadata summary (from database)")
    if meta is None or meta.empty:
        st.warning("No metadata rows returned for this experiment.")
        return

    st.caption(f"Rows: {meta.shape[0]} · Columns: {meta.shape[1]}")
    summary_cols = ["CellType", "Genotype", "Treatment", "Dose", "Timepoint", "Maturity", "Background"]
    rows = []
    for col in summary_cols:
        if col not in meta.columns:
            continue
        values = meta[col].dropna().astype(str).unique().tolist()
        values_sorted = sorted(values, key=lambda v: str(v))
        rows.append({"Key": col, "Values": ", ".join(values_sorted)})
    rows.append({"Key": "Number of samples", "Values": str(int(meta.shape[0]))})
    st.dataframe(pd.DataFrame(rows, columns=["Key", "Values"]), use_container_width=True, hide_index=True)


def _ensure_session():
    st.session_state.setdefault("deg_api_url", DEG_API_URL)
    st.session_state.setdefault("deg_schema", None)
    st.session_state.setdefault("deg_experiment_browser_in_de_novo", False)
    st.session_state.setdefault("deg_experiment_browser_extract_data", False)
    st.session_state.setdefault("deg_extract_mode_experiment", None)
    st.session_state.setdefault("deg_experiment_browser_precomputed", False)


def _group_selection_keys_for_mode(mode: str) -> dict:
    p = "deg_exp_" if mode == "experiment" else "deg_novo_"
    return {
        "metadata_df": f"{p}metadata_df",
        "group_a": f"{p}group_a",
        "group_b": f"{p}group_b",
        "filters": f"{p}filters",
        "editor_reset": f"{p}editor_reset",
        "heatmap_df": f"{p}heatmap_df",
        "heatmap_annotation_df": f"{p}heatmap_annotation_df",
        "heatmap_image": f"{p}heatmap_image",
        "gsea_df": f"{p}gsea_df",
        "results_df": f"{p}results_df",
    }


@st.cache_data(show_spinner=False)
def _fetch_experiment_metadata(api_url: str, experiment: str) -> pd.DataFrame:
    payload = {"filters": {"Experiment": [experiment]}}
    resp = requests.post(f"{api_url}get_samples/", json=payload, timeout=60)
    resp.raise_for_status()
    return pd.read_feather(io.BytesIO(resp.content))


def run_experiment_browser():
    apply_custom_theme()

    ensure_auth_session()
    if not st.session_state["authenticated"]:
        authenticate()
        return

    ensure_ec2_wake_session(st.session_state["deg_api_url"], DEG_LAMBDA_URL)
    if not st.session_state.get("deg_ec2_start_triggered"):
        start_ec2_once()

    if not st.session_state.get("deg_fastapi_ready", "False"):
        st.info("Waking up the compute node… Please wait until it is ready to proceed.")
        check_health(st.session_state["deg_api_url"].rstrip("/") + "/healthz")
        st_autorefresh(interval=8000, key="deg_experiment_wake_refresh")
        return

    _ensure_session()
    api_url = st.session_state["deg_api_url"]

    if st.session_state.get("deg_experiment_browser_in_de_novo", False):
        if st.button("← Back to experiment list", key="deg_back_to_experiment_list"):
            st.session_state["deg_experiment_browser_in_de_novo"] = False
            st.rerun()

        st.subheader(f'The experiment selected is:  {st.session_state["current_experiment"]}')
        from deg.browse_brb_by_metadata.group_selection import run_group_selection

        run_group_selection(mode="experiment")
        return

    if st.session_state.get("deg_experiment_browser_precomputed", False):
        if st.button("← Back to experiment list", key="deg_back_from_precomputed"):
            st.session_state["deg_experiment_browser_precomputed"] = False
            st.rerun()

        from deg.browse_brb_by_experiment.precomputed_results_browser import run_precomputed_results_browser

        run_precomputed_results_browser()
        return

    if st.session_state.get("deg_experiment_browser_extract_data", False):
        exp_name = st.session_state.get("deg_extract_mode_experiment")
        if st.button("← Back to experiment list", key="deg_back_from_extract_data"):
            st.session_state["deg_experiment_browser_extract_data"] = False
            st.rerun()

        st.title("Download metadata and counts")
        if not exp_name:
            st.error("No experiment selected.")
            return

        schema = st.session_state.get("deg_schema")
        if not schema:
            st.warning("Metadata schema not loaded. Return to the experiment list and wait for schema to load.")
            return

        from deg.browse_brb_by_metadata.experiment_extract_ui import render_experiment_extract_ui

        render_experiment_extract_ui(api_url, exp_name, schema)
        return

    st.title("DEG Analysis: Browse by Experiment")

    if st.session_state.get("deg_fastapi_ready", "False") and st.session_state.get("deg_schema") is None:
        try:
            r = requests.get(f"{api_url}get_metadata/", timeout=30)
            r.raise_for_status()
            st.session_state["deg_schema"] = r.json()
            st.success("Compute node is ready.")
            st.success("Metadata schema loaded.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load schema: {e}")
            return

    schema = st.session_state.get("deg_schema")
    if schema is None:
        st.info("Click to load metadata schema.")
        return

    st.subheader("Experiments")
    st.caption("Click a row to select an experiment.")

    curated = _curated_experiment_table()
    selection = st.dataframe(
        curated,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Experiment": st.column_config.TextColumn("Experiment", width="medium"),
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
    )

    selected_experiment = st.session_state.get("deg_selected_experiment")
    try:
        sel_rows = getattr(selection, "selection", {}).get("rows", [])
        if sel_rows:
            idx = sel_rows[0]
            selected_experiment = curated.iloc[int(idx)]["Experiment"]
            st.session_state["deg_selected_experiment"] = selected_experiment
    except Exception:
        pass

    if not selected_experiment:
        st.info("Select an experiment to view metadata summary and proceed.")
        return

    unique_vals = schema.get("unique_values", {})
    experiments_in_db = set(map(str, unique_vals.get("Experiment", [])))
    if str(selected_experiment) not in experiments_in_db:
        st.warning(
            f"Experiment `{selected_experiment}` is not present in the current database schema. "
            "The curated table is displayed, but metadata fetch may return no rows."
        )

    st.subheader(f"Selected experiment: `{selected_experiment}`")

    try:
        with st.spinner("Loading experiment-matched metadata…"):
            meta_preview = _fetch_experiment_metadata(api_url, selected_experiment)
        _render_metadata_summary(meta_preview)
    except Exception as e:
        st.caption("Could not load experiment preview. The de novo flow may still work.")
        st.caption(str(e))

    col_run, col_pre, col_dl = st.columns(3)
    with col_run:
        run_btn = st.button(
            "Run de novo analysis (for this experiment)",
            type="primary",
            key="deg_run_de_novo_experiment",
        )
    with col_pre:
        pre_btn = st.button("Browse precomputed DE results", key="deg_browse_precomputed_experiment")
    with col_dl:
        dl_btn = st.button("Download metadata and counts data", key="deg_download_meta_counts_experiment")

    if pre_btn:
        st.session_state["deg_experiment_browser_precomputed"] = True
        st.session_state["deg_precomputed_view"] = "groups"
        st.session_state["deg_precomputed_experiment"] = selected_experiment
        st.session_state["deg_precomputed_group_data"] = None
        st.session_state["deg_precomputed_terms"] = None
        st.session_state["deg_precomputed_deg_df"] = None
        st.session_state["deg_precomputed_heatmap"] = None
        st.session_state["deg_precomputed_loaded_key"] = None
        st.session_state["deg_precomputed_selected_term"] = None
        st.rerun()

    if dl_btn:
        st.session_state["deg_experiment_browser_extract_data"] = True
        st.session_state["deg_extract_mode_experiment"] = selected_experiment
        st.session_state["deg_exp_extract_metadata_df"] = None
        st.session_state["deg_exp_extract_counts_tmp"] = None
        st.session_state["deg_exp_extract_job_id_tmp"] = None
        st.session_state["deg_exp_extract_visualize"] = False
        st.rerun()

    if not run_btn:
        return

    st.session_state["deg_experiment_browser_in_de_novo"] = True
    st.session_state["current_experiment"] = selected_experiment
    k = _group_selection_keys_for_mode("experiment")
    st.session_state[k["group_a"]] = []
    st.session_state[k["group_b"]] = []
    st.session_state[k["filters"]] = {"Experiment": [selected_experiment]}
    st.session_state[k["heatmap_df"]] = None
    st.session_state[k["heatmap_annotation_df"]] = None
    st.session_state[k["gsea_df"]] = None
    st.session_state[k["results_df"]] = None
    st.session_state[k["metadata_df"]] = None
    st.session_state[k["editor_reset"]] = st.session_state.get(k["editor_reset"], 0) + 1

    st.rerun()


__all__ = ["run_experiment_browser"]

