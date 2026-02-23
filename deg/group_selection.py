"""
DEG group selection page: load metadata from backend, display as dataframe with
checkboxes, assign samples to Group A or Group B, remove from groups, submit to backend.
"""
import streamlit as st
import requests
import pandas as pd
import io
import zipfile
from ui_theme import apply_custom_theme
from deg.group_helpers import add_to_group, remove_from_group, clear_group, SAMPLE_COL
from deg.viz_helpers import render_deg_results_and_visualizations

# Default API URL (same pattern as NMF pages)
#DEG_API_URL = "http://18.218.84.81:8000/"
DEG_API_URL = "http://3.141.231.76:8000/"


def _ensure_session():
    st.session_state.setdefault("deg_api_url", DEG_API_URL)
    st.session_state.setdefault("deg_schema", None)
    st.session_state.setdefault("deg_metadata_df", None)  # full metadata + Select column
    st.session_state.setdefault("deg_group_a", [])
    st.session_state.setdefault("deg_group_b", [])
    st.session_state.setdefault("deg_filters", {})
    st.session_state.setdefault("deg_editor_reset", 0)  # increment to reset data_editor checkboxes
    st.session_state.setdefault("deg_heatmap_df", None)   # matrix (genes x samples), row index = gene label
    st.session_state.setdefault("deg_heatmap_annotation_df", None)  # SampleName, Group
    st.session_state.setdefault("deg_gsea_df", None)      # GSEA Hallmark results table


def run_group_selection():
    apply_custom_theme()
    _ensure_session()

    api_url = st.session_state["deg_api_url"]
    ga = st.session_state["deg_group_a"]
    gb = st.session_state["deg_group_b"]

    st.title("DEG Analysis: Group Selection")

    # ----- Step 1: Load schema -----
    st.subheader("Step 1: Load metadata schema")
    if st.button("Load schema"):
        try:
            r = requests.get(f"{api_url}get_metadata/", timeout=30)
            r.raise_for_status()
            st.session_state["deg_schema"] = r.json()
            st.success("Schema loaded.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load schema: {e}")

    schema = st.session_state["deg_schema"]
    if schema is None:
        st.info("Click 'Load schema' to fetch metadata columns and filter options.")
        return

    columns = schema.get("columns", [])
    unique_vals = schema.get("unique_values", {})
    filterable_columns = [c for c in columns if c != SAMPLE_COL]

    # ----- Step 2: Optional filters -----
    st.subheader("Step 2: (Optional) Filter metadata")
    selected_cols = st.multiselect("Filter by columns:", filterable_columns, key="deg_filter_cols")
    deg_filters = {}
    for col in selected_cols:
        if col in unique_vals:
            vals = st.multiselect(f"Values for {col}:", unique_vals[col], key=f"deg_vals_{col}")
            if vals:
                deg_filters[col] = vals
    st.session_state["deg_filters"] = deg_filters

    # ----- Step 3: Load metadata table -----
    st.subheader("Step 3: Load metadata table")
    if st.button("Load metadata table"):
        try:
            payload = {"filters": deg_filters}
            resp = requests.post(f"{api_url}get_samples/", json=payload, timeout=60)
            resp.raise_for_status()
            buf = io.BytesIO(resp.content)
            meta = pd.read_feather(buf)
            if SAMPLE_COL not in meta.columns:
                st.error(f"Metadata has no column '{SAMPLE_COL}'.")
            else:
                meta = meta.copy()
                meta["Select"] = False
                cols_ordered = ["Select"] + [c for c in meta.columns if c != "Select"]
                st.session_state["deg_metadata_df"] = meta[cols_ordered]
                st.session_state["deg_editor_reset"] = st.session_state.get("deg_editor_reset", 0) + 1
                st.success("Metadata loaded.")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to load metadata: {e}")

    meta_df = st.session_state["deg_metadata_df"]
    if meta_df is None:
        st.info("Configure filters (or leave empty for all) and click 'Load metadata table'.")
        return

    # Keep Select column first (leftmost); migrate once if needed
    if "Select" in meta_df.columns and meta_df.columns[0] != "Select":
        cols_ordered = ["Select"] + [c for c in meta_df.columns if c != "Select"]
        meta_df = meta_df[cols_ordered]
        st.session_state["deg_metadata_df"] = meta_df

    # ----- Step 4: Select samples and assign to groups -----
    st.subheader("Step 4: Select samples and assign to Group A or Group B")
    editor_key = f"deg_data_editor_{st.session_state.get('deg_editor_reset', 0)}"
    edited = st.data_editor(
        meta_df,
        column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)},
        use_container_width=True,
        key=editor_key,
        disabled=[c for c in meta_df.columns if c != "Select"],
    )
    # Don't sync edited back to session state - avoids checkbox double-click bug
    # Session state is only updated on load or when we reset Select (add/clear group)

    if st.session_state.get("deg_dup_warning"):
        st.warning(st.session_state["deg_dup_warning"])
        st.session_state["deg_dup_warning"] = None

    GROUP_WINDOW_HEIGHT = 220

    col1, col2 = st.columns(2)
    with col1:
        st.button("Add selected → Group A", on_click=add_to_group, args=(edited, "deg_group_a", "deg_group_b"), key="add_grp_a")
        st.markdown("**Group A**")
        with st.container(height=GROUP_WINDOW_HEIGHT):
            if not ga:
                st.caption("No samples yet.")
            else:
                for i, s in enumerate(ga):
                    st.button(f"✕ {s}", key=f"rm_a_{i}_{s}", on_click=remove_from_group, args=("deg_group_a", s))
        st.button("Clear Group A", on_click=clear_group, args=("deg_group_a",))
    with col2:
        st.button("Add selected → Group B", on_click=add_to_group, args=(edited, "deg_group_b", "deg_group_a"), key="add_grp_b")
        st.markdown("**Group B**")
        with st.container(height=GROUP_WINDOW_HEIGHT):
            if not gb:
                st.caption("No samples yet.")
            else:
                for i, s in enumerate(gb):
                    st.button(f"✕ {s}", key=f"rm_b_{i}_{s}", on_click=remove_from_group, args=("deg_group_b", s))
        st.button("Clear Group B", on_click=clear_group, args=("deg_group_b",))

    st.caption(f"Group A: {len(ga)} samples · Group B: {len(gb)} samples")
    

    # ----- Step 5: Submit to backend -----
    st.subheader("Step 5: Send groups to backend")
    if st.button("Submit groups for DEG analysis", type="primary"):
        if not ga or not gb:
            st.warning("Both groups must have at least one sample.")
        elif len(ga) < 2 or len(gb) < 2:
            st.warning("DEG analysis requires at least 2 samples per group. Add more samples to each group.")
        else:
            with st.spinner("Running DEG analysis… this may take 1–2 minutes."):
                try:
                    resp = requests.post(
                        f"{api_url}deg_submit_groups/",
                        json={"group_a": ga, "group_b": gb},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    data = resp.content
                    buf = io.BytesIO(data)
                    # Response is ZIP (deg_analysis.feather + optional heatmap + GSEA) or legacy single feather
                    st.session_state["deg_heatmap_df"] = None
                    st.session_state["deg_heatmap_annotation_df"] = None
                    st.session_state["deg_gsea_df"] = None
                    try:
                        with zipfile.ZipFile(buf, "r") as zf:
                            st.session_state["deg_results_df"] = pd.read_feather(io.BytesIO(zf.read("deg_analysis.feather")))
                            if "heatmap_matrix.csv" in zf.namelist():
                                hm = pd.read_csv(io.BytesIO(zf.read("heatmap_matrix.csv")), index_col=0)
                                st.session_state["deg_heatmap_df"] = hm
                            if "heatmap_annotation.csv" in zf.namelist():
                                st.session_state["deg_heatmap_annotation_df"] = pd.read_csv(io.BytesIO(zf.read("heatmap_annotation.csv")))
                            if "gsea_results.csv" in zf.namelist():
                                st.session_state["deg_gsea_df"] = pd.read_csv(io.BytesIO(zf.read("gsea_results.csv")))
                    except zipfile.BadZipFile:
                        buf.seek(0)
                        st.session_state["deg_results_df"] = pd.read_feather(buf)
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    try:
                        err_msg = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
                    except Exception:
                        err_msg = str(e)
                    st.error(f"Request failed: {err_msg}")
                except Exception as e:
                    st.error(f"Submit failed: {e}")

    # Show DEG results and visualizations from last successful run
    if st.session_state.get("deg_results_df") is not None:
        render_deg_results_and_visualizations(st.session_state["deg_results_df"])
