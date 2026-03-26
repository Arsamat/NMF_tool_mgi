"""
Filter/retrieve/download section for DEG group selection.

This keeps Step 1-3 UI logic out of `group_selection.py` so the main page file
can focus on orchestration and group assignment.
"""

import io
import json
import zipfile

import pandas as pd
import requests
import streamlit as st

from brb_data_pages.backend_download import presigned_download_url
from brb_data_pages.visualize_data import visualize_metadata
from deg.group_helpers import SAMPLE_COL, natural_language_to_filters


@st.cache_data
def _to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


def render_filter_and_data_steps(
    *,
    keys: dict,
    wk,
    api_url: str,
    schema: dict,
    render_step_header,
):
    """
    Render Step 1/2/3 UI (filters, metadata load, visualize/download).

    Returns:
        pd.DataFrame | None: loaded metadata dataframe including Select column.
    """
    columns = schema.get("columns", [])
    unique_vals = schema.get("unique_values", {})
    filterable_columns = [c for c in columns if c != SAMPLE_COL]

    # ----- Step 1: Optional filters -----
    render_step_header(
        1,
        "(Optional) Build Filters to Subset Samples",
        "Use manual filters or natural language prompts to narrow the sample set.",
        optional=True,
    )

    tab1, tab2, tab3 = st.tabs(["Manual Selection", "Natural Language", "View Available Sample Values"])

    with tab1:
        st.markdown("### Manual Filter Selection")
        selected_cols = st.multiselect("Filter by columns:", filterable_columns, key=wk("filter_cols"))
        deg_filters = st.session_state.get(keys["filters"], {})
        for col in selected_cols:
            if col in unique_vals:
                vals = st.multiselect(f"Values for {col}:", unique_vals[col], key=wk(f"vals_{col}"))
                if vals:
                    deg_filters[col] = vals
        if "Dose" in deg_filters and "Treatment" in deg_filters and "Vehicle" in deg_filters["Treatment"]:
            deg_filters["Dose"].append("Vehicle")

        if deg_filters:
            st.markdown("#### Filter Preview")
            col1, col2 = st.columns(2)
            with col1:
                preview_cols = st.columns(len(deg_filters))
                for idx, (col_name, values) in enumerate(deg_filters.items()):
                    with preview_cols[idx]:
                        st.metric(col_name, len(values), "values selected")
                        st.caption(", ".join([str(v)[:20] for v in values[:3]]) + ("..." if len(values) > 3 else ""))
                st.session_state[keys["filters"]] = deg_filters
            with col2:
                if st.button("Clear Filters", key=wk("clear_filters_btn")):
                    st.session_state[keys["filters"]] = {}
                    st.rerun()
        else:
            st.caption("No filters applied - all samples will be included")

    with tab2:
        st.markdown("### Natural Language Filter")
        st.caption("Describe what samples you want in natural language")
        st.caption("Note: This is an experimental feature and may not work as expected all the time.")
        st.write(":red[When building a query mention what samples you want to use as control samples!]")
        user_description = st.text_area(
            "Example query: Give me WT or TDP43 iMNs treated with Rotenone at any timepoint excluding 96 hr",
            key=f"{wk('nl_filter_input')}_{st.session_state.get(keys['nl_filter_input_reset'], 0)}",
            height=80,
            placeholder="Type your filter description here...",
        )

        col_nl1, col_nl2 = st.columns(2)
        with col_nl1:
            generate_btn = st.button("🤖 Generate Filters from Description", key=wk("gen_filters_btn"))
        with col_nl2:
            clear_nl_btn = st.button("Clear", key=wk("clear_nl_btn"))

        if clear_nl_btn:
            st.session_state[keys["nl_filters"]] = {}
            st.session_state[keys["nl_filter_input_reset"]] = st.session_state.get(keys["nl_filter_input_reset"], 0) + 1
            st.rerun()

        if generate_btn:
            if not user_description.strip():
                st.warning("Please enter a filter description")
            else:
                with st.spinner("🔍 Generating filters..."):
                    generated_filters = natural_language_to_filters(user_description, schema)
                    st.session_state[keys["nl_filters"]] = generated_filters
                    st.rerun()

        nl_filters = st.session_state.get(keys["nl_filters"], {})
        if nl_filters:
            st.markdown("#### Generated Filters")
            filter_cols = st.columns(min(3, len(nl_filters)) if nl_filters else 1)
            for idx, (col_name, values) in enumerate(nl_filters.items()):
                with filter_cols[idx % 3]:
                    st.info(f"**{col_name}**\n{len(values)} value(s) selected")
                    st.caption(", ".join([str(v)[:15] for v in values[:5]]) + ("..." if len(values) > 5 else ""))

            st.write(":red[Click Apply These Filters to load metadata table based on AI generated filters]")
            if st.button("✓ Apply These Filters", key=wk("apply_nl_filters")):
                st.success("Filters applied!")
                st.session_state[keys["filters"]] = nl_filters
        else:
            st.caption("Generated filters will appear here")

    with tab3:
        st.markdown("### View Available Metadata Values")
        vals_to_view = ["Genotype", "Treatment2", "Timepoint", "Treatment", "Dose", "CellType", "Maturity", "Experiment"]
        for val in vals_to_view:
            values = sorted(unique_vals[val])
            with st.expander(f"{val} ({len(values)} unique)"):
                st.write(values)

    # ----- Step 2: Load metadata table -----
    render_step_header(
        2,
        "Retrieve Sample Data",
        "If no filters are selected, the full metadata table is retrieved.",
        optional=False,
    )
    if st.button("Retrieve metadata table"):
        try:
            applied_filters = st.session_state.get(keys["filters"], {})
            payload = {"filters": applied_filters}
            resp = requests.post(f"{api_url}get_samples/", json=payload, timeout=60)
            resp.raise_for_status()
            meta = pd.read_feather(io.BytesIO(resp.content))
            if SAMPLE_COL not in meta.columns:
                st.error(f"Metadata has no column '{SAMPLE_COL}'.")
            else:
                meta = meta.copy()
                meta["Select"] = False
                cols_ordered = ["Select"] + [c for c in meta.columns if c != "Select"]
                st.session_state[keys["metadata_df"]] = meta[cols_ordered]
                st.session_state[keys["editor_reset"]] = st.session_state.get(keys["editor_reset"], 0) + 1
                st.session_state[keys["counts_tmp"]] = None
                st.session_state[keys["job_id_tmp"]] = None
                st.success("Metadata loaded.")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to retrieve metadata: {e}")

    meta_df = st.session_state[keys["metadata_df"]]
    if meta_df is None:
        st.info("Build filters to subset samples (or leave empty to extract all samples) and click 'Retrieve metadata table'.")
        return None

    if "Select" in meta_df.columns and meta_df.columns[0] != "Select":
        cols_ordered = ["Select"] + [c for c in meta_df.columns if c != "Select"]
        meta_df = meta_df[cols_ordered]
        st.session_state[keys["metadata_df"]] = meta_df

    # ----- Step 3: Visualize + downloads -----
    meta_for_export = meta_df.drop(columns=["Select"], errors="ignore").copy()
    st.write(f"Filtered metadata contains **{len(meta_for_export)}** samples.")

    if not st.session_state.get(keys["visualize_data"]):
        if st.button("Visualize filtered metadata", key=wk("visualize_data_btn")):
            st.session_state[keys["visualize_data"]] = True
            st.rerun()

    if st.session_state.get(keys["visualize_data"]):
        if st.button("Close Visualization", key=wk("stop_visualize_data_btn")):
            st.session_state[keys["visualize_data"]] = False
            st.rerun()
        visualize_metadata(meta_for_export)

    render_step_header(
        3,
        "(Optional) Download Filtered Data and Visualize",
        "Preview distribution, export metadata, or request filtered counts for download.",
        optional=True,
    )

    md_col1, md_col2 = st.columns(2)
    with md_col1:
        st.download_button(
            label="Download filtered metadata CSV",
            data=_to_csv_bytes(meta_for_export),
            file_name="filtered_metadata.csv",
            mime="text/csv",
            icon=":material/download:",
        )
    with md_col2:
        if st.button("Retrieve filtered counts table for download", key=wk("load_filtered_counts")):
            try:
                buf = io.BytesIO()
                meta_for_export.to_feather(buf)
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
                        st.session_state[keys["counts_tmp"]] = pd.read_feather(f)
                    with zf.open("job.json") as f:
                        st.session_state[keys["job_id_tmp"]] = json.load(f).get("job_id")
                st.success("Filtered counts loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load filtered counts: {e}")

    if st.session_state.get(keys["counts_tmp"]) is not None:
        st.markdown("**Counts preview**")
        st.dataframe(st.session_state[keys["counts_tmp"]].head(), use_container_width=True)
        job_id = st.session_state.get(keys["job_id_tmp"])
        if job_id:
            try:
                dl_url = presigned_download_url(api_url, job_id, "counts")
                st.link_button("Download full filtered counts table", dl_url)
            except Exception as e:
                st.error(f"Could not get download link: {e}")

            if st.button("Import data to NMF Tool"):
                st.session_state["job_id"] = job_id
                st.session_state["meta"] = meta_df
                st.session_state["metadata_index"] = "SampleName"
                st.session_state["design_factor"] = "Group"
                st.session_state["brb_data"] = True
                st.session_state["gene_column"] = "Geneid"

                st.session_state["active_page"] = "NMF for Bulk RNA"

                #st.session_state["_go_to_main"] = "NMF for Bulk RNA"
                st.rerun()

    return meta_df

