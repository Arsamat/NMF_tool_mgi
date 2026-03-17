"""
DEG group selection page: load metadata from backend, display as dataframe with
checkboxes, assign samples to Group A or Group B, remove from groups, submit to backend.
Uses same authentication and EC2 wake-up (Lambda) pattern as brb_data_pages/extract_counts_frontend.
"""
import streamlit as st
import requests
import pandas as pd
import io
import zipfile
from streamlit_autorefresh import st_autorefresh
from ui_theme import apply_custom_theme
from deg.group_helpers import (
    add_to_group,
    remove_from_group,
    clear_group,
    add_samples_to_group,
    natural_language_group_assignment,
    SAMPLE_COL,
    ensure_auth_session,
    authenticate,
    ensure_ec2_wake_session,
    start_ec2_once,
    check_health,
    natural_language_to_filters
)
from deg.viz_helpers import render_deg_results_and_visualizations



# Default API URL and Lambda URL (wake EC2; same pattern as extract_counts_frontend)
#DEG_API_URL = "http://3.141.231.76:8000/"
DEG_API_URL = "http://18.218.84.81:8000/"
DEG_LAMBDA_URL = "https://hc5ycktqbvxywpf4f4xhxfvm2e0dpozl.lambda-url.us-east-2.on.aws/"


def _ensure_session():
    st.session_state.setdefault("deg_api_url", DEG_API_URL)
    st.session_state.setdefault("deg_schema", None)
    st.session_state.setdefault("deg_metadata_df", None)  # full metadata + Select column
    st.session_state.setdefault("deg_group_a", [])
    st.session_state.setdefault("deg_group_b", [])
    st.session_state.setdefault("deg_filters", {})
    st.session_state.setdefault("deg_editor_reset", 0)  # increment to reset data_editor checkboxes
    st.session_state.setdefault("deg_nl_filter_input_reset", 0)
    st.session_state.setdefault("deg_heatmap_df", None)   # matrix (genes x samples), row index = gene label
    st.session_state.setdefault("deg_heatmap_annotation_df", None)  # SampleName, Group
    st.session_state.setdefault("deg_gsea_df", None)      # GSEA Hallmark results table


def run_group_selection():
    apply_custom_theme()

    # --- Authentication (same as extract_counts_frontend) ---
    ensure_auth_session()
    if not st.session_state["authenticated"]:
        authenticate()
        return

    # --- EC2 wake-up via Lambda (same as extract_counts_frontend) ---
    ensure_ec2_wake_session(DEG_API_URL, DEG_LAMBDA_URL)
    if not st.session_state.get("deg_ec2_start_triggered"):
        start_ec2_once()
    if not st.session_state.get("deg_fastapi_ready", "False"):
        st.info("Waking up the compute node… Please wait until it is ready to proceed.")
        check_health(DEG_API_URL.rstrip("/") + "/healthz")
        st_autorefresh(interval=8000, key="deg_wake_refresh")
        return
    st.success("Compute node is ready.")

    _ensure_session()
    api_url = st.session_state["deg_api_url"]
    ga = st.session_state["deg_group_a"]
    gb = st.session_state["deg_group_b"]

    st.title("DEG Analysis: Group Selection")

    # ----- Step 1: Load schema -----
    #st.subheader("Step 1: Load metadata schema")
    if st.session_state.get("deg_fastapi_ready", "False") and st.session_state.get("deg_schema") is None:
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

    # ----- Step 2: Optional filters (with two filtering methods) -----
    st.subheader("Step 1: (Optional) Filter metadata")

    tab1, tab2, tab3= st.tabs(["Manual Selection", "Natural Language", "View Available Sample Values"])

    with tab1:
        st.markdown("### Manual Filter Selection")
        selected_cols = st.multiselect("Filter by columns:", filterable_columns, key="deg_filter_cols")
        deg_filters = {}
        for col in selected_cols:
            if col in unique_vals:
                vals = st.multiselect(f"Values for {col}:", unique_vals[col], key=f"deg_vals_{col}")
                if vals:
                    deg_filters[col] = vals
        if "Dose" in deg_filters and "Treatment" in deg_filters and "Vehicle" in deg_filters["Treatment"]:
            deg_filters["Dose"].append("Vehicle")

        # Preview filters in Tab 1
        if deg_filters:
            st.markdown("#### Filter Preview")
            preview_cols = st.columns(len(deg_filters))
            for idx, (col_name, values) in enumerate(deg_filters.items()):
                with preview_cols[idx]:
                    st.metric(col_name, len(values), f"values selected")
                    st.caption(", ".join([str(v)[:20] for v in values[:3]]) + ("..." if len(values) > 3 else ""))
            # Only update session state if filters are actually selected
            st.session_state["deg_filters"] = deg_filters
        else:
            st.caption("No filters applied - all samples will be included")

    with tab2:
        st.markdown("### Natural Language Filter")
        st.caption("Describe what samples you want in natural language")
        st.caption("Note: This is an experimental feature and may not work as expected all the time.")
        st.write(":red[When building a query mention what samples you want to use as control samples!]")
        user_description = st.text_area(
            "Example query: Give me WT or TDP43 iMNs treated with Rotenone at any timepoint excluding 96 hr",
            key=f"deg_nl_filter_input_{st.session_state.get('deg_nl_filter_input_reset', 0)}",
            height=80,
            placeholder="Type your filter description here..."
        )

        col_nl1, col_nl2 = st.columns(2)
        with col_nl1:
            generate_btn = st.button("🤖 Generate Filters from Description", key="deg_gen_filters_btn")
        with col_nl2:
            clear_nl_btn = st.button("Clear", key="deg_clear_nl_btn")

        if clear_nl_btn:
            st.session_state["deg_nl_filters"] = {}
            #st.session_state["deg_nl_filter_input"] = ""
            st.session_state["deg_nl_filter_input_reset"] = st.session_state.get("deg_nl_filter_input_reset", 0) + 1
            st.rerun()

        if generate_btn:
            if not user_description.strip():
                st.warning("Please enter a filter description")
            else:
                with st.spinner("🔍 Generating filters..."):
                    generated_filters = natural_language_to_filters(user_description, schema)
                    st.session_state["deg_nl_filters"] = generated_filters
                    st.rerun()

        # Show generated filters
        nl_filters = st.session_state.get("deg_nl_filters", {})

        if nl_filters:
            st.markdown("#### Generated Filters")

            # Display filter cards
            filter_cols = st.columns(min(3, len(nl_filters)) if nl_filters else 1)
            for idx, (col_name, values) in enumerate(nl_filters.items()):
                with filter_cols[idx % 3]:
                    st.info(f"**{col_name}**\n{len(values)} value(s) selected")
                    st.caption(", ".join([str(v)[:15] for v in values[:5]]) + ("..." if len(values) > 5 else ""))


            st.write(":red[Click Apply These Filters to load metadata table based on AI generated filters]")
            # Apply button
            if st.button("✓ Apply These Filters", key="deg_apply_nl_filters"):
                st.success("Filters applied!")
                st.session_state["deg_filters"] = nl_filters
        else:
            st.caption("Generated filters will appear here")
    
    with tab3:
        st.markdown("### View Available Metadata Values")
        vals_to_view = ["Genotype", "Treatment2", "Timepoint", "Treatment","Dose", "CellType", "Maturity"]
        for val in vals_to_view:
            values = sorted(unique_vals[val])
            with st.expander(f"{val} ({len(values)} unique)"):
                st.write(values)
        

    # ----- Step 3: Load metadata table -----
    st.subheader("Step 2: Load metadata table")
    if st.button("Load metadata table"):
        try:
            # Get filters from session state (set by either tab)
            applied_filters = st.session_state.get("deg_filters", {})
            payload = {"filters": applied_filters}
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
    st.subheader("Step 3: Select samples and assign to Comparison Group or Reference Group")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Select all rows", key="deg_select_all"):
            df = st.session_state["deg_metadata_df"].copy()
            df["Select"] = True
            st.session_state["deg_metadata_df"] = df
            st.session_state["deg_editor_reset"] = st.session_state.get("deg_editor_reset", 0) + 1
            st.rerun()
    with col2:
        if st.button("Clear selection", key="deg_clear_all"):
            df = st.session_state["deg_metadata_df"].copy()
            df["Select"] = False
            st.session_state["deg_metadata_df"] = df
            st.session_state["deg_editor_reset"] = st.session_state.get("deg_editor_reset", 0) + 1
            st.rerun()
    

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
        st.button("Add selected → Comparison Group", on_click=add_to_group, args=(edited, "deg_group_a", "deg_group_b"), key="add_grp_a")
        st.markdown("**Comparison Group**")
        with st.container(height=GROUP_WINDOW_HEIGHT):
            if not ga:
                st.caption("No samples yet.")
            else:
                for i, s in enumerate(ga):
                    st.button(f"✕ {s}", key=f"rm_a_{i}_{s}", on_click=remove_from_group, args=("deg_group_a", s))
        st.button("Clear Comparison Group", on_click=clear_group, args=("deg_group_a",))
    with col2:
        st.button("Add selected → Reference Group", on_click=add_to_group, args=(edited, "deg_group_b", "deg_group_a"), key="add_grp_b")
        st.markdown("**Reference Group**")
        with st.container(height=GROUP_WINDOW_HEIGHT):
            if not gb:
                st.caption("No samples yet.")
            else:
                for i, s in enumerate(gb):
                    st.button(f"✕ {s}", key=f"rm_b_{i}_{s}", on_click=remove_from_group, args=("deg_group_b", s))
        st.button("Clear Reference Group", on_click=clear_group, args=("deg_group_b",))

    st.caption(f"Comparison Group: {len(ga)} samples · Reference Group: {len(gb)} samples")

    with st.expander("Natural Language Group Assignment", expanded=True):
        # ----- AI Group Assignment -----
        st.divider()
        st.markdown("#### 🤖 Add Samples via AI Query")
        st.caption(
            "Describe the samples you want and which group to put them in. "
            "Example: *Add WT samples treated with Rotenone at 24h to Comparison Group*"
        )

        llm_query = st.text_area(
            "Query",
            key="deg_llm_group_query",
            height=80,
            placeholder="e.g. Add TDP43 Vehicle controls to Reference Group",
            label_visibility="collapsed",
        )

        col_llm1, col_llm2 = st.columns([2, 1])
        with col_llm1:
            ask_btn = st.button("🤖 Ask AI to Add Samples", key="deg_llm_group_btn")
        with col_llm2:
            if st.button("Clear AI result", key="deg_llm_clear_btn"):
                st.session_state.pop("deg_llm_group_result", None)
                st.rerun()

        if ask_btn:
            if not llm_query.strip():
                st.warning("Please enter a query first.")
            else:
                with st.spinner("Asking AI…"):
                    result = natural_language_group_assignment(
                        llm_query,
                        st.session_state["deg_metadata_df"],
                        st.session_state["deg_schema"],
                    )
                if result["error"]:
                    st.error(f"AI error: {result['error']}")
                else:
                    res_a = result["group_a"]
                    res_b = result["group_b"]
                    added_a, added_b = [], []
                    if res_a["samples"]:
                        added_a = add_samples_to_group(res_a["samples"], "deg_group_a", "deg_group_b") or []
                    if res_b["samples"]:
                        added_b = add_samples_to_group(res_b["samples"], "deg_group_b", "deg_group_a") or []
                    st.session_state["deg_llm_group_result"] = {
                        "group_a": {"filters": res_a["filters"], "matched": res_a["samples"], "added": added_a},
                        "group_b": {"filters": res_b["filters"], "matched": res_b["samples"], "added": added_b},
                    }
                    st.rerun()

        llm_result = st.session_state.get("deg_llm_group_result")
        if llm_result:
            def _show_group_result(label, data):
                filters = data["filters"]
                matched = data["matched"]
                added = data["added"]
                if not filters and not matched:
                    return
                if filters:
                    filter_summary = ", ".join(f"{k}: {v}" for k, v in filters.items())
                    st.info(f"**{label} filters:** {filter_summary}")
                if not matched:
                    st.warning(f"No samples matched for {label}. Try rephrasing.")
                else:
                    skipped = len(matched) - len(added)
                    st.success(
                        f"Added **{len(added)}** sample(s) to **{label}**"
                        + (f" ({skipped} skipped — already in a group)" if skipped else "")
                    )
                    with st.expander(f"{label} matched samples ({len(matched)})"):
                        st.write(matched)

            _show_group_result("Comparison Group", llm_result["group_a"])
            _show_group_result("Reference Group", llm_result["group_b"])

    # ----- Step 5: Submit to backend -----
    st.subheader("Step 4: Send groups to backend")
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
