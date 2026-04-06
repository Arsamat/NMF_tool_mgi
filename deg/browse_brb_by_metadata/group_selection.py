"""
DEG group selection page: load metadata from backend, display as dataframe with
checkboxes, assign samples to Group A or Group B, remove from groups, submit to backend.
"""

import io
import zipfile

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ui_theme import apply_custom_theme
from deg.group_helpers import (
    add_to_group,
    add_samples_to_group,
    authenticate,
    check_health,
    clear_group,
    ensure_auth_session,
    ensure_ec2_wake_session,
    natural_language_group_assignment,
    remove_from_group,
    start_ec2_once,
)
from deg.viz_helpers import render_deg_results_and_visualizations
from deg.browse_brb_by_metadata.group_selection_data_flow import render_filter_and_data_steps

DEG_API_URL = "http://18.218.84.81:8000/"
DEG_LAMBDA_URL = "https://hc5ycktqbvxywpf4f4xhxfvm2e0dpozl.lambda-url.us-east-2.on.aws/"


def _render_step_header(step_num: int, title: str, subtitle: str = "", optional: bool = False):
    if optional:
        border_color = "rgba(245, 158, 11, 0.55)"
        bg_color = "rgba(245, 158, 11, 0.10)"
        badge_text = "OPTIONAL"
    else:
        border_color = "rgba(34, 197, 94, 0.55)"
        bg_color = "rgba(34, 197, 94, 0.10)"
        badge_text = "REQUIRED"

    st.markdown(
        f"""
        <div style="
            margin-top: 1rem;
            margin-bottom: 0.75rem;
            padding: 0.75rem 0.9rem;
            border-radius: 10px;
            border: 1px solid {border_color};
            background: {bg_color};
        ">
            <div style="font-size:0.8rem; opacity:0.85; margin-bottom:0.1rem;">
                STEP {step_num} · {badge_text}
            </div>
            <div style="font-size:1.05rem; font-weight:600; margin-bottom:0.15rem;">
                {title}
            </div>
            <div style="font-size:0.88rem; opacity:0.8;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _keys_for_mode(mode: str) -> dict:
    if mode == "experiment":
        state_prefix = "deg_exp_"
        widget_prefix = "deg_exp_w_"
    else:
        state_prefix = "deg_novo_"
        widget_prefix = "deg_novo_w_"

    return {
        "state_prefix": state_prefix,
        "widget_prefix": widget_prefix,
        "metadata_df": f"{state_prefix}metadata_df",
        "group_a": f"{state_prefix}group_a",
        "group_b": f"{state_prefix}group_b",
        "filters": f"{state_prefix}filters",
        "editor_reset": f"{state_prefix}editor_reset",
        "nl_filter_input_reset": f"{state_prefix}nl_filter_input_reset",
        "nl_filters": f"{state_prefix}nl_filters",
        "llm_group_result": f"{state_prefix}llm_group_result",
        "dup_warning": f"{state_prefix}dup_warning",
        "heatmap_df": f"{state_prefix}heatmap_df",
        "heatmap_annotation_df": f"{state_prefix}heatmap_annotation_df",
        "heatmap_image": f"{state_prefix}heatmap_image",
        "gsea_df": f"{state_prefix}gsea_df",
        "results_df": f"{state_prefix}results_df",
        "counts_tmp": f"{state_prefix}counts_tmp",
        "job_id_tmp": f"{state_prefix}job_id_tmp",
        "visualize_data": f"{state_prefix}visualize_data",
    }


def _ensure_session(keys: dict):
    st.session_state.setdefault("deg_api_url", DEG_API_URL)
    st.session_state.setdefault("deg_schema", None)
    st.session_state.setdefault(keys["metadata_df"], None)
    st.session_state.setdefault(keys["group_a"], [])
    st.session_state.setdefault(keys["group_b"], [])
    st.session_state.setdefault(keys["filters"], {})
    st.session_state.setdefault(keys["editor_reset"], 0)
    st.session_state.setdefault(keys["nl_filter_input_reset"], 0)
    st.session_state.setdefault(keys["nl_filters"], {})
    st.session_state.setdefault(keys["llm_group_result"], None)
    st.session_state.setdefault(keys["dup_warning"], None)
    st.session_state.setdefault(keys["heatmap_df"], None)
    st.session_state.setdefault(keys["heatmap_annotation_df"], None)
    st.session_state.setdefault(keys["heatmap_image"], None)
    st.session_state.setdefault(keys["gsea_df"], None)
    st.session_state.setdefault(keys["results_df"], None)
    st.session_state.setdefault(keys["counts_tmp"], None)
    st.session_state.setdefault(keys["job_id_tmp"], None)
    st.session_state.setdefault(keys["visualize_data"], False)


def run_group_selection(mode: str = "novo"):
    apply_custom_theme()

    ensure_auth_session()
    if not st.session_state["authenticated"]:
        authenticate()
        return

    ensure_ec2_wake_session(DEG_API_URL, DEG_LAMBDA_URL)
    if not st.session_state.get("deg_ec2_start_triggered"):
        start_ec2_once()
    if not st.session_state.get("deg_fastapi_ready", "False"):
        st.info("Waking up the compute node… Please wait until it is ready to proceed.")
        check_health(DEG_API_URL.rstrip("/") + "/healthz")
        st_autorefresh(interval=8000, key="deg_wake_refresh")
        return

    keys = _keys_for_mode(mode)
    _ensure_session(keys)
    wk = lambda name: f"{keys['widget_prefix']}{name}"
    api_url = st.session_state["deg_api_url"]
    ga = st.session_state[keys["group_a"]]
    gb = st.session_state[keys["group_b"]]

    st.title("DEG Analysis: Group Selection")
    st.caption("Workflow: Filter samples -> Retrieve metadata -> Visualize/download -> Assign groups -> Run DEG")

    if st.session_state.get("deg_fastapi_ready", "False") and st.session_state.get("deg_schema") is None:
        try:
            r = requests.get(f"{api_url}get_metadata/", timeout=30)
            r.raise_for_status()
            st.session_state["deg_schema"] = r.json()
            st.success("Compute node is ready.")
            st.success("Schema loaded.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load schema: {e}")

    schema = st.session_state["deg_schema"]
    if schema is None:
        st.info("Click 'Load schema' to fetch metadata columns and filter options.")
        return

    meta_df = render_filter_and_data_steps(
        keys=keys,
        wk=wk,
        api_url=api_url,
        schema=schema,
        render_step_header=_render_step_header,
    )
    if meta_df is None:
        return

    _render_step_header(
        4,
        "Assign Samples to Comparison and Reference Groups",
        "Select rows in the table, then add them to either group.",
        optional=False,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select all rows", key=wk("select_all")):
            df = st.session_state[keys["metadata_df"]].copy()
            df["Select"] = True
            st.session_state[keys["metadata_df"]] = df
            st.session_state[keys["editor_reset"]] = st.session_state.get(keys["editor_reset"], 0) + 1
            st.rerun()
    with col2:
        if st.button("Clear selection", key=wk("clear_all")):
            df = st.session_state[keys["metadata_df"]].copy()
            df["Select"] = False
            st.session_state[keys["metadata_df"]] = df
            st.session_state[keys["editor_reset"]] = st.session_state.get(keys["editor_reset"], 0) + 1
            st.rerun()

    editor_key = f"{wk('data_editor')}_{st.session_state.get(keys['editor_reset'], 0)}"
    edited = st.data_editor(
        meta_df,
        column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)},
        use_container_width=True,
        key=editor_key,
        disabled=[c for c in meta_df.columns if c != "Select"],
    )

    if st.session_state.get(keys["dup_warning"]):
        st.warning(st.session_state[keys["dup_warning"]])
        st.session_state[keys["dup_warning"]] = None

    GROUP_WINDOW_HEIGHT = 220
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "Add selected → Comparison Group",
            on_click=add_to_group,
            args=(edited, keys["group_a"], keys["group_b"]),
            kwargs={
                "metadata_df_key": keys["metadata_df"],
                "editor_reset_key": keys["editor_reset"],
                "dup_warning_key": keys["dup_warning"],
            },
            key=wk("add_grp_a"),
        )
        st.markdown("**Comparison Group**")
        with st.container(height=GROUP_WINDOW_HEIGHT):
            if not ga:
                st.caption("No samples yet.")
            else:
                for i, s in enumerate(ga):
                    st.button(f"✕ {s}", key=wk(f"rm_a_{i}_{s}"), on_click=remove_from_group, args=(keys["group_a"], s))
        st.button(
            "Clear Comparison Group",
            on_click=clear_group,
            args=(keys["group_a"],),
            kwargs={"metadata_df_key": keys["metadata_df"], "editor_reset_key": keys["editor_reset"]},
            key=wk("clear_grp_a"),
        )
    with col2:
        st.button(
            "Add selected → Reference Group",
            on_click=add_to_group,
            args=(edited, keys["group_b"], keys["group_a"]),
            kwargs={
                "metadata_df_key": keys["metadata_df"],
                "editor_reset_key": keys["editor_reset"],
                "dup_warning_key": keys["dup_warning"],
            },
            key=wk("add_grp_b"),
        )
        st.markdown("**Reference Group**")
        with st.container(height=GROUP_WINDOW_HEIGHT):
            if not gb:
                st.caption("No samples yet.")
            else:
                for i, s in enumerate(gb):
                    st.button(f"✕ {s}", key=wk(f"rm_b_{i}_{s}"), on_click=remove_from_group, args=(keys["group_b"], s))
        st.button(
            "Clear Reference Group",
            on_click=clear_group,
            args=(keys["group_b"],),
            kwargs={"metadata_df_key": keys["metadata_df"], "editor_reset_key": keys["editor_reset"]},
            key=wk("clear_grp_b"),
        )

    st.caption(f"Comparison Group: {len(ga)} samples · Reference Group: {len(gb)} samples")

    with st.expander("Natural Language Group Assignment", expanded=True):
        st.divider()
        st.markdown("#### 🤖 Add Samples via AI Query")
        st.caption(
            "Describe the samples you want and which group to put them in. "
            "Example: *Add WT samples treated with Rotenone at 24h to Comparison Group*"
        )
        llm_query = st.text_area(
            "Query",
            key=wk("llm_group_query"),
            height=80,
            placeholder="e.g. Add TDP43 Vehicle controls to Reference Group",
            label_visibility="collapsed",
        )
        col_llm1, col_llm2 = st.columns([2, 1])
        with col_llm1:
            ask_btn = st.button("🤖 Ask AI to Add Samples", key=wk("llm_group_btn"))
        with col_llm2:
            if st.button("Clear AI result", key=wk("llm_clear_btn")):
                st.session_state.pop(keys["llm_group_result"], None)
                st.rerun()

        if ask_btn:
            if not llm_query.strip():
                st.warning("Please enter a query first.")
            else:
                with st.spinner("Asking AI…"):
                    result = natural_language_group_assignment(
                        llm_query,
                        st.session_state[keys["metadata_df"]],
                        st.session_state["deg_schema"],
                    )
                if result["error"]:
                    st.error(f"AI error: {result['error']}")
                else:
                    res_a = result["group_a"]
                    res_b = result["group_b"]
                    added_a, added_b = [], []
                    if res_a["samples"]:
                        added_a = add_samples_to_group(
                            res_a["samples"],
                            keys["group_a"],
                            keys["group_b"],
                            dup_warning_key=keys["dup_warning"],
                        ) or []
                    if res_b["samples"]:
                        added_b = add_samples_to_group(
                            res_b["samples"],
                            keys["group_b"],
                            keys["group_a"],
                            dup_warning_key=keys["dup_warning"],
                        ) or []
                    st.session_state[keys["llm_group_result"]] = {
                        "group_a": {"filters": res_a["filters"], "matched": res_a["samples"], "added": added_a},
                        "group_b": {"filters": res_b["filters"], "matched": res_b["samples"], "added": added_b},
                    }
                    st.rerun()

        llm_result = st.session_state.get(keys["llm_group_result"])
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

    _render_step_header(
        5,
        "Run DEG Analysis",
        "Submit both groups to backend and generate DEG results and visualizations.",
        optional=False,
    )
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
                    buf = io.BytesIO(resp.content)
                    st.session_state[keys["heatmap_df"]] = None
                    st.session_state[keys["heatmap_annotation_df"]] = None
                    st.session_state[keys["gsea_df"]] = None
                    st.session_state.pop(f"{keys['widget_prefix']}heatmap_render_sig", None)
                    st.session_state.pop(f"{keys['widget_prefix']}heatmap_render_png", None)
                    try:
                        with zipfile.ZipFile(buf, "r") as zf:
                            st.session_state[keys["results_df"]] = pd.read_feather(io.BytesIO(zf.read("deg_analysis.feather")))
                            if "heatmap_matrix.csv" in zf.namelist():
                                st.session_state[keys["heatmap_df"]] = pd.read_csv(io.BytesIO(zf.read("heatmap_matrix.csv")), index_col=0)
                            if "heatmap_annotation.csv" in zf.namelist():
                                st.session_state[keys["heatmap_annotation_df"]] = pd.read_csv(io.BytesIO(zf.read("heatmap_annotation.csv")))
                            if "deg_heatmap.png" in zf.namelist():
                                st.session_state[keys["heatmap_image"]] = io.BytesIO(zf.read("deg_heatmap.png"))
                            if "gsea_results.csv" in zf.namelist():
                                st.session_state[keys["gsea_df"]] = pd.read_csv(io.BytesIO(zf.read("gsea_results.csv")))
                    except zipfile.BadZipFile:
                        buf.seek(0)
                        st.session_state[keys["results_df"]] = pd.read_feather(buf)
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    try:
                        err_msg = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
                    except Exception:
                        err_msg = str(e)
                    st.error(f"Request failed: {err_msg}")
                except Exception as e:
                    st.error(f"Submit failed: {e}")

    if st.session_state.get(keys["results_df"]) is not None:
        render_deg_results_and_visualizations(
            st.session_state[keys["results_df"]],
            state_prefix=keys["state_prefix"],
            widget_prefix=keys["widget_prefix"],
            group_a=st.session_state.get(keys["group_a"], []),
            group_b=st.session_state.get(keys["group_b"], []),
        )


__all__ = ["run_group_selection"]

