"""
Browse precomputed DEG results stored in MongoDB / S3.

Navigation flow (driven by st.session_state["deg_precomputed_view"]):
  "groups"       → groups for the selected experiment
  "results"      → interaction-term selector + DEG table/plots
"""

import io
import zipfile
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from deg.precomputed_results_browser_helpers import (
    render_precomputed_deg_term_sidebar,
    render_table_and_volcano,
)

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


def ensure_session():
    st.session_state.setdefault("deg_precomputed_view", "groups")
    st.session_state.setdefault("deg_precomputed_experiment", None)
    st.session_state.setdefault("deg_precomputed_group_data", None)
    st.session_state.setdefault("deg_precomputed_terms", None)
    st.session_state.setdefault("deg_precomputed_deg_df", None)
    st.session_state.setdefault("deg_precomputed_barplot", None)
    st.session_state.setdefault("deg_precomputed_loaded_key", None)
    st.session_state.setdefault("deg_precomputed_selected_term", None)


def run_precomputed_results_browser():
    apply_custom_theme()

    ensure_auth_session()
    if not st.session_state["authenticated"]:
        authenticate()
        return

    ensure_ec2_wake_session(DEG_API_URL, DEG_LAMBDA_URL)
    if not st.session_state.get("deg_ec2_start_triggered"):
        start_ec2_once()

    if not st.session_state.get("deg_fastapi_ready", False):
        st.info("Waking up the compute node… Please wait until it is ready.")
        check_health(DEG_API_URL.rstrip("/") + "/healthz")
        st_autorefresh(interval=8000, key="deg_pre_wake_refresh")
        return

    ensure_session()
    api_url = st.session_state.get("deg_api_url", DEG_API_URL)

    view = st.session_state["deg_precomputed_view"]
    if view == "groups":
        render_groups_view(api_url)
    elif view == "results":
        render_results_view(api_url)


# ---------------------------------------------------------------------------
# Level 2: groups for the selected experiment
# ---------------------------------------------------------------------------

def render_groups_view(api_url: str):
    experiment = st.session_state.get("deg_precomputed_experiment", "")

    st.title(f"Groups: {experiment}")
    st.caption(
        "Each row is a unique comparison group. "
        "Click a row to view available interaction terms and their DEG results."
    )

    try:
        resp = requests.post(
            f"{api_url}deg_results/groups",
            json={"experiment": experiment},
            timeout=30,
        )
        resp.raise_for_status()
        groups = resp.json().get("groups", [])
    except Exception as exc:
        st.error(f"Could not load groups: {exc}")
        return

    if not groups:
        st.warning("No groups found for this experiment.")
        return

    # Build a display dataframe with one row per returned (group, context)
    display_rows = []
    row_sources = []
    for g in groups:
        sample_names = str(g.get("sample_names", ""))
        sample_preview = sample_names[:100] + "…" if len(sample_names) > 100 else sample_names
        context = g.get("context", "")
        if isinstance(context, list):
            context = ", ".join([str(c) for c in context if c])
        display_rows.append({
            "Group": g.get("group", ""),
            "Context": context,
            "Model Design": g.get("model_design", ""),
            "N Samples": g.get("n_samples", ""),
            "Sample Names": sample_preview,
        })
        row_sources.append(g)

    display_df = pd.DataFrame(display_rows)

    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Group": st.column_config.TextColumn("Group", width="small"),
            "Context": st.column_config.TextColumn("Context", width="small"),
            "Model Design": st.column_config.TextColumn("Model Design", width="medium"),
            "N Samples": st.column_config.NumberColumn("N Samples", width="small"),
            "Sample Names": st.column_config.TextColumn("Sample Names", width="large"),
        },
    )

    try:
        sel_rows = getattr(selection, "selection", {}).get("rows", [])
        if sel_rows:
            idx = int(sel_rows[0])
            st.session_state["deg_precomputed_group_data"] = row_sources[idx]
            st.session_state["deg_precomputed_terms"] = None
            st.session_state["deg_precomputed_deg_df"] = None
            st.session_state["deg_precomputed_loaded_key"] = None
            st.session_state["deg_precomputed_selected_term"] = None
            st.session_state["deg_precomputed_view"] = "results"
            st.rerun()
    except Exception:
        pass

    # Full sample names in an expander
    with st.expander("View full sample names for all groups"):
        for g in groups:
            st.markdown(f"**{g.get('group', '')}** ({g.get('context', '')})")
            st.caption(g.get("sample_names", ""))


# ---------------------------------------------------------------------------
# Level 3: interaction-term selector + DEG results
# ---------------------------------------------------------------------------

def render_results_view(api_url: str):
    experiment = st.session_state.get("deg_precomputed_experiment", "")
    group_data = st.session_state.get("deg_precomputed_group_data") or {}
    group = group_data.get("group", "")
    context = group_data.get("context", "")

    if st.button("← Back to groups", key="deg_pre_back_to_groups"):
        st.session_state["deg_precomputed_view"] = "groups"
        st.session_state["deg_precomputed_deg_df"] = None
        st.session_state["deg_precomputed_loaded_key"] = None
        st.session_state["deg_precomputed_selected_term"] = None
        st.rerun()

    st.title(f"{experiment} — {group}")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("N Samples", group_data.get("n_samples", "—"))
    with col2:
        st.metric("Group", group)
    with col3:
        st.metric("Context", context or "—")
    st.metric("Model design", group_data.get('model_design', '-'))

    with st.expander("Sample names"):
        st.write(group_data.get("sample_names", "—"))

    st.divider()

    # Load available terms (cached in session state)
    if st.session_state.get("deg_precomputed_terms") is None:
        try:
            resp = requests.post(
                f"{api_url}deg_results/terms",
                json={"experiment": experiment, "group": group, "context": context},
                timeout=30,
            )
            resp.raise_for_status()
            terms = resp.json().get("terms", [])
            st.session_state["deg_precomputed_terms"] = terms
        except Exception as exc:
            st.error(f"Could not load interaction terms: {exc}")
            return

    terms = st.session_state.get("deg_precomputed_terms") or []
    if not terms:
        st.warning("No DEG result files found for this group (de_csv_exists may be False for all terms).")
        return
    
    with st.sidebar:
        selected_term = render_precomputed_deg_term_sidebar(
            terms,
            group_data.get("model_design", "") or "",
        )

    if selected_term is None:
        return

    term_entry = next((t for t in terms if t["term_label"] == selected_term), None)
    if term_entry is None:
        return

    # Fetch results from S3 (only when term or group changes)
    cache_key = f"{experiment}|{group}|{context}|{selected_term}"
    if (
        st.session_state.get("deg_precomputed_deg_df") is None
        or st.session_state.get("deg_precomputed_loaded_key") != cache_key
    ):
        with st.spinner("Fetching DEG results from database…"):
            try:
                resp = requests.post(
                    f"{api_url}deg_results/fetch_csv",
                    json={
                        "experiment": experiment,
                        "output_dir": term_entry["output_dir"],
                        "de_csv_path": term_entry["de_csv_path"],
                        "gsea_barplot_path": term_entry.get("gsea_barplot_path", ""),
                    },
                    timeout=60,
                )
                if not resp.ok:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    st.error(f"Backend error ({resp.status_code}): {detail}")
                    return

                # Unzip response
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    deg_df = pd.read_csv(z.open("deg_results.csv"))
                    deg_df = deg_df.rename(columns={"gene_symbol": "SYMBOL"})
                    barplot_bytes = z.read("gsea_barplot.png") if "gsea_barplot.png" in z.namelist() else None

                st.session_state["deg_precomputed_deg_df"] = deg_df
                st.session_state["deg_precomputed_barplot"] = barplot_bytes
                st.session_state["deg_precomputed_loaded_key"] = cache_key
            except Exception as exc:
                st.error(f"Could not fetch DEG results: {exc}")
                return

    deg_df = st.session_state.get("deg_precomputed_deg_df")
    if deg_df is None or deg_df.empty:
        st.warning("DEG results table is empty.")
        return

    render_table_and_volcano(deg_df, widget_prefix=f"deg_pre_{selected_term}_")

    barplot_bytes = st.session_state.get("deg_precomputed_barplot")
    if barplot_bytes:
        st.subheader("GSEA barplot")
        st.image(barplot_bytes, use_container_width=True)
