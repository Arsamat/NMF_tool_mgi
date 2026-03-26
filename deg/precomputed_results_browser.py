"""
Browse precomputed DEG results stored in MongoDB / S3.

Navigation flow (driven by st.session_state["deg_pre_view"]):
  "experiments"  → table of all experiments from the results map
  "groups"       → groups for the selected experiment
  "results"      → interaction-term selector + DEG table/plots
"""

import io
import zipfile

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
import numpy as np
import plotly.express as px
from deg.viz_helpers import MIN_LFC, MAX_FDR, _fig_to_html, _valid_display_label
from deg.plot_helpers import place_labels_no_overlap

DEG_API_URL = "http://18.218.84.81:8000/"
DEG_LAMBDA_URL = "https://hc5ycktqbvxywpf4f4xhxfvm2e0dpozl.lambda-url.us-east-2.on.aws/"


def render_table_and_volcano(deg_df, widget_prefix: str = "deg_pre_"):
    """DEG results table + volcano plot only."""
    # --- Table ---
    st.subheader("DEG results")
    pval_cols = {
        col: st.column_config.NumberColumn(col, format="%.2e")
        for col in ["P.Value", "adj.P.Val", "pvalue", "padj", "p.adjust"]
        if col in deg_df.columns
    }
    st.dataframe(deg_df, use_container_width=True, column_config=pval_cols if pval_cols else None)
    st.download_button(
        "Download DEG table (CSV)",
        deg_df.to_csv(index=False),
        "deg_results.csv",
        "text/csv",
        key=f"{widget_prefix}download",
    )

    if "logFC" not in deg_df.columns or "P.Value" not in deg_df.columns:
        return

    # --- Prepare ---
    deg_df = deg_df.copy()
    deg_df["neg_log10_p"] = -np.log10(deg_df["P.Value"].clip(lower=1e-300))
    deg_df["Significant"] = "NS"
    if "adj.P.Val" in deg_df.columns:
        deg_df.loc[(deg_df["logFC"] >= MIN_LFC) & (deg_df["adj.P.Val"] < MAX_FDR), "Significant"] = "Up"
        deg_df.loc[(deg_df["logFC"] <= -MIN_LFC) & (deg_df["adj.P.Val"] < MAX_FDR), "Significant"] = "Down"
    else:
        deg_df.loc[(deg_df["logFC"] >= MIN_LFC) & (deg_df["P.Value"] < MAX_FDR), "Significant"] = "Up"
        deg_df.loc[(deg_df["logFC"] <= -MIN_LFC) & (deg_df["P.Value"] < MAX_FDR), "Significant"] = "Down"

    deg_df["display_label"] = deg_df.apply(_valid_display_label, axis=1)
    sig = deg_df[deg_df["Significant"] != "NS"].sort_values("P.Value")
    top10 = sig.head(10)
    gene_options = sorted([
        str(x).strip() for x in deg_df["display_label"].dropna().unique()
        if str(x).strip() and str(x).strip() != "—"
    ])

    # --- Volcano ---
    with st.expander("Volcano plot", expanded=True):
        st.subheader("Volcano plot (top 10 DEGs labeled)")
        volcano_df = deg_df.copy()
        selected_genes = st.multiselect(
            "Select genes (type to search/filter):",
            options=gene_options,
            default=[],
            key=f"{widget_prefix}volcano_gene_select",
            placeholder="Type gene name to search…",
        )
        volcano_df["label"] = ""
        volcano_df["label_source"] = ""
        if len(top10) > 0:
            volcano_df.loc[top10.index, "label"] = top10["display_label"].values
            volcano_df.loc[top10.index, "label_source"] = "top10"
        if selected_genes:
            mask = volcano_df["display_label"].isin(selected_genes)
            volcano_df.loc[mask, "label"] = volcano_df.loc[mask, "display_label"].values
            volcano_df.loc[mask, "label_source"] = "selected"

        fig = px.scatter(
            volcano_df, x="logFC", y="neg_log10_p", color="Significant",
            hover_data=[c for c in ["display_label", "logFC", "P.Value", "adj.P.Val"] if c in volcano_df.columns],
            color_discrete_map={"Up": "#e74c3c", "Down": "#3498db", "NS": "#95a5a6"},
        )
        fig.update_traces(marker=dict(size=6))

        if volcano_df["label"].str.len().gt(0).any():
            labeled = volcano_df[volcano_df["label"] != ""].copy()
            x_min, x_max = volcano_df["logFC"].min(), volcano_df["logFC"].max()
            y_min, y_max = volcano_df["neg_log10_p"].min(), volcano_df["neg_log10_p"].max()
            x_pad = max(0.1 * (x_max - x_min), 0.1)
            y_pad = max(0.1 * (y_max - y_min), 0.1)
            offsets = place_labels_no_overlap(
                labeled["logFC"].values, labeled["neg_log10_p"].values,
                labeled["label"].tolist(),
                x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad,
            )
            for i, row in enumerate(labeled.itertuples()):
                ax, ay = offsets[i]
                fig.add_annotation(
                    x=row.logFC, y=row.neg_log10_p, text=row.label,
                    showarrow=True, arrowhead=1, ax=ax, ay=ay,
                    font=dict(size=11, color="#1a1a1a"),
                    arrowcolor="#16a085" if getattr(row, "label_source", "") == "selected" else "#444",
                    arrowsize=0.8,
                )

        fig.update_layout(
            xaxis_title="log₂ Fold Change",
            yaxis_title="-log₁₀ P-value",
            height=500, showlegend=True,
            margin=dict(l=70, r=70, t=50, b=70),
        )
        st.caption("Label colors: **top 10 DEGs** = dark gray; **genes you select** = teal.")
        st.plotly_chart(fig, use_container_width=True)
        html = _fig_to_html(fig, width=1400, height=500)
        if html:
            st.download_button(
                "Download volcano plot (HTML)",
                html, "deg_volcano.html", "text/html",
                key=f"{widget_prefix}volcano_html_download",
            )


def ensure_session():
    st.session_state.setdefault("deg_pre_view", "experiments")
    st.session_state.setdefault("deg_pre_experiment", None)
    st.session_state.setdefault("deg_pre_group_data", None)
    st.session_state.setdefault("deg_pre_terms", None)
    st.session_state.setdefault("deg_pre_deg_df", None)
    st.session_state.setdefault("deg_pre_barplot", None)
    st.session_state.setdefault("deg_pre_loaded_key", None)


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

    view = st.session_state["deg_pre_view"]
    if view == "groups":
        render_groups_view(api_url)
    elif view == "results":
        render_results_view(api_url)


# ---------------------------------------------------------------------------
# Level 2: groups for the selected experiment
# ---------------------------------------------------------------------------

def render_groups_view(api_url: str):
    experiment = st.session_state.get("deg_pre_experiment", "")

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

    # Build a display dataframe with truncated sample names
    display_rows = []
    for g in groups:
        sample_names = str(g.get("sample_names", ""))
        sample_preview = sample_names[:100] + "…" if len(sample_names) > 100 else sample_names
        display_rows.append({
            "Group": g.get("group", ""),
            "Context": g.get("context", ""),
            "Model Design": g.get("model_design", ""),
            "N Samples": g.get("n_samples", ""),
            "Sample Names": sample_preview,
        })

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
            st.session_state["deg_pre_group_data"] = groups[idx]
            st.session_state["deg_pre_terms"] = None
            st.session_state["deg_pre_deg_df"] = None
            st.session_state["deg_pre_loaded_key"] = None
            st.session_state["deg_pre_view"] = "results"
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
    experiment = st.session_state.get("deg_pre_experiment", "")
    group_data = st.session_state.get("deg_pre_group_data") or {}
    group = group_data.get("group", "")

    if st.button("← Back to groups", key="deg_pre_back_to_groups"):
        st.session_state["deg_pre_view"] = "groups"
        st.session_state["deg_pre_deg_df"] = None
        st.session_state["deg_pre_loaded_key"] = None
        st.rerun()

    st.title(f"{experiment} — {group}")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Context", group_data.get("context", "—"))
    with col2:
        st.metric("N Samples", group_data.get("n_samples", "—"))
    with col3:
        st.metric("Group", group)
    st.caption(f"Model design: `{group_data.get('model_design', '')}`")

    with st.expander("Sample names"):
        st.write(group_data.get("sample_names", "—"))

    st.divider()

    # Load available terms (cached in session state)
    if st.session_state.get("deg_pre_terms") is None:
        try:
            resp = requests.post(
                f"{api_url}deg_results/terms",
                json={"experiment": experiment, "group": group},
                timeout=30,
            )
            resp.raise_for_status()
            terms = resp.json().get("terms", [])
            st.session_state["deg_pre_terms"] = terms
        except Exception as exc:
            st.error(f"Could not load interaction terms: {exc}")
            return

    terms = st.session_state.get("deg_pre_terms") or []
    if not terms:
        st.warning("No DEG result files found for this group (de_csv_exists may be False for all terms).")
        return

    term_labels = [t["term_label"] for t in terms]
    selected_term = st.selectbox(
        "Select interaction term",
        term_labels,
        key="deg_pre_term_selectbox",
    )

    if selected_term is None:
        return

    term_entry = next((t for t in terms if t["term_label"] == selected_term), None)
    if term_entry is None:
        return

    # Fetch results from S3 (only when term or group changes)
    cache_key = f"{experiment}|{group}|{selected_term}"
    if (
        st.session_state.get("deg_pre_deg_df") is None
        or st.session_state.get("deg_pre_loaded_key") != cache_key
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

                st.session_state["deg_pre_deg_df"] = deg_df
                st.session_state["deg_pre_barplot"] = barplot_bytes
                st.session_state["deg_pre_loaded_key"] = cache_key
            except Exception as exc:
                st.error(f"Could not fetch DEG results: {exc}")
                return

    deg_df = st.session_state.get("deg_pre_deg_df")
    if deg_df is None or deg_df.empty:
        st.warning("DEG results table is empty.")
        return

    render_table_and_volcano(deg_df, widget_prefix=f"deg_pre_{selected_term}_")

    barplot_bytes = st.session_state.get("deg_pre_barplot")
    if barplot_bytes:
        st.subheader("GSEA barplot")
        st.image(barplot_bytes, use_container_width=True)
