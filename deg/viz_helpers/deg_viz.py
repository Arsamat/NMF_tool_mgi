"""Orchestrates DEG table and all result visualizations."""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from deg.viz_helpers.common import (
    MAX_FDR,
    MIN_LFC,
    ensure_research_session,
    valid_display_label,
)
from deg.viz_helpers.gsea_plot import render_gsea_expander
from deg.viz_helpers.heatmap import render_heatmap_expander
from deg.viz_helpers.ma_plot import render_ma_expander
from deg.viz_helpers.research_pipeline import render_research_section
from deg.viz_helpers.volcano_plot import render_volcano_expander


def render_deg_results_and_visualizations(
    deg_results_df,
    state_prefix: str = "deg_",
    widget_prefix: str = "deg_",
    group_a: list | None = None,
    group_b: list | None = None,
):
    """
    Render DEG results table, download button, volcano plot, MA plot,
    heatmap, GSEA section, and research LLM pipeline.
    Uses st.session_state for heatmap, GSEA, and research data.

    group_a / group_b: sample ids for the comparison; required to render the heatmap on the backend.
    """
    ensure_research_session(state_prefix)
    deg_df = deg_results_df

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

    deg_df = deg_df.copy()
    deg_df["neg_log10_p"] = -np.log10(deg_df["P.Value"].clip(lower=1e-300))
    deg_df["Significant"] = "NS"
    if "adj.P.Val" in deg_df.columns:
        deg_df.loc[(deg_df["logFC"] >= MIN_LFC) & (deg_df["adj.P.Val"] < MAX_FDR), "Significant"] = "Up"
        deg_df.loc[(deg_df["logFC"] <= -MIN_LFC) & (deg_df["adj.P.Val"] < MAX_FDR), "Significant"] = "Down"
    else:
        deg_df.loc[(deg_df["logFC"] >= MIN_LFC) & (deg_df["P.Value"] < MAX_FDR), "Significant"] = "Up"
        deg_df.loc[(deg_df["logFC"] <= -MIN_LFC) & (deg_df["P.Value"] < MAX_FDR), "Significant"] = "Down"

    deg_df["display_label"] = deg_df.apply(valid_display_label, axis=1)

    sig = deg_df[deg_df["Significant"] != "NS"].sort_values("P.Value")
    top10 = sig.head(10) if len(sig) >= 10 else sig
    gene_options = deg_df["display_label"].dropna().unique().tolist()
    gene_options = [str(x).strip() for x in gene_options if str(x).strip() and str(x).strip() != "—"]
    gene_options = sorted(gene_options)

    render_volcano_expander(deg_df, gene_options, top10, widget_prefix)
    render_ma_expander(deg_df, gene_options, top10, widget_prefix)
    render_heatmap_expander(state_prefix, widget_prefix, group_a=group_a, group_b=group_b)
    render_gsea_expander(state_prefix, widget_prefix)
    render_research_section(deg_df, state_prefix, widget_prefix)
