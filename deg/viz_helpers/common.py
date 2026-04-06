"""Shared constants and helpers for DEG visualizations."""

import streamlit as st
import pandas as pd

MIN_LFC = 0.6
MAX_FDR = 0.05

DEG_API_URL = "http://18.218.84.81:8000/"


def fig_to_html(fig, width=1400, height=None):
    """Export Plotly figure to interactive HTML at full display size."""
    try:
        if height is None:
            height = int(fig.layout.height or 500)
        fig.update_layout(width=width, height=height)
        return fig.to_html(include_plotlyjs="cdn")
    except Exception:
        return None


def valid_display_label(row):
    gene_val = row.get("gene", "")
    if pd.isna(gene_val):
        gene_val = ""
    gene_str = str(gene_val).strip()
    if "SYMBOL" in row.index:
        sym = row["SYMBOL"]
        if sym is not None and str(sym).strip().lower() not in ("", "nan", "na", "none"):
            return str(sym).strip()
    return gene_str if gene_str else "—"


def ensure_research_session(state_prefix: str):
    """Initialize session state for research pipeline."""
    st.session_state.setdefault(f"{state_prefix}research_disease_context", "Unknown")
    st.session_state.setdefault(f"{state_prefix}research_tissue", "Unknown")
    st.session_state.setdefault(f"{state_prefix}research_num_genes", 10)
    st.session_state.setdefault(f"{state_prefix}research_results", None)
    st.session_state.setdefault(f"{state_prefix}research_loading", False)
    st.session_state.setdefault(f"{state_prefix}comparison_description", "")
