"""GSEA barplot for DEG results."""

import streamlit as st
import numpy as np
import plotly.express as px

from deg.viz_helpers.common import fig_to_html


def render_gsea_expander(state_prefix: str, widget_prefix: str):
    with st.expander("GSEA", expanded=True):
        gsea_df = st.session_state.get(f"{state_prefix}gsea_df")
        if gsea_df is None or gsea_df.empty:
            return
        st.subheader("Enriched gene sets (GSEA Hallmark)")
        id_col = (
            "ID"
            if "ID" in gsea_df.columns
            else "Description"
            if "Description" in gsea_df.columns
            else gsea_df.columns[0]
        )
        desc_col = "Description" if "Description" in gsea_df.columns else id_col
        p_col = (
            "p.adjust"
            if "p.adjust" in gsea_df.columns
            else "pvalue"
            if "pvalue" in gsea_df.columns
            else None
        )
        st.dataframe(gsea_df, use_container_width=True)
        if p_col and (id_col in gsea_df.columns or desc_col in gsea_df.columns):
            plot_df = gsea_df.head(20).copy()
            plot_df["neg_log10_p"] = -np.log10(plot_df[p_col].clip(lower=1e-300))
            y_label = desc_col if desc_col in plot_df.columns else id_col
            fig_gsea = px.bar(
                plot_df,
                x="neg_log10_p",
                y=plot_df[y_label],
                orientation="h",
                title="Top 20 enriched sets (−log₁₀ p.adjust)",
            )
            fig_gsea.update_layout(height=max(350, 25 * min(20, len(plot_df))), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_gsea, use_container_width=True)
            gsea_html = fig_to_html(fig_gsea, width=1400, height=int(fig_gsea.layout.height or 450))
            if gsea_html:
                st.download_button(
                    "Download GSEA barplot (HTML)",
                    gsea_html,
                    "deg_gsea_barplot.html",
                    "text/html",
                    key=f"{widget_prefix}gsea_html_download",
                )
