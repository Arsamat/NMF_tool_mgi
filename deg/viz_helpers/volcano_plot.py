"""Volcano plot for DEG results."""

import streamlit as st
import plotly.express as px

from deg.plot_helpers import place_labels_no_overlap
from deg.viz_helpers.common import fig_to_html


def render_volcano_expander(deg_df, gene_options, top10, widget_prefix: str):
    with st.expander("Volcano plot", expanded=True):
        st.subheader("Volcano plot (top 10 DEGs labeled)")
        volcano_df = deg_df.copy()
        selected_genes_volcano = st.multiselect(
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
        if selected_genes_volcano:
            mask = volcano_df["display_label"].isin(selected_genes_volcano)
            volcano_df.loc[mask, "label"] = volcano_df.loc[mask, "display_label"].values
            volcano_df.loc[mask, "label_source"] = "selected"

        fig_volcano = px.scatter(
            volcano_df,
            x="logFC",
            y="neg_log10_p",
            color="Significant",
            hover_data=[c for c in ["display_label", "logFC", "P.Value", "adj.P.Val"] if c in volcano_df.columns],
            color_discrete_map={"Up": "#e74c3c", "Down": "#3498db", "NS": "#95a5a6"},
        )
        fig_volcano.update_traces(marker=dict(size=6))
        if "label" in volcano_df.columns and volcano_df["label"].str.len().gt(0).any():
            labeled = volcano_df[volcano_df["label"] != ""].copy()
            if len(labeled) > 0:
                x_min = volcano_df["logFC"].min()
                x_max = volcano_df["logFC"].max()
                y_min = volcano_df["neg_log10_p"].min()
                y_max = volcano_df["neg_log10_p"].max()
                x_pad = max(0.1 * (x_max - x_min), 0.1)
                y_pad = max(0.1 * (y_max - y_min), 0.1)
                x_min, x_max = x_min - x_pad, x_max + x_pad
                y_min, y_max = y_min - y_pad, y_max + y_pad
                offsets = place_labels_no_overlap(
                    labeled["logFC"].values,
                    labeled["neg_log10_p"].values,
                    labeled["label"].tolist(),
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                )
                color_top10 = "#1a1a1a"
                color_selected = "#16a085"
                for i in range(len(labeled)):
                    row = labeled.iloc[i]
                    ax, ay = offsets[i]
                    src = row.get("label_source", "top10")
                    font_color = color_top10
                    arrow_color = color_selected if src == "selected" else "#444"
                    fig_volcano.add_annotation(
                        x=row["logFC"],
                        y=row["neg_log10_p"],
                        text=row["label"],
                        showarrow=True,
                        arrowhead=1,
                        ax=ax,
                        ay=ay,
                        font=dict(size=11, color=font_color),
                        arrowcolor=arrow_color,
                        arrowsize=0.8,
                    )
        fig_volcano.update_layout(
            xaxis_title="log₂ Fold Change",
            yaxis_title="-log₁₀ P-value",
            height=500,
            showlegend=True,
            margin=dict(l=70, r=70, t=50, b=70),
        )
        st.caption("Label colors: **top 10 DEGs** = dark gray; **genes you select** = teal.")
        st.plotly_chart(fig_volcano, use_container_width=True)
        volcano_html = fig_to_html(fig_volcano, width=1400, height=int(fig_volcano.layout.height or 500))
        if volcano_html:
            st.download_button(
                "Download volcano plot (HTML)",
                volcano_html,
                "deg_volcano.html",
                "text/html",
                key=f"{widget_prefix}volcano_html_download",
            )
