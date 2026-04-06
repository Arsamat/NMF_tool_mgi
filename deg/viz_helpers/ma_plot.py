"""MA plot for DEG results."""

import streamlit as st
import plotly.express as px

from deg.plot_helpers import place_labels_no_overlap
from deg.viz_helpers.common import fig_to_html


def render_ma_expander(deg_df, gene_options, top10, widget_prefix: str):
    with st.expander("MA plot", expanded=True):
        if "AveExpr" not in deg_df.columns:
            return
        st.subheader("MA plot (AveExpr vs logFC)")
        ma_df = deg_df.copy()
        selected_genes_ma = st.multiselect(
            "Select genes for MA plot (type to search/filter):",
            options=gene_options,
            default=[],
            key=f"{widget_prefix}ma_gene_select",
            placeholder="Type gene name to search…",
        )
        ma_df["label"] = ""
        ma_df["label_source"] = ""
        if len(top10) > 0:
            ma_df.loc[top10.index, "label"] = top10["display_label"].values
            ma_df.loc[top10.index, "label_source"] = "top10"
        if selected_genes_ma:
            mask = ma_df["display_label"].isin(selected_genes_ma)
            ma_df.loc[mask, "label"] = ma_df.loc[mask, "display_label"].values
            ma_df.loc[mask, "label_source"] = "selected"

        fig_ma = px.scatter(
            ma_df,
            x="AveExpr",
            y="logFC",
            color="Significant" if "Significant" in ma_df.columns else None,
            hover_data=[c for c in ["display_label", "logFC", "P.Value"] if c in ma_df.columns],
            color_discrete_map={"Up": "#e74c3c", "Down": "#3498db", "NS": "#95a5a6"}
            if "Significant" in ma_df.columns
            else None,
        )
        fig_ma.update_traces(marker=dict(size=5))
        if "label" in ma_df.columns and ma_df["label"].str.len().gt(0).any():
            labeled_ma = ma_df[ma_df["label"] != ""].copy()
            if len(labeled_ma) > 0:
                x_min = ma_df["AveExpr"].min()
                x_max = ma_df["AveExpr"].max()
                y_min = ma_df["logFC"].min()
                y_max = ma_df["logFC"].max()
                x_pad = max(0.1 * (x_max - x_min), 0.1)
                y_pad = max(0.1 * (y_max - y_min), 0.1)
                x_min, x_max = x_min - x_pad, x_max + x_pad
                y_min, y_max = y_min - y_pad, y_max + y_pad
                offsets_ma = place_labels_no_overlap(
                    labeled_ma["AveExpr"].values,
                    labeled_ma["logFC"].values,
                    labeled_ma["label"].tolist(),
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                )
                color_top10 = "#1a1a1a"
                color_selected = "#16a085"
                for i in range(len(labeled_ma)):
                    row = labeled_ma.iloc[i]
                    ax, ay = offsets_ma[i]
                    src = row.get("label_source", "top10")
                    font_color = color_top10
                    arrow_color = color_selected if src == "selected" else "#444"
                    fig_ma.add_annotation(
                        x=row["AveExpr"],
                        y=row["logFC"],
                        text=row["label"],
                        showarrow=True,
                        arrowhead=1,
                        ax=ax,
                        ay=ay,
                        font=dict(size=11, color=font_color),
                        arrowcolor=arrow_color,
                        arrowsize=0.8,
                    )
        fig_ma.update_layout(
            xaxis_title="Average expression (log₂)",
            yaxis_title="log₂ Fold Change",
            height=450,
            margin=dict(l=70, r=70, t=50, b=70),
        )
        st.caption("Label colors: **top 10 DEGs** = dark gray; **genes you select** = teal.")
        st.plotly_chart(fig_ma, use_container_width=True)
        ma_html = fig_to_html(fig_ma, width=1400, height=int(fig_ma.layout.height or 450))
        if ma_html:
            st.download_button(
                "Download MA plot (HTML)",
                ma_html,
                "deg_ma_plot.html",
                "text/html",
                key=f"{widget_prefix}ma_html_download",
            )
