"""
DEG result table and visualizations: volcano, MA plot, heatmap, GSEA.
Renders in Streamlit when called from the group selection page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from deg.plot_helpers import place_labels_no_overlap

MIN_LFC = 0.6
MAX_FDR = 0.05


def _valid_display_label(row):
    gene_val = row.get("gene", "")
    if pd.isna(gene_val):
        gene_val = ""
    gene_str = str(gene_val).strip()
    if "SYMBOL" in row.index:
        sym = row["SYMBOL"]
        if pd.notna(sym) and str(sym).strip().lower() not in ("", "nan", "na", "none"):
            return str(sym).strip()
    return gene_str if gene_str else "—"


def render_deg_results_and_visualizations(deg_results_df):
    """
    Render DEG results table, download button, volcano plot, MA plot,
    heatmap, and GSEA section. Uses st.session_state for heatmap and GSEA data.
    """
    deg_df = deg_results_df

    st.subheader("DEG results")
    st.dataframe(deg_df, use_container_width=True)
    st.download_button(
        "Download DEG table (CSV)",
        deg_df.to_csv(index=False),
        "deg_results.csv",
        "text/csv",
        key="deg_download",
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

    deg_df["display_label"] = deg_df.apply(_valid_display_label, axis=1)

    sig = deg_df[deg_df["Significant"] != "NS"].sort_values("P.Value")
    top10 = sig.head(10) if len(sig) >= 10 else sig
    gene_options = deg_df["display_label"].dropna().unique().tolist()
    gene_options = [str(x).strip() for x in gene_options if str(x).strip() and str(x).strip() != "—"]
    gene_options = sorted(gene_options)

    # --- Volcano plot ---
    st.subheader("Volcano plot (top 10 DEGs labeled)")
    volcano_df = deg_df.copy()
    selected_genes_volcano = st.multiselect(
        "Select genes (type to search/filter):",
        options=gene_options,
        default=[],
        key="deg_volcano_gene_select",
        placeholder="Type gene name to search…",
    )
    volcano_df["label"] = ""
    if len(top10) > 0:
        volcano_df.loc[top10.index, "label"] = top10["display_label"].values
    if selected_genes_volcano:
        mask = volcano_df["display_label"].isin(selected_genes_volcano)
        volcano_df.loc[mask, "label"] = volcano_df.loc[mask, "display_label"].values

    fig_volcano = px.scatter(
        volcano_df, x="logFC", y="neg_log10_p", color="Significant",
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
                x_min, x_max, y_min, y_max,
            )
            for i in range(len(labeled)):
                row = labeled.iloc[i]
                ax, ay = offsets[i]
                fig_volcano.add_annotation(
                    x=row["logFC"], y=row["neg_log10_p"], text=row["label"],
                    showarrow=True, arrowhead=1, ax=ax, ay=ay,
                    font=dict(size=11, color="#1a1a1a"),
                    arrowcolor="#444", arrowsize=0.8,
                )
    fig_volcano.update_layout(
        xaxis_title="log₂ Fold Change",
        yaxis_title="-log₁₀ P-value",
        height=500, showlegend=True,
    )
    st.plotly_chart(fig_volcano, use_container_width=True)

    # --- MA plot ---
    if "AveExpr" in deg_df.columns:
        st.subheader("MA plot (AveExpr vs logFC)")
        ma_df = deg_df.copy()
        selected_genes_ma = st.multiselect(
            "Select genes for MA plot (type to search/filter):",
            options=gene_options,
            default=[],
            key="deg_ma_gene_select",
            placeholder="Type gene name to search…",
        )
        ma_df["label"] = ""
        if len(top10) > 0:
            ma_df.loc[top10.index, "label"] = top10["display_label"].values
        if selected_genes_ma:
            mask = ma_df["display_label"].isin(selected_genes_ma)
            ma_df.loc[mask, "label"] = ma_df.loc[mask, "display_label"].values

        fig_ma = px.scatter(
            ma_df, x="AveExpr", y="logFC", color="Significant" if "Significant" in ma_df.columns else None,
            hover_data=[c for c in ["display_label", "logFC", "P.Value"] if c in ma_df.columns],
            color_discrete_map={"Up": "#e74c3c", "Down": "#3498db", "NS": "#95a5a6"} if "Significant" in ma_df.columns else None,
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
                    x_min, x_max, y_min, y_max,
                )
                for i in range(len(labeled_ma)):
                    row = labeled_ma.iloc[i]
                    ax, ay = offsets_ma[i]
                    fig_ma.add_annotation(
                        x=row["AveExpr"], y=row["logFC"], text=row["label"],
                        showarrow=True, arrowhead=1, ax=ax, ay=ay,
                        font=dict(size=11, color="#1a1a1a"),
                        arrowcolor="#444", arrowsize=0.8,
                    )
        fig_ma.update_layout(xaxis_title="Average expression (log₂)", yaxis_title="log₂ Fold Change", height=450)
        st.plotly_chart(fig_ma, use_container_width=True)

    # --- Heatmap ---
    hm_df = st.session_state.get("deg_heatmap_df")
    hm_anno = st.session_state.get("deg_heatmap_annotation_df")
    if hm_df is not None and not hm_df.empty:
        st.subheader("Gene expression heatmap")
        st.caption("Gene expression (log₂ CPM) across your samples. Defaults to the top 30 genes with the most variability in expression.")
        if hm_anno is not None and "SampleName" in hm_anno.columns and "Group" in hm_anno.columns:
            order = list(hm_anno.sort_values("Group")["SampleName"])
            ordered_cols = [c for c in order if c in hm_df.columns]
            other_cols = [c for c in hm_df.columns if c not in ordered_cols]
            if ordered_cols:
                hm_df = hm_df[ordered_cols + other_cols]
        fig_heat = go.Figure(data=go.Heatmap(
            z=hm_df.values,
            x=hm_df.columns.tolist(),
            y=hm_df.index.tolist(),
            colorscale="viridis",
            hoverongaps=False,
            colorbar=dict(title="log₂ CPM"),
        ))
        fig_heat.update_layout(
            xaxis_title="Sample",
            yaxis_title="Gene",
            height=max(400, 25 * len(hm_df)),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- GSEA ---
    gsea_df = st.session_state.get("deg_gsea_df")
    if gsea_df is not None and not gsea_df.empty:
        st.subheader("Enriched gene sets (GSEA Hallmark)")
        id_col = "ID" if "ID" in gsea_df.columns else "Description" if "Description" in gsea_df.columns else gsea_df.columns[0]
        desc_col = "Description" if "Description" in gsea_df.columns else id_col
        p_col = "p.adjust" if "p.adjust" in gsea_df.columns else "pvalue" if "pvalue" in gsea_df.columns else None
        st.dataframe(gsea_df, use_container_width=True)
        if p_col and (id_col in gsea_df.columns or desc_col in gsea_df.columns):
            plot_df = gsea_df.head(20).copy()
            plot_df["neg_log10_p"] = -np.log10(plot_df[p_col].clip(lower=1e-300))
            y_label = desc_col if desc_col in plot_df.columns else id_col
            fig_gsea = px.bar(
                plot_df, x="neg_log10_p", y=plot_df[y_label],
                orientation="h",
                title="Top 20 enriched sets (−log₁₀ p.adjust)",
            )
            fig_gsea.update_layout(height=max(350, 22 * min(20, len(plot_df))), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_gsea, use_container_width=True)
