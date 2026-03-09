"""
DEG result table and visualizations: volcano, MA plot, heatmap, GSEA.
Renders in Streamlit when called from the group selection page.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import io

from deg.plot_helpers import place_labels_no_overlap


def _fig_to_svg(fig, width=1400, height=None):
    """Convert Plotly figure to SVG at specified dimensions (no system dependencies needed)."""
    try:
        # Set figure dimensions if not already set
        if height is None:
            height = int(fig.layout.height or 500)
        # Pass width and height directly to to_image
        return fig.to_image(format="svg", width=width, height=height)
    except Exception:
        try:
            # Fallback: export as HTML if SVG fails
            return fig.to_html()
        except Exception:
            return None

MIN_LFC = 0.6
MAX_FDR = 0.05

# Default API URL
DEG_API_URL = "http://3.141.231.76:8000/"


def _ensure_research_session():
    """Initialize session state for research pipeline."""
    st.session_state.setdefault("deg_research_disease_context", "Unknown")
    st.session_state.setdefault("deg_research_tissue", "Unknown")
    st.session_state.setdefault("deg_research_num_genes", 10)
    st.session_state.setdefault("deg_research_results", None)
    st.session_state.setdefault("deg_research_loading", False)


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
    heatmap, GSEA section, and research LLM pipeline.
    Uses st.session_state for heatmap, GSEA, and research data.
    """
    _ensure_research_session()
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
    volcano_df["label_source"] = ""  # "top10" or "selected"
    if len(top10) > 0:
        volcano_df.loc[top10.index, "label"] = top10["display_label"].values
        volcano_df.loc[top10.index, "label_source"] = "top10"
    if selected_genes_volcano:
        mask = volcano_df["display_label"].isin(selected_genes_volcano)
        volcano_df.loc[mask, "label"] = volcano_df.loc[mask, "display_label"].values
        volcano_df.loc[mask, "label_source"] = "selected"

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
            # Colors: top10 = dark gray, user-selected = teal
            color_top10 = "#1a1a1a"
            color_selected = "#16a085"
            for i in range(len(labeled)):
                row = labeled.iloc[i]
                ax, ay = offsets[i]
                src = row.get("label_source", "top10")
                font_color = color_top10
                arrow_color = color_selected if src == "selected" else "#444"
                fig_volcano.add_annotation(
                    x=row["logFC"], y=row["neg_log10_p"], text=row["label"],
                    showarrow=True, arrowhead=1, ax=ax, ay=ay,
                    font=dict(size=11, color=font_color),
                    arrowcolor=arrow_color, arrowsize=0.8,
                )
    fig_volcano.update_layout(
        xaxis_title="log₂ Fold Change",
        yaxis_title="-log₁₀ P-value",
        height=500, showlegend=True,
        margin=dict(l=70, r=70, t=50, b=70),
    )
    st.caption("Label colors: **top 10 DEGs** = dark gray; **genes you select** = teal.")
    st.plotly_chart(fig_volcano, use_container_width=True)
    volcano_svg = _fig_to_svg(fig_volcano, width=1400, height=int(fig_volcano.layout.height or 500))
    if volcano_svg:
        st.download_button(
            "Download volcano plot (SVG)",
            volcano_svg,
            "deg_volcano.svg",
            "image/svg+xml",
            key="deg_volcano_svg_download",
        )

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
        ma_df["label_source"] = ""
        if len(top10) > 0:
            ma_df.loc[top10.index, "label"] = top10["display_label"].values
            ma_df.loc[top10.index, "label_source"] = "top10"
        if selected_genes_ma:
            mask = ma_df["display_label"].isin(selected_genes_ma)
            ma_df.loc[mask, "label"] = ma_df.loc[mask, "display_label"].values
            ma_df.loc[mask, "label_source"] = "selected"

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
                color_top10 = "#1a1a1a"
                color_selected = "#16a085"
                for i in range(len(labeled_ma)):
                    row = labeled_ma.iloc[i]
                    ax, ay = offsets_ma[i]
                    src = row.get("label_source", "top10")
                    font_color = color_top10
                    arrow_color = color_selected if src == "selected" else "#444"
                    fig_ma.add_annotation(
                        x=row["AveExpr"], y=row["logFC"], text=row["label"],
                        showarrow=True, arrowhead=1, ax=ax, ay=ay,
                        font=dict(size=11, color=font_color),
                        arrowcolor=arrow_color, arrowsize=0.8,
                    )
        fig_ma.update_layout(
            xaxis_title="Average expression (log₂)",
            yaxis_title="log₂ Fold Change",
            height=450,
            margin=dict(l=70, r=70, t=50, b=70),
        )
        st.caption("Label colors: **top 10 DEGs** = dark gray; **genes you select** = teal.")
        st.plotly_chart(fig_ma, use_container_width=True)
        ma_svg = _fig_to_svg(fig_ma, width=1400, height=int(fig_ma.layout.height or 450))
        if ma_svg:
            st.download_button(
                "Download MA plot (SVG)",
                ma_svg,
                "deg_ma_plot.svg",
                "image/svg+xml",
                key="deg_ma_svg_download",
            )

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
        # Vertical line between each sample (between every pair of columns)
        n_cols = len(hm_df.columns)
        shapes = []
        for i in range(n_cols - 1):
            x_pos = i + 0.5
            shapes.append(dict(
                type="line",
                x0=x_pos, x1=x_pos,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="rgba(0,0,0,0.4)", width=1, dash="solid"),
            ))
        fig_heat.update_layout(
            xaxis_title="Sample",
            yaxis_title="Gene",
            height=max(400, 25 * len(hm_df)),
            yaxis=dict(autorange="reversed"),
            shapes=shapes,
            margin=dict(l=120, r=80, t=50, b=120),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        heatmap_svg = _fig_to_svg(fig_heat, width=1600, height=int(fig_heat.layout.height or 600))
        if heatmap_svg:
            st.download_button(
                "Download heatmap (SVG)",
                heatmap_svg,
                "deg_expression_heatmap.svg",
                "image/svg+xml",
                key="deg_heatmap_svg_download",
            )

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
            fig_gsea.update_layout(height=max(350, 25 * min(20, len(plot_df))), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_gsea, use_container_width=True)
            gsea_svg = _fig_to_svg(fig_gsea, width=1400, height=int(fig_gsea.layout.height or 450))
            if gsea_svg:
                st.download_button(
                    "Download GSEA barplot (SVG)",
                    gsea_svg,
                    "deg_gsea_barplot.svg",
                    "image/svg+xml",
                    key="deg_gsea_svg_download",
                )

    # --- Research LLM Pipeline ---
    st.divider()
    st.subheader("AI-Powered Research Insights")
    st.caption("Generate disease associations, drug response predictions, and research hypotheses using Claude LLM.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state["deg_research_disease_context"] = st.text_input(
            "Disease context:",
            key="deg_disease_input"
        )
    with col2:
        st.session_state["deg_research_tissue"] = st.text_input(
            "Tissue type:",
            key="deg_tissue_input"
        )
    with col3:
        st.session_state["deg_research_num_genes"] = st.number_input(
            "Number of top genes:",
            min_value=1,
            max_value=100,
            value=st.session_state.get("deg_research_num_genes", 10),
            key="deg_num_genes_input"
        )

    if st.button("Run Research Pipeline", type="primary", key="deg_research_run"):
        _ensure_research_session()
        with st.spinner("Running research pipeline… generating LLM predictions (this may take 1–2 minutes)"):
            try:
                # Get the API URL from session or use default
                api_url = st.session_state.get("deg_api_url", DEG_API_URL)

                # Convert DEG table to CSV bytes
                deg_csv_bytes = deg_df.to_csv(index=False).encode()

                # Prepare files and form data for multipart upload
                files = {"deg_table": ("deg_results.csv", io.BytesIO(deg_csv_bytes), "text/csv")}
                data = {
                    "disease_context": st.session_state["deg_research_disease_context"],
                    "tissue": st.session_state["deg_research_tissue"],
                    "num_genes": st.session_state["deg_research_num_genes"]
                }

                # Send request to backend
                resp = requests.post(
                    f"{api_url}deg_with_research/",
                    files=files,
                    data=data,
                    timeout=300
                )
                resp.raise_for_status()

                # Store results in session state
                st.session_state["deg_research_results"] = resp.json()
                st.success("Research pipeline completed successfully!")
                st.rerun()

            except requests.exceptions.RequestException as e:
                try:
                    err_msg = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
                except Exception:
                    err_msg = str(e)
                st.error(f"Pipeline failed: {err_msg}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Display research results
    research_results = st.session_state.get("deg_research_results")
    if research_results is not None:
        st.markdown("---")

        # DEG Summary
        if "deg_summary" in research_results:
            with st.expander("Analysis Summary", expanded=True):
                summary_text = research_results["deg_summary"]
                st.write(summary_text)
                st.download_button(
                    "Download analysis summary (TXT)",
                    summary_text,
                    "deg_analysis_summary.txt",
                    "text/plain",
                    key="deg_summary_download",
                )

        # Biological Context
        if "biological_context" in research_results:
            with st.expander("Biological Context"):
                bio_text = research_results["biological_context"]
                st.text(bio_text)
                st.download_button(
                    "Download biological context (TXT)",
                    bio_text,
                    "deg_biological_context.txt",
                    "text/plain",
                    key="deg_bio_download",
                )

        # Disease Predictions
        if "disease_predictions" in research_results:
            pred = research_results["disease_predictions"]
            with st.expander("Disease Association Predictions"):
                pred_text = pred.get("predictions", "")
                st.write(pred_text)
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Model: {pred.get('model', 'Unknown')}")
                with col2:
                    st.caption(f"Tokens used: {pred.get('tokens_used', 'N/A')}")
                st.download_button(
                    "Download disease predictions (TXT)",
                    pred_text,
                    "deg_disease_predictions.txt",
                    "text/plain",
                    key="deg_disease_download",
                )

        # Drug Response Predictions
        if "drug_predictions" in research_results:
            pred = research_results["drug_predictions"]
            with st.expander("Drug Response Predictions"):
                pred_text = pred.get("predictions", "")
                st.write(pred_text)
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Model: {pred.get('model', 'Unknown')}")
                with col2:
                    st.caption(f"Tokens used: {pred.get('tokens_used', 'N/A')}")
                st.download_button(
                    "Download drug predictions (TXT)",
                    pred_text,
                    "deg_drug_predictions.txt",
                    "text/plain",
                    key="deg_drug_download",
                )

        # Research Hypotheses
        if "research_hypotheses" in research_results:
            hyp = research_results["research_hypotheses"]
            with st.expander("Novel Research Hypotheses"):
                hyp_text = hyp.get("hypotheses", "")
                st.write(hyp_text)
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Model: {hyp.get('model', 'Unknown')}")
                with col2:
                    st.caption(f"Tokens used: {hyp.get('tokens_used', 'N/A')}")
                st.download_button(
                    "Download research hypotheses (TXT)",
                    hyp_text,
                    "deg_research_hypotheses.txt",
                    "text/plain",
                    key="deg_hypotheses_download",
                )

        # Show error/note if present
        if "prediction_error" in research_results:
            st.warning(f"Prediction error: {research_results['prediction_error']}")
        if "note" in research_results:
            st.info(f"Note: {research_results['note']}")

        # (Per-section download buttons above replace the previous JSON download)
