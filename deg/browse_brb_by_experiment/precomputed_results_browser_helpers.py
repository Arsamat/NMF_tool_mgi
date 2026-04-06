"""
Canonical implementation for precomputed DEG results browser helpers.

Moved from: `deg/precomputed_results_browser_helpers.py`
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import plotly.express as px
import requests
import streamlit as st

from deg.plot_helpers import place_labels_no_overlap
from deg.viz_helpers import MIN_LFC, MAX_FDR, _fig_to_html, _valid_display_label

METADATA_SORT_KEYS = ("CellType", "Treatment", "Genotype", "Timepoint")
_NONE = "—"


def parse_precomputed_sample_names(raw: Any) -> list[str]:
    """Parse sample_names from deg_mapping (string or list) into a clean list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    normalized = s.replace("\r\n", "\n")
    for sep in ("\n", ";", ","):
        if sep in normalized:
            parts = [p.strip().strip('"').strip("'") for p in normalized.split(sep)]
            return [p for p in parts if p]
    return [s]


def render_precomputed_expression_heatmap(
    deg_df,
    group_data: dict,
    widget_prefix: str,
    api_url: str,
    result_cache_key: str,
):
    """
    Top-30 (by P-value) gene expression heatmap; server pulls counts + Mongo metadata.
    """
    samples = parse_precomputed_sample_names(group_data.get("sample_names"))
    if len(samples) < 2:
        return

    with st.expander("Expression heatmap (top 30 genes by p-value)", expanded=True):
        st.caption(
            "log₂(CPM+1) across samples listed for this precomputed group. "
            "Annotation bars use **MongoDB** metadata (not Group). "
            "Leave sort levels as — to use all available: CellType, Treatment, Genotype, Timepoint."
        )
        options = list(METADATA_SORT_KEYS)
        with st.form("preview_form"):
            sort_keys = st.multiselect(
                "Select annotation columns",
                options=options,
                key=f"{widget_prefix}annotations_widget"
            )

            submitted = st.form_submit_button("Generate Preview")

        base = api_url.rstrip("/") + "/"
        if submitted:
            with st.spinner("Building heatmap on server…"):
                try:
                    csv_buf = io.BytesIO()
                    deg_df.to_csv(csv_buf, index=False)
                    csv_buf.seek(0)
                    resp = requests.post(
                        f"{base}deg_results/precomputed_heatmap",
                        files={"deg_csv": ("deg_results.csv", csv_buf, "text/csv")},
                        data={
                            "samples_json": json.dumps(samples),
                            "annotation_cols_json": json.dumps(sort_keys),
                        },
                        timeout=180,
                    )
                    if resp.status_code != 200:
                        try:
                            detail = resp.json().get("detail", resp.text)
                        except Exception:
                            detail = resp.text
                        st.error(f"Heatmap failed ({resp.status_code}): {detail}")
                        return
                    st.session_state["deg_precomputed_heatmap"] = resp.content
                except requests.exceptions.RequestException as e:
                    st.error(f"Heatmap request failed: {e}")
                    return

        png_bytes = st.session_state["deg_precomputed_heatmap"]
        if png_bytes:
            st.image(png_bytes, use_container_width=True)
            st.download_button(
                "Download heatmap (PNG)",
                png_bytes,
                "precomputed_deg_expression_heatmap.png",
                "image/png",
                key=f"{widget_prefix}pre_hm_dl",
            )


def render_precomputed_deg_term_sidebar(
    terms: list[dict],
    model_design: str,
    selected_key: str = "deg_precomputed_selected_term",
) -> Optional[str]:
    """
    Render a sidebar chooser for precomputed DEG contrast terms.

    Expected term structure:
        {
            "term_label": "GenotypeTDP43",
            "effect_interaction": "main"
        }
    """

    def make_button_key(prefix: str, term_label: str) -> str:
        short_hash = hashlib.sha256(term_label.encode()).hexdigest()[:16]
        return f"{prefix}_{short_hash}"

    def normalize_kind(raw: Any) -> str:
        value = str(raw or "").strip().lower()
        return value if value in {"main", "interaction"} else "unknown"

    def parse_main_vars(design: str) -> list[str]:
        if not design or "~" not in design:
            return []
        rhs = design.split("~", 1)[1]
        parts = [part.strip() for part in rhs.split("+") if part.strip()]
        return [part for part in parts if ":" not in part]

    def infer_main_var(term_label: str, main_vars: list[str]) -> Optional[str]:
        for var in sorted(main_vars, key=len, reverse=True):
            if term_label.startswith(var) or re.search(rf"(^|_){re.escape(var)}", term_label):
                return var
        return None

    def render_term_buttons(group_prefix: str, items: list[dict], current_label: str) -> None:
        for term in sorted(items, key=lambda x: x["term_label"]):
            label = term["term_label"]
            visible_text = f"● {label}" if label == current_label else label
            if st.button(
                visible_text,
                key=make_button_key(group_prefix, label),
                use_container_width=True,
            ):
                st.session_state[selected_key] = label
                st.rerun()

    labels = [term["term_label"] for term in terms if term.get("term_label")]
    if not labels:
        st.session_state[selected_key] = None
        return None

    current = st.session_state.get(selected_key)
    if current not in labels:
        current = labels[0]
        st.session_state[selected_key] = current

    main_terms = []
    interaction_terms = []
    unknown_terms = []
    for term in terms:
        if not term.get("term_label"):
            continue
        kind = normalize_kind(term.get("effect_interaction"))
        if kind == "main":
            main_terms.append(term)
        elif kind == "interaction":
            interaction_terms.append(term)
        else:
            unknown_terms.append(term)

    st.markdown("**DEG contrasts**")
    st.write(f"Selected: `{current}`")

    main_vars = parse_main_vars(model_design)
    grouped_main_terms: dict[str, list[dict]] = defaultdict(list)
    unscoped_main_terms: list[dict] = []
    for term in main_terms:
        label = term["term_label"]
        variable = infer_main_var(label, main_vars) if main_vars else None
        if variable:
            grouped_main_terms[variable].append(term)
        else:
            unscoped_main_terms.append(term)

    st.markdown("**Main effects**")
    for variable, term in grouped_main_terms.items():
        st.subheader(variable)
        st.caption(f"vs {term[0].get('reference', '')}")
        render_term_buttons(f"deg_sb_main_{variable}", grouped_main_terms[variable], current)
    if unscoped_main_terms:
        st.subheader("Other main effects")
        render_term_buttons("deg_sb_main_other", unscoped_main_terms, current)

    if interaction_terms:
        st.markdown("**Interaction terms**")
        render_term_buttons("deg_sb_ixn", interaction_terms, current)
    if unknown_terms:
        st.markdown("**Other**")
        st.caption("Missing or unrecognized effect_interaction.")
        render_term_buttons("deg_sb_unknown", unknown_terms, current)

    return st.session_state.get(selected_key)


def render_table_and_volcano(deg_df, widget_prefix: str = "deg_pre_"):
    """DEG results table + volcano plot only."""
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

    deg_df["display_label"] = deg_df.apply(_valid_display_label, axis=1)
    sig = deg_df[deg_df["Significant"] != "NS"].sort_values("P.Value")
    top10 = sig.head(10)
    gene_options = sorted(
        [
            str(x).strip()
            for x in deg_df["display_label"].dropna().unique()
            if str(x).strip() and str(x).strip() != "—"
        ]
    )

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
            volcano_df,
            x="logFC",
            y="neg_log10_p",
            color="Significant",
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
                labeled["logFC"].values,
                labeled["neg_log10_p"].values,
                labeled["label"].tolist(),
                x_min - x_pad,
                x_max + x_pad,
                y_min - y_pad,
                y_max + y_pad,
            )
            for i, row in enumerate(labeled.itertuples()):
                ax, ay = offsets[i]
                fig.add_annotation(
                    x=row.logFC,
                    y=row.neg_log10_p,
                    text=row.label,
                    showarrow=True,
                    arrowhead=1,
                    ax=ax,
                    ay=ay,
                    font=dict(size=11, color="#1a1a1a"),
                    arrowcolor="#16a085" if getattr(row, "label_source", "") == "selected" else "#444",
                    arrowsize=0.8,
                )

        fig.update_layout(
            xaxis_title="log₂ Fold Change",
            yaxis_title="-log₁₀ P-value",
            height=500,
            showlegend=True,
            margin=dict(l=70, r=70, t=50, b=70),
        )
        st.caption("Label colors: **top 10 DEGs** = dark gray; **genes you select** = teal.")
        st.plotly_chart(fig, use_container_width=True)
        html = _fig_to_html(fig, width=1400, height=500)
        if html:
            st.download_button(
                "Download volcano plot (HTML)",
                html,
                "deg_volcano.html",
                "text/html",
                key=f"{widget_prefix}volcano_html_download",
            )

