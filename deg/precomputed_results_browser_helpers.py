from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

import streamlit as st
import numpy as np
import plotly.express as px
from deg.viz_helpers import MIN_LFC, MAX_FDR, _fig_to_html, _valid_display_label
from deg.plot_helpers import place_labels_no_overlap
from collections import defaultdict


def render_precomputed_deg_term_sidebar(
    terms: list[dict],
    model_design: str,
    selected_key: str = "deg_precomputed_selected_term",
) -> Optional[str]:
    """
    Render a sidebar chooser for precomputed DEG contrast terms.

    What this function does:
    1. Collect all valid term labels.
    2. Keep the currently selected term in Streamlit session state.
    3. Split terms into:
       - main effects
       - interaction terms
       - unknown terms
    4. Group main effects by variable inferred from the model design.
    5. Render each group as clickable buttons in the sidebar.

    Expected term structure:
        {
            "term_label": "GenotypeTDP43",
            "effect_interaction": "main"
        }

    Example model design:
        "~ Genotype + Treatment2 + Genotype:Treatment2 + Run"
    """

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def make_button_key(prefix: str, term_label: str) -> str:
        """
        Build a unique Streamlit widget key for a term button.

        We hash the label because labels may be long or contain characters
        that are awkward in widget keys.
        """
        short_hash = hashlib.sha256(term_label.encode()).hexdigest()[:16]
        return f"{prefix}_{short_hash}"

    def normalize_kind(raw: Any) -> str:
        """
        Normalize the effect_interaction value from the DB.

        Allowed outputs:
            "main"
            "interaction"
            "unknown"
        """
        value = str(raw or "").strip().lower()
        return value if value in {"main", "interaction"} else "unknown"

    def parse_main_vars(design: str) -> list[str]:
        """
        Extract only main-effect variables from a model formula.

        Example:
            "~ Genotype + Treatment2 + Genotype:Treatment2 + Run"
        becomes:
            ["Genotype", "Treatment2", "Run"]
        """
        if not design or "~" not in design:
            return []

        rhs = design.split("~", 1)[1]
        parts = [part.strip() for part in rhs.split("+") if part.strip()]
        return [part for part in parts if ":" not in part]

    def infer_main_var(term_label: str, main_vars: list[str]) -> Optional[str]:
        """
        Guess which main variable a term belongs to based on its label.

        Longest names are checked first so that:
            "Treatment2" matches before "Treatment"
        """
        for var in sorted(main_vars, key=len, reverse=True):
            if term_label.startswith(var) or re.search(rf"(^|_){re.escape(var)}", term_label):
                return var
        return None

    def group_sort_key(variable: str) -> tuple[int, str]:
        """
        Sort variable groups so the most common ones appear first.
        """
        if variable == "Genotype":
            return (0, variable)
        if variable in {"Treatment", "Treatment2"}:
            return (1, variable)
        return (2, variable)

    def render_group_header(title: str) -> None:
        """
        Render a variable header with optional reference-group caption.
        """
        st.subheader(title)

        if title == "Genotype":
            st.caption("vs WT")
        elif title in {"Treatment", "Treatment2"}:
            st.caption("vs Vehicle")

    def render_term_buttons(group_prefix: str, items: list[dict], current_label: str) -> None:
        """
        Render one button per term in a group.

        Clicking a button updates session state and reruns the app so the
        rest of the interface refreshes using the newly selected term.
        """
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

    # ------------------------------------------------------------------
    # Step 1: collect valid labels
    # ------------------------------------------------------------------

    labels = [term["term_label"] for term in terms if term.get("term_label")]

    if not labels:
        # No valid terms means there is nothing to show or select.
        st.session_state[selected_key] = None
        return None

    # ------------------------------------------------------------------
    # Step 2: sync current selection with session state
    # ------------------------------------------------------------------

    current = st.session_state.get(selected_key)
    if current not in labels:
        # On first render, or when available terms changed,
        # default to the first valid label.
        current = labels[0]
        st.session_state[selected_key] = current

    # ------------------------------------------------------------------
    # Step 3: split terms by effect type
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Step 4: show overall sidebar header
    # ------------------------------------------------------------------

    st.markdown("**DEG contrasts**")
    st.write(f"Selected: `{current}`")

    # ------------------------------------------------------------------
    # Step 5: group main terms by variable inferred from model design
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Step 6: render main-effect groups
    # ------------------------------------------------------------------

    st.markdown("**Main effects**")

    for variable, term in grouped_main_terms.items():
        st.subheader(variable)
        st.caption(f'vs {term[0].get("reference", "")}')
        render_term_buttons(f"deg_sb_main_{variable}", grouped_main_terms[variable], current)

    if unscoped_main_terms:
        st.subheader("Other main effects")
        render_term_buttons("deg_sb_main_other", unscoped_main_terms, current)

    # ------------------------------------------------------------------
    # Step 7: render interaction terms
    # ------------------------------------------------------------------

    if interaction_terms:
        st.markdown("**Interaction terms**")
        render_term_buttons("deg_sb_ixn", interaction_terms, current)

    # ------------------------------------------------------------------
    # Step 8: render unknown terms
    # ------------------------------------------------------------------

    if unknown_terms:
        st.markdown("**Other**")
        st.caption("Missing or unrecognized effect_interaction.")
        render_term_buttons("deg_sb_unknown", unknown_terms, current)

    # Return the currently selected term label
    return st.session_state.get(selected_key)


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

