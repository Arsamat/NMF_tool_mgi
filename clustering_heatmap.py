import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def build_color_annotations(meta_aligned, annotation_cols):
    """Generate color lookup tables and annotation color bars."""
    palettes = [
        "Set1", "Set2", "Paired", "Pastel1", "Pastel2",
        "Dark2", "Accent", "tab10", "tab20"
    ]
    col_colors = pd.DataFrame(index=meta_aligned.index)
    lut = {}

    if annotation_cols:
        for i, col in enumerate(annotation_cols):
            if col not in meta_aligned.columns:
                continue
            unique_vals = pd.unique(meta_aligned[col].dropna())
            palette = sns.color_palette(palettes[i % len(palettes)], n_colors=len(unique_vals))
            lut[col] = dict(zip(unique_vals, palette))
            col_colors[col] = meta_aligned[col].astype(object).map(lut[col])
    return col_colors, lut


def make_heatmap(df_reordered, col_colors, lut, annotation_cols):
    """Render the heatmap using seaborn.clustermap (without clustering)."""
    g = sns.clustermap(
        df_reordered,
        cmap="viridis",
        figsize=(40, 18),
        col_colors=col_colors if col_colors is not None and not col_colors.empty else None,
        col_cluster=False,
        row_cluster=False
    )

    # Hide sample labels if desired
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=12)

    # Add legends for annotations
    if annotation_cols and lut:
        for i, col in enumerate(annotation_cols):
            if col not in lut:
                continue
            for label, color in lut[col].items():
                g.ax_col_dendrogram.bar(
                    0, 0, color=color, label=f"{col}: {label}", linewidth=0
                )
        g.ax_col_dendrogram.legend(
            loc="center", ncol=2, bbox_to_anchor=(0.5, 1.1), prop={"size": 22}
        )
    
    return g.fig


def plot_heatmap_with_metadata(df, meta, leaf_order, cluster_labels=None):
    """
    Main Streamlit integration.
    Adds an optional 'Cluster' color bar if cluster_labels is provided.
    """
    # Fix sample order
    df_reordered = df[leaf_order]

    # Align metadata
    meta_indexed = meta.set_index(st.session_state["metadata_index"])
    meta_aligned = meta_indexed.loc[leaf_order]


    # If cluster labels provided, add them as a column
    if cluster_labels is not None:
        cluster_labels_ordered = cluster_labels.loc[leaf_order]
        meta_aligned["Cluster"] = cluster_labels_ordered
        
    # Let user select metadata columns
    annotation_cols = st.multiselect(
        "Select metadata variables to annotate (Cluster included automatically if provided)",
        options=meta_aligned.columns.tolist(),
        default=["Cluster"] if "Cluster" in meta_aligned.columns else [],
        help="Samples stay in the same order as clustering."
    )
    
    # Build annotations and plot
    col_colors, lut = build_color_annotations(meta_aligned, annotation_cols)
    fig = make_heatmap(df_reordered, col_colors, lut, annotation_cols)

    st.pyplot(fig)

    return fig


def plot_heatmap_clustered_modules(
    df,
    cluster_labels,
    meta=None,
    annotation_cols=None,
    module_leaf_order=None,
    figsize=(40, 18),
    cmap="viridis"
):
    """
    Plot a heatmap where:
      - Rows (modules) are ordered by hierarchical clustering (module_leaf_order)
      - Columns (samples) keep their original order (no sorting)
      - Metadata annotations are overlaid as color bars (no reordering)
      - Module clusters are visually separated with horizontal lines
    """

    # --- 1️⃣ Reorder rows (modules) using hierarchical clustering ---
    if module_leaf_order is not None:
        df_reordered = df.iloc[module_leaf_order, :]
        cluster_labels = cluster_labels.iloc[module_leaf_order]
    else:
        df_reordered = df.copy()

    # --- 2️⃣ Build sample annotation color bars (no reordering of samples) ---
    col_colors = None
    lut = {}

    if meta is not None and annotation_cols:
        metadata_index = st.session_state["metadata_index"]
        meta_indexed = meta.set_index(metadata_index)

        # Align metadata order to df sample columns (same order)
        meta_aligned = meta_indexed.loc[df.columns.intersection(meta_indexed.index)]

        # Build color bars (no sorting)
        palettes = [
            "Set1", "Set2", "Paired", "Pastel1", "Pastel2",
            "Dark2", "Accent", "tab10", "tab20"
        ]
        col_colors = pd.DataFrame(index=meta_aligned.index)
        for i, col in enumerate(annotation_cols):
            if col not in meta_aligned.columns:
                continue
            unique_vals = pd.unique(meta_aligned[col].dropna())
            palette = sns.color_palette(palettes[i % len(palettes)], n_colors=len(unique_vals))
            lut[col] = dict(zip(unique_vals, palette))
            col_colors[col] = meta_aligned[col].astype(object).map(lut[col])

        # Ensure col_colors exactly matches df column order and length
        col_colors = col_colors.reindex(df.columns)

        # Convert to numpy (breaks Seaborn’s implicit reindexing behavior)
        col_colors = col_colors.reindex(df.columns).to_numpy().T

    # --- 3️⃣ Plot heatmap ---
    g = sns.clustermap(
        df_reordered,
        cmap=cmap,
        figsize=figsize,
        col_cluster=False,   # keep sample order fixed
        row_cluster=False,   # already ordered by module_leaf_order
        col_colors=col_colors if col_colors is not None and len(col_colors) > 0 else None,
    )
    

    # --- 4️⃣ Hide sample labels (x-axis) ---
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=12)

    # --- 5️⃣ Add horizontal separators for cluster boundaries ---
    cluster_boundaries = []
    
    for i in range(1, len(cluster_labels)):
        if cluster_labels.iloc[i] != cluster_labels.iloc[i - 1]:
            cluster_boundaries.append(i)
    
    for y in cluster_boundaries:
        g.ax_heatmap.hlines(
            y=y, xmin=0, xmax=df.shape[1],
            colors="white", linewidth=3, linestyles="-", zorder=10
        )

    # --- 6️⃣ Add legends for metadata annotations ---
    if annotation_cols and lut:
        for i, col in enumerate(annotation_cols):
            for label, color in lut[col].items():
                g.ax_col_dendrogram.bar(0, 0, color=color, label=f"{col}: {label}", linewidth=0)
        g.ax_col_dendrogram.legend(
            loc="center", ncol=2, bbox_to_anchor=(0.5, 1.1), prop={"size": 22}
        )

    st.pyplot(g.fig)

    return g.fig

