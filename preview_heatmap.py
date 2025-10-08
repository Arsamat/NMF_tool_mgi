import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def preview_wide_heatmap_inline(df, meta=None, annotation_cols=None, step=5):
    """
    Show a heatmap preview with metadata annotation bars using seaborn.clustermap.
    Each annotation column gets its own distinct color palette.
    """
    # Downsample columns for preview
    df_small = df.iloc[:, ::step]
    samples = df_small.columns

    col_colors = None
    lut = {}

    if meta is not None and annotation_cols:
        # Index metadata by sample ID
        meta_indexed = meta.set_index(st.session_state["metadata_index"])

        # Keep only samples present in the heatmap
        common_samples = [s for s in samples if s in meta_indexed.index]
        df_small = df_small[common_samples]

        # Rotate through a list of distinct palettes
        palettes = [
            "Set1", "Set2", "Paired", "Pastel1", "Pastel2", "Dark2", "Accent", "tab10", "tab20"
        ]

        # Build color annotations
        col_colors = pd.DataFrame(index=common_samples)
        for i, col in enumerate(annotation_cols):
            if col not in meta_indexed.columns:
                continue
            unique_vals = meta_indexed[col].dropna().unique()

            # assign a different palette per condition
            palette = sns.color_palette(palettes[i % len(palettes)], n_colors=len(unique_vals))
            lut[col] = dict(zip(unique_vals, palette))
            col_colors[col] = meta_indexed.loc[common_samples, col].map(lut[col])

    # --- Plot heatmap ---
    g = sns.clustermap(
        df_small,
        cmap="viridis",
        figsize=(20, 8),
        col_colors=col_colors if col_colors is not None and not col_colors.empty else None,
        col_cluster=False,  # keep given sample order
        row_cluster=False   # keep module order
    )

    # --- Add legends for annotation bars ---
    if annotation_cols:
        for i, col in enumerate(annotation_cols):
            if col not in lut:
                continue
            for label, color in lut[col].items():
                g.ax_col_dendrogram.bar(
                    0, 0, color=color,
                    label=f"{col}: {label}", linewidth=0
                )
        g.ax_col_dendrogram.legend(
            loc="center", ncol=2, bbox_to_anchor=(0.5, 1.1)
        )

    return g.fig



