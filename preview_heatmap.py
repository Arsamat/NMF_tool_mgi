import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def preview_wide_heatmap_inline(df, meta=None, annotation_cols=None, average_groups=False, sample_order=None):
    """
    Show a heatmap preview with metadata annotation bars using seaborn.clustermap.
    Samples are ordered according to the hierarchy defined by annotation_cols,
    following their order of appearance in the metadata file.
    If average_groups=True, samples sharing the same combination of metadata values
    are averaged to smooth the visualization.
    """

    # Downsample columns for preview (for large datasets)
    #df_small = df.iloc[:, ::step]
    df_small = df.copy()
    # If a specific order is provided (e.g. clustering or top module order), apply it
    if sample_order is not None:
        # Only keep columns present in both
        sample_order = [s for s in sample_order if s in df_small.columns]
        df_small = df_small[sample_order]

    samples = df_small.columns


    col_colors = None
    lut = {}

    if meta is not None and annotation_cols:
        
        # Ensure metadata_index is a single column name
        metadata_index = st.session_state["metadata_index"]
        if isinstance(metadata_index, (list, tuple)):
            metadata_index = metadata_index[0]

        # Index metadata by sample ID
        meta_indexed = meta.set_index(metadata_index)
        
    
        # Keep only samples present in the heatmap
        common_samples = [s for s in samples if s in meta_indexed.index]
        if not common_samples:
            st.warning("No overlapping sample names between metadata and expression matrix.")
            return None

        # Subset metadata to match heatmap
        meta_filtered = meta_indexed.loc[common_samples]

        # Convert each annotation column to ordered categorical following metadata order
        if sample_order is None:
            for col in annotation_cols:
                if col in meta_filtered.columns:
                    unique_vals_in_order = pd.unique(meta_filtered[col].dropna())
                    meta_filtered[col] = pd.Categorical(
                        meta_filtered[col],
                        categories=unique_vals_in_order,
                        ordered=True
                    )

            # Preserve order of appearance in metadata (no alphabetical sorting)
            meta_filtered["__order__"] = range(len(meta_filtered))
            meta_sorted = meta_filtered.sort_values(
                by=annotation_cols + ["__order__"],
                kind="stable"
            ).drop(columns="__order__")

            # Reorder df_small columns to match metadata order
            ordered_samples = [s for s in meta_sorted.index if s in df_small.columns]
            df_small = df_small[ordered_samples]
        else:
            meta_sorted = meta_filtered.loc[df_small.columns]

        # --- Averaging across groups if requested ---
        if average_groups:
            # Align metadata and create group keys
            meta_aligned = meta_sorted.loc[df_small.columns]
            group_keys = meta_aligned[annotation_cols].astype(str).agg("_".join, axis=1)

            # Record the unique groups in their order of appearance (preserves metadata order)
            group_order = pd.unique(group_keys)

            # Average module usage scores per group
            df_small = df_small.groupby(group_keys, axis=1).mean()

            # Reorder columns to follow original group appearance order
            df_small = df_small[group_order]

            # Build new metadata table at group level (for annotation colors)
            meta_grouped = (
                meta_aligned[annotation_cols]
                .assign(_group=group_keys)
                .drop_duplicates(subset="_group")
                .set_index("_group")
                .loc[group_order]  # enforce same column order
            )

            meta_sorted = meta_grouped


        # --- Build color annotations ---
        palettes = [
            "Set1", "Set2", "Paired", "Pastel1", "Pastel2",
            "Dark2", "Accent", "tab10", "tab20"
        ]
        col_colors = pd.DataFrame(index=df_small.columns)
        for i, col in enumerate(annotation_cols):
            if col not in meta_sorted.columns:
                continue
            unique_vals = pd.unique(meta_sorted[col].dropna())
            palette = sns.color_palette(palettes[i % len(palettes)], n_colors=len(unique_vals))
            lut[col] = dict(zip(unique_vals, palette))
            # Safe mapping: avoids MultiIndex error
            col_colors[col] = meta_sorted[col].astype(object).map(lambda x: lut[col].get(x, None))

    # --- Plot heatmap ---
    g = sns.clustermap(
        df_small,
        cmap="viridis",
        figsize=(40, 18),
        col_colors=col_colors if col_colors is not None and not col_colors.empty else None,
        col_cluster=False,
        row_cluster=False
    )

    # --- Hide sample (x-axis) labels ---
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=12)

    # --- Add legends for annotation bars ---
    if annotation_cols and lut:
        for i, col in enumerate(annotation_cols):
            if col not in lut:
                continue
            for label, color in lut[col].items():
                g.ax_col_dendrogram.bar(
                    0, 0, color=color,
                    label=f"{col}: {label}", linewidth=0
                )
        g.ax_col_dendrogram.legend(
            loc="center", ncol=2, bbox_to_anchor=(0.5, 1.1), prop={'size': 20}
        )

    return g.fig








