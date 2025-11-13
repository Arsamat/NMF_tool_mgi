import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import io
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
    # Dynamically scale height so rows are thicker
    num_genes = df_reordered.shape[0]
    fig_height = max(10, num_genes * 0.2)  # 0.6 = taller rows

    g = sns.clustermap(
        df_reordered,
        cmap = sns.color_palette("Reds", as_cmap=True),
        figsize=(30, fig_height),
        col_colors=col_colors if col_colors is not None and not col_colors.empty else None,
        col_cluster=False,
        row_cluster=False,
        cbar_kws={"orientation": "vertical", "shrink": 0.5},
        xticklabels=False,
        yticklabels=True
    )

    # # --- Adjust layout spacing ---
    # if col_colors is None or col_colors.empty:
    #     g.ax_col_dendrogram.set_visible(False)
    #     g.fig.subplots_adjust(top=0.98, bottom=0.05, left=0.25, right=0.95)
    # else:
    #     # Bring heatmap and annotation bars closer together
    #     g.fig.subplots_adjust(top=0.93, bottom=0.05, left=0.25, right=0.95)

    # --- Move colorbar to LEFT ---
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.ax.set_position([0.08, 0.25, 0.02, 0.4])
    cbar.set_label("Expression Level", fontsize=12)

    # --- Make gene labels readable ---
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=12)
    g.ax_heatmap.tick_params(axis="y", labelsize=11, pad=10)
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("Genes", fontsize=12)

    # --- Add legends for annotation bars ---
    if annotation_cols and lut and col_colors is not None and not col_colors.empty:
        legend_entries = []
        for col in annotation_cols:
            if col not in lut:
                continue
            for label, color in lut[col].items():
                legend_entries.append(mpatches.Patch(color=color, label=f"{col}: {label}"))
        if legend_entries:
            g.ax_col_dendrogram.legend(
                handles=legend_entries,
                loc="center",
                ncol=2,
               # bbox_to_anchor=(0.5, 1.05),  # moved closer to heatmap
                prop={'size': 14},
                frameon=False
            )

    plt.title("Top Genes Expression Heatmap", fontsize=14, pad=10)
    return g


def plot_expression_heatmap(gene_loadings, preprocessed_df):
    """Main Streamlit wrapper using your preferred logic."""
    preprocessed_df = io.BytesIO(preprocessed_df)
    df = pd.read_feather(preprocessed_df)

    X = st.selectbox("Select number of top genes from each module", options=[i for i in range(1, 21)])
    
    # Extract top genes per module
    df_long = gene_loadings.stack().reset_index()
    df_long.columns = ["module", "gene", "loading"]
    df_long = df_long.drop(0)

    unique_df = df_long.loc[df_long.groupby("gene")["loading"].idxmax()]
    grouped_genes = (
        unique_df.sort_values(by=["module", "loading"], ascending=[True, False])
        .groupby("module")
        .head(X)
    )

    #find each gene that is in the start of each module
    module_labels = grouped_genes[["module", "gene"]]
    
    cluster_boundaries = []

    for i in range(1, len(module_labels)):
        if module_labels.iloc[i, 0] != module_labels.iloc[i - 1, 0]:
            cluster_boundaries.append(i)

    genes_graph = list(grouped_genes["gene"])

    # Keep only selected genes
    df = df.T.reset_index()
    df = df[df["Geneid"].isin(genes_graph)]
    df["Category"] = pd.Categorical(df["Geneid"], categories=genes_graph, ordered=True)
    df_ordered = df.sort_values(by="Category").drop(columns={"Category"})
    df_plot = df_ordered.set_index("Geneid")

    # Align metadata
    meta = st.session_state["meta"]
    meta_index = st.session_state.get("metadata_index", meta.columns[0])
    meta_aligned = meta.set_index(meta_index)
    meta_aligned = meta_aligned.loc[df_plot.columns.intersection(meta_aligned.index)]

    annotation_cols = st.multiselect(
        "Select metadata columns for annotation bars",
        options=meta_aligned.columns.tolist(),
        key="annotation_cols"
    )

    # Build annotation colors + plot
    col_colors, lut = build_color_annotations(meta_aligned, annotation_cols)
    g = make_heatmap(df_plot, col_colors, lut, annotation_cols)

    for y in cluster_boundaries:
        g.ax_heatmap.hlines(
            y=y, xmin=0, xmax=df.shape[1],
            colors="black", linewidth=3, linestyles="-", zorder=10
        )
    st.write("Black lines separate modules")

    st.pyplot(g)
    return g

