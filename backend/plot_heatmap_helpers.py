import io
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from fastapi.responses import JSONResponse




def create_dendrogram(Z, cluster_labels=None, title="Dendrogram"):
    """
    Creates a dendrogram where each cluster receives its own color.
    This version guarantees link_color_func always returns a valid matplotlib color.
    """

    fig, ax = plt.subplots(figsize=(14, 5))

    if cluster_labels is not None:
        # Ensure cluster labels are plain ints
        cluster_labels = np.array(cluster_labels).astype(int)

        # Unique clusters
        unique_clusters = sorted(set(cluster_labels))

        # Build a color map: cluster_id -> color
        palette = sns.color_palette("husl", len(unique_clusters))
        cluster_color_map = {
            cl: mcolors.to_hex(palette[i])     # 🔥 ALWAYS valid matplotlib color string
            for i, cl in enumerate(unique_clusters)
        }

        # Leaf node colors
        leaf_color_map = {
            leaf_index: cluster_color_map[cluster_labels[leaf_index]]
            for leaf_index in range(len(cluster_labels))
        }

        def link_color_func(node_id):
            """
            Determines color of a branch.
            Must ALWAYS return a valid matplotlib color string.
            """

            # Case 1 — leaf node
            if node_id < len(cluster_labels):
                return leaf_color_map[node_id]

            # Case 2 — internal node
            left_child = int(Z[node_id - len(cluster_labels), 0])
            right_child = int(Z[node_id - len(cluster_labels), 1])

            left_color = link_color_func(left_child)
            right_color = link_color_func(right_child)

            # If both children have same cluster → keep that color
            if left_color == right_color:
                return left_color

            # Otherwise mixed cluster → gray
            return "#999999"

        dendro = dendrogram(
            Z,
            no_labels=True,
            color_threshold=0,
            link_color_func=link_color_func,   # 🔥 our safe function
            ax=ax
        )

    else:
        # No cluster labels → plain dendrogram
        dendro = dendrogram(
            Z,
            no_labels=True,
            color_threshold=0,
            ax=ax
        )

    ax.set_title(title)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return dendro["leaves"], buf


def make_heatmap_png(df, col_colors=None, lut=None):
    """
    Create heatmap PNG with annotation bars that use the SAME palette list
    (Set1, Set2, Paired, Pastel1, Pastel2, Dark2, Accent, tab10, tab20).
    """
    import matplotlib.patches as mpatches

    # -------------------------
    # Build clustermap
    # -------------------------
    if col_colors is None or col_colors.empty:
        g = sns.clustermap(
            df,
            cmap="viridis",
            col_cluster=False,
            row_cluster=False,
            figsize=(22, 12)
        )
    else:
        g = sns.clustermap(
            df,
            cmap="viridis",
            col_cluster=False,
            row_cluster=False,
            col_colors=col_colors,
            figsize=(22, 12)
        )

    # Clean x-labels / titles
    g.ax_heatmap.set_title("")
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.tick_params(axis="x", which="both", length=0)

    # -------------------------
    # LEGEND (LUT-driven)
    # -------------------------
    if lut:
        handles = []
        for col, mapping in lut.items():
            for val, color in mapping.items():
                handles.append(mpatches.Patch(color=color, label=f"{col}: {val}"))

        if handles:
            g.ax_col_dendrogram.legend(
                handles,
                [h.get_label() for h in handles],
                title="Annotations",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.12),
                ncol=3,
                frameon=False,
                prop={"size": 16},       # 🔥 Larger label text
                title_fontsize=18        # 🔥 Larger legend title
            )

    # -------------------------
    # Export PNG
    # -------------------------
    buf = io.BytesIO()
    g.fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    buf.seek(0)
    return buf




def build_annotations(meta, annotation_cols):
    """
    Build annotation colors using the SAME palette list you provided.
    Each annotation column gets its own palette (cycled through).
    """

    palettes = [
        "Set1", "Set2", "Paired", "Pastel1", "Pastel2",
        "Dark2", "Accent", "tab10", "tab20"
    ]

    lut = {}
    col_colors = pd.DataFrame(index=meta.index)

    if not annotation_cols:
        return col_colors, lut

    # Loop through annotation columns and assign palettes
    for i, col in enumerate(annotation_cols):
        if col not in meta.columns:
            continue

        unique_vals = meta[col].dropna().unique()
        palette_name = palettes[i % len(palettes)]
        palette = sns.color_palette(palette_name, len(unique_vals))

        # Build LUT entry
        lut[col] = dict(zip(unique_vals, palette))

        # Apply colors
        col_colors[col] = meta[col].map(lut[col])

    return col_colors, lut


def dendogram_modules(df, n_clusters):

    Z = linkage(df.values, method="ward")
    module_names = df.index.tolist()

    # flat cluster assignments based on original df ordering
    orig_cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

    # build fig + dendrogram
    fig, ax = plt.subplots(figsize=(14, 5))
    dendro = dendrogram(
        Z,
        labels=module_names,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0,
        above_threshold_color="black",
        ax=ax
    )

    leaf_order = dendro["leaves"]  # order of module indices in dendrogram

    # === CRITICAL FIX: reorder cluster labels to match dendrogram leaf order ===
    reordered_cluster_labels = orig_cluster_labels[leaf_order]
    reordered_module_names = [module_names[i] for i in leaf_order]

    # unique clusters in dendrogram order
    unique_clusters = np.unique(reordered_cluster_labels)
    palette = sns.color_palette("husl", len(unique_clusters))
    cluster_color_map = {
        cluster: palette[i] for i, cluster in enumerate(unique_clusters)
    }

    # Color leaf labels in dendrogram order
    xtick_labels = ax.get_xmajorticklabels()

    for i, lbl in enumerate(xtick_labels):
        cluster_id = reordered_cluster_labels[i]
        lbl.set_color(cluster_color_map[cluster_id])

    # Convert to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    dendro_hex = buf.read().hex()

    return JSONResponse({
        "module_leaf_order": leaf_order,
        "cluster_labels": orig_cluster_labels.tolist(),
        "dendrogram_png": dendro_hex
    })




def module_annotated_heatmap(
    df_reordered,
    meta,
    annotation_cols,
    cluster_labels_ordered,
    module_leaf_order
):
    """
    Updated version that uses the same palette list as make_heatmap_png()
    and produces larger annotation legend font sizes.
    """

    # ----------------------------------------------------------
    # 1. Annotation palettes (consistent with build_annotations)
    # ----------------------------------------------------------
    palettes = [
        "Set1", "Set2", "Paired", "Pastel1", "Pastel2",
        "Dark2", "Accent", "tab10", "tab20"
    ]

    lut = {}
    col_colors_df = pd.DataFrame(index=meta.index)

    # Assign palette per annotation column (cycling through list)
    for i, col in enumerate(annotation_cols):
        if col not in meta.columns:
            continue

        unique_vals = meta[col].dropna().unique()
        palette_name = palettes[i % len(palettes)]
        palette = sns.color_palette(palette_name, len(unique_vals))

        # LUT entry for this column
        lut[col] = dict(zip(unique_vals, palette))

        # Map selected colors
        col_colors_df[col] = meta[col].map(lut[col])

    # Convert to list-of-lists for seaborn
    col_colors = [col_colors_df[col].tolist() for col in annotation_cols]

    # ----------------------------------------------------------
    # 2. Build heatmap
    # ----------------------------------------------------------
    if col_colors is None or len(col_colors) == 0:
        g = sns.clustermap(
            df_reordered,
            cmap="viridis",
            col_cluster=False,
            row_cluster=False,
            figsize=(22, 12)
        )
    else:
        g = sns.clustermap(
            df_reordered,
            cmap="viridis",
            col_cluster=False,
            row_cluster=False,
            col_colors=col_colors,
            figsize=(22, 12)
        )

    # Hide x labels
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.tick_params(axis="x", length=0)
    g.ax_heatmap.set_xlabel("")

    # ----------------------------------------------------------
    # 3. Add separator lines between module clusters
    # ----------------------------------------------------------
    boundaries = []
    for i in range(1, len(cluster_labels_ordered)):
        if cluster_labels_ordered[i] != cluster_labels_ordered[i - 1]:
            boundaries.append(i)

    for y in boundaries:
        g.ax_heatmap.hlines(
            y=y,
            xmin=0,
            xmax=df_reordered.shape[1],
            colors="white",
            linewidth=3,
            zorder=10
        )

    # ----------------------------------------------------------
    # 4. Annotation Legends (larger fonts!)
    # ----------------------------------------------------------
    if lut:
        handles = []
        for col, mapping in lut.items():
            for val, color in mapping.items():
                handles.append(
                    mpatches.Patch(color=color, label=f"{col}: {val}")
                )

        g.ax_col_dendrogram.legend(
            handles,
            [h.get_label() for h in handles],
            title="Annotations",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=3,
            frameon=False,
            prop={"size": 16},      # 🔥 Larger legend label font
            title_fontsize=18       # 🔥 Larger legend title font
        )

    # ----------------------------------------------------------
    # 5. Export PNG → hex
    # ----------------------------------------------------------
    buf = io.BytesIO()
    g.fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    buf.seek(0)

    return JSONResponse({
        "heatmap_png": buf.read().hex(),
        "module_leaf_order": module_leaf_order,
        "cluster_labels": cluster_labels_ordered,
        "boundaries": boundaries,
        "lut": {col: {str(k): v for k, v in mapping.items()} for col, mapping in lut.items()}
    })
