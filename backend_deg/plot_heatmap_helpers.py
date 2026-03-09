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
            figsize=(32, 18)
        )
    else:
        g = sns.clustermap(
            df,
            cmap="viridis",
            col_cluster=False,
            row_cluster=False,
            col_colors=col_colors,
            figsize=(32, 18)
        )
    if col_colors is not None and not col_colors.empty:
        ax = g.ax_col_colors

        # Move annotation labels to the LEFT
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")

        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
            pad=8,     # spacing between labels and bars
            labelsize=20
        )

        # Ensure labels are horizontal and readable
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

    # Clean x-labels / titles
    g.ax_heatmap.set_title("")
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xlabel("")
    # g.ax_heatmap.tick_params(axis="x", which="both", length=0)

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
                loc="upper left",
                bbox_to_anchor=(1.02, 1),   # 👉 move legend to the RIGHT
                ncol=1,                     # vertical stack
                frameon=False,
                prop={"size": 24},
                title_fontsize=24
            )

    # -------------------------
    # Export PNG
    # -------------------------
    buf = io.BytesIO()
    g.fig.savefig(buf, format="png", dpi=400, bbox_inches="tight")
    plt.close(g.fig)
    buf.seek(0)
    return buf


def build_annotations(meta, annotation_cols, use_glasbey=True):
    """
    Build annotation colors with a GLOBAL palette across ALL columns.
    Each (column, value) pair gets a unique color, so colors won't repeat
    across different annotation variables.

    Returns:
      col_colors: DataFrame (index=meta.index, columns=annotation_cols) with RGB tuples
      lut: dict {col: {value: color}}
    """
    lut = {}
    col_colors = pd.DataFrame(index=meta.index)

    if not annotation_cols:
        return col_colors, lut

    # ----------------------------
    # 1) Collect all tokens: (col, value)
    # ----------------------------
    tokens = []
    for col in annotation_cols:
        if col not in meta.columns:
            continue
        vals = pd.unique(meta[col].dropna().astype(object))
        tokens.extend([(col, v) for v in vals])

    # If nothing to color
    if len(tokens) == 0:
        return col_colors, lut

    # ----------------------------
    # 2) Create one big palette
    # ----------------------------
    if use_glasbey:
        # colorcet Glasbey is great for many distinct categorical colors
        # pip install colorcet
        import colorcet as cc
        palette = cc.glasbey[:len(tokens)]
    else:
        # fallback if you don't want colorcet
        import seaborn as sns
        palette = sns.color_palette("husl", len(tokens))

    global_lut = {tok: palette[i] for i, tok in enumerate(tokens)}

    # ----------------------------
    # 3) Build per-column LUT + col_colors
    # ----------------------------
    for col in annotation_cols:
        if col not in meta.columns:
            continue

        vals = pd.unique(meta[col].dropna().astype(object))
        lut[col] = {v: global_lut[(col, v)] for v in vals}

        col_colors[col] = (
            meta[col].astype(object)
                .map(lut[col])
        )

    return col_colors, lut
        

def dendogram_modules(df, n_clusters):

    Z = linkage(df.values, method="ward")
    module_names = df.index.tolist()

    # flat cluster assignments based on original df ordering
    # (fcluster will likely treat n_clusters=0 as n_clusters=1)
    orig_cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

    # build fig + dendrogram
    fig, ax = plt.subplots(figsize=(14, 5))
    
    if n_clusters > 1:
        # 1. Custom coloring for branches AND labels
        dendro = dendrogram(
            Z,
            labels=module_names,
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold= 0,
            above_threshold_color="black",
            ax=ax
        )
    else:
        # 2. Force ALL branches to be BLACK for unclustered view
        dendro = dendrogram(
            Z,
            labels=module_names,
            leaf_rotation=90,
            leaf_font_size=10,
            # Force color threshold high so all lines are 'above' and colored black
            color_threshold=float('inf'), 
            above_threshold_color="black",
            ax=ax
        )
        
    # --- Code execution MUST continue from here for both branches ---
    
    leaf_order = dendro["leaves"]  # order of module indices in dendrogram

    # === CRITICAL FIX: reorder cluster labels to match dendrogram leaf order ===
    reordered_cluster_labels = orig_cluster_labels[leaf_order]

    # --- Only apply custom leaf label coloring if n_clusters > 1 ---
    if n_clusters > 1:

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
    else:
        # force all leaf labels to black
        for lbl in ax.get_xmajorticklabels():
            lbl.set_color("black")

    # The labels will remain the default black (from above dendrogram call) if n_clusters <= 1

    # Convert to PNG (Final save for BOTH cases)
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
    module_leaf_order,
    use_glasbey=True,          # matches build_annotations default
):
    """
    Preserves:
      - heatmap core behavior (viridis, no clustering, etc.)
      - boundary line placement between module clusters
      - JSONResponse payload fields

    Updates to match:
      - build_annotations(): GLOBAL palette across ALL annotation columns
      - make_heatmap_png(): annotation label styling + legend styling/placement
    """

    # -----------------------------
    # 0) Align metadata to df columns
    # -----------------------------
    # df_reordered columns are the heatmap columns; annotations must follow that order.
    meta_aligned = meta.reindex(df_reordered.columns)

    # Keep only valid annotation columns
    annotation_cols = [c for c in (annotation_cols or []) if c in meta_aligned.columns]

    # -----------------------------
    # 1) Build annotations exactly like build_annotations()
    # -----------------------------
    lut = {}
    col_colors = pd.DataFrame(index=meta_aligned.index)

    if annotation_cols:
        # 1) Collect all tokens (col, value) across all annotation columns
        tokens = []
        for col in annotation_cols:
            vals = pd.unique(meta_aligned[col].dropna().astype(object))
            tokens.extend([(col, v) for v in vals])

        if len(tokens) > 0:
            # 2) Create one big palette
            if use_glasbey:
                try:
                    import colorcet as cc
                    palette = cc.glasbey[:len(tokens)]
                except Exception:
                    # fallback (still global + distinct-ish)
                    palette = sns.color_palette("husl", len(tokens))
            else:
                palette = sns.color_palette("husl", len(tokens))

            global_lut = {tok: palette[i] for i, tok in enumerate(tokens)}

            # 3) Per-column LUT + col_colors DataFrame
            for col in annotation_cols:
                vals = pd.unique(meta_aligned[col].dropna().astype(object))
                lut[col] = {v: global_lut[(col, v)] for v in vals}

                col_colors[col] = meta_aligned[col].astype(object).map(lut[col])

    # -----------------------------
    # 2) Build heatmap (preserve core behavior)
    # -----------------------------
    # Seaborn accepts col_colors as DataFrame; your make_heatmap_png checks empty DataFrame too.
    if col_colors is None or col_colors.empty:
        g = sns.clustermap(
            df_reordered,
            cmap="viridis",
            col_cluster=False,
            row_cluster=False,
            figsize=(22, 12)  # keep original size; change to (32,18) if you want identical sizing
        )
    else:
        g = sns.clustermap(
            df_reordered,
            cmap="viridis",
            col_cluster=False,
            row_cluster=False,
            col_colors=col_colors,
            figsize=(26, 14)
        )

    # Match make_heatmap_png() annotation bar label styling
    if col_colors is not None and not col_colors.empty:
        ax = g.ax_col_colors

        # Move annotation labels to the LEFT
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")

        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
            pad=8,
            labelsize=20
        )

        # Ensure labels are horizontal and readable
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

    # Clean x-labels / titles (same as your original + make_heatmap_png)
    g.ax_heatmap.set_title("")
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xlabel("")

    # -----------------------------
    # 3) Boundary lines between module clusters (preserve exactly)
    # -----------------------------
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

    # -----------------------------
    # 4) Legend exactly like make_heatmap_png() (LUT-driven, right side)
    # -----------------------------
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
                loc="upper left",
                bbox_to_anchor=(1.02, 1),  # move legend to RIGHT (identical)
                ncol=1,
                frameon=False,
                prop={"size": 20},
                title_fontsize=20
            )

    # -----------------------------
    # 5) Export PNG -> hex (preserve)
    # -----------------------------
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

def build_color_annotations(meta_aligned, annotation_cols):
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
    num_genes = df_reordered.shape[0]
    fig_height = max(10, num_genes * 0.2)

    g = sns.clustermap(
        df_reordered,
        cmap=sns.color_palette("Reds", as_cmap=True),
        figsize=(30, fig_height),
        col_colors=col_colors if col_colors is not None and not col_colors.empty else None,
        col_cluster=False,
        row_cluster=False,
        cbar_kws={"orientation": "vertical", "shrink": 0.5},
        xticklabels=False,
        yticklabels=True
    )

    # Move colorbar left
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.ax.set_position([0.08, 0.25, 0.02, 0.4])
    cbar.set_label("Expression Level", fontsize=12)

    # Fix labels
    # g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=12)
    # g.ax_heatmap.tick_params(axis="y", labelsize=11, pad=10)
    # g.ax_heatmap.set_xticklabels([])
    # g.ax_heatmap.set_ylabel("Genes", fontsize=12)

    if col_colors is not None and not col_colors.empty:
        ax = g.ax_col_colors

        # Move annotation labels to the LEFT
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")

        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
            pad=8,
            labelsize=20
        )

        # Ensure labels are horizontal and readable
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

    # Add annotation legends
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
                title="Annotations",
                loc="upper left",
                bbox_to_anchor=(1.02, 1),   # 👉 move legend to the RIGHT
                ncol=1,                     # vertical stack
                frameon=False,
                prop={"size": 16},
                title_fontsize=18
            )
            

    plt.title("Top Genes Expression Heatmap", fontsize=14, pad=10)
    return g