import pandas as pd
from plot_heatmap_helpers import create_dendrogram, make_heatmap_png, build_annotations, dendogram_modules, module_annotated_heatmap
from plot_heatmap_helpers import build_color_annotations, make_heatmap
from scipy.stats import hypergeom
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import io
from fastapi.responses import JSONResponse
import numpy as np
from fastapi.responses import StreamingResponse
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import tempfile
import boto3
import os
from fastapi import HTTPException
from starlette.background import BackgroundTask
import shutil
import colorcet as cc

BUCKET = "nmf-tool-bucket"
REGION = "us-east-2"

s3 = boto3.client("s3", region_name=REGION)

async def preview_wide_heatmap_inline(df, meta=None, metadata_index=None, annotation_cols=None, 
        average_groups=False, sample_order=None
):
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
        if isinstance(metadata_index, (list, tuple)):
            metadata_index = metadata_index[0]

        # Index metadata by sample ID
        meta_indexed = meta.set_index(metadata_index)
        
    
        # Keep only samples present in the heatmap
        common_samples = [s for s in samples if s in meta_indexed.index]
        if not common_samples:
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
        # palettes = [
        #     "Set1", "Set2", "Paired", "Pastel1", "Pastel2",
        #     "Dark2", "Accent", "tab10", "tab20"
        # ]
        

        # col_colors = pd.DataFrame(index=df_small.columns)
        # for i, col in enumerate(annotation_cols):
        #     if col not in meta_sorted.columns:
        #         continue
        #     unique_vals = pd.unique(meta_sorted[col].dropna())
        #     palette = sns.color_palette(palettes[i % len(palettes)], n_colors=len(unique_vals))
        #     lut[col] = dict(zip(unique_vals, palette))
        #     # Safe mapping: avoids MultiIndex error
        #     col_colors[col] = meta_sorted[col].astype(object).map(lambda x: lut[col].get(x, None))

        col_colors = pd.DataFrame(index=df_small.columns)
        lut = {}

        # 1) Collect all category "tokens" across ALL annotation columns
        tokens = []
        for col in annotation_cols:
            if col not in meta_sorted.columns:
                continue
            vals = pd.unique(meta_sorted[col].dropna().astype(object))
            tokens.extend([(col, v) for v in vals])

        # 2) Make one big palette for every (col, value) pair
        #palette = sns.color_palette("husl", len(tokens))
        n = 60
        palette = cc.glasbey[:len(tokens)]
        global_lut = {tok: palette[i] for i, tok in enumerate(tokens)}

        # 3) Build per-column LUTs + col_colors from the global LUT
        for col in annotation_cols:
            if col not in meta_sorted.columns:
                continue

            vals = pd.unique(meta_sorted[col].dropna().astype(object))
            lut[col] = {v: global_lut[(col, v)] for v in vals}

            col_colors[col] = (
                meta_sorted[col].astype(object)
                .map(lut[col])
            )

    

    # --- Plot heatmap ---
    g = sns.clustermap(
        df_small,
        col_colors=col_colors if col_colors is not None and not col_colors.empty else None,
        cmap="viridis",
        figsize=(26, 14),   # wider figure
        col_cluster=False,
        row_cluster=False
    )


    if col_colors is not None and not col_colors.empty:
        ax = g.ax_col_colors
        # n_rows = col_colors.shape[1]

        # # Draw separators between annotation rows (ONLY if > 1 row)
        # if n_rows > 1:
        #     xmin, xmax = ax.get_xlim()
        #     for y in range(1, n_rows):
        #         ax.hlines(
        #             y - 0.5, xmin, xmax,
        #             colors="white", linewidth=1.2, zorder=10
        #         )

        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_label_position("left")

        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
            pad=6,
            labelsize=20
        )

        # Optional: ensure labels are horizontal
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


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
            loc="upper left",
            bbox_to_anchor=(1.02, 1),   # move right of plot
            borderaxespad=0,
            ncol=1,
            prop={"size": 20}
        )

    
    buf = io.BytesIO()
    g.fig.savefig(buf, format="png", dpi=600, bbox_inches="tight")
    plt.close(g.fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



async def cluster_samples_utils(module_usages,
    metadata,
    metadata_index,
    k):

    df = pd.read_feather(io.BytesIO(await module_usages.read()))
    df.columns = ["Sample"] + list(df.columns[1:])
    df = df.set_index("Sample")

    meta = pd.read_feather(io.BytesIO(await metadata.read())).set_index(metadata_index)

    # --- clustering
    dist = pdist(df, metric="euclidean")
    Z = linkage(dist, method="ward")
    cluster_labels = fcluster(Z, k, criterion="maxclust")

    # --- dendrogram
    if k != 0:
        leaf_order, dendro_png = create_dendrogram(Z, cluster_labels, "Sample Clustering")
    else:
        leaf_order, dendro_png = create_dendrogram(Z, title="Initial Sample Clustering")


    # --- reorder df
    ordered_cols = df.index[leaf_order]

    heatmap_png = make_heatmap_png(df.loc[ordered_cols].T)

    return {
        "leaf_order": leaf_order,
        "cluster_labels": cluster_labels.tolist(),
        "sample_order": ordered_cols.tolist(),
        "dendrogram_png": dendro_png.getvalue().hex(),
        "heatmap_png": heatmap_png.getvalue().hex(),
    }


def cleanup_after_send(dirs=None):
    dirs  = dirs or []
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

#this function plots top unique genes from each module and how those genes are expressed in each sample
async def plot_expression_heatmap(
    gene_loadings,
    #preprocessed_df,
    job_id,
    metadata,
    annotation_cols,
    metadata_index,
    X
):
    """
    Returns a PNG heatmap computed fully in-memory.
    """
    # -----------------------------
    # Load feather inputs
    # -----------------------------
    gl_bytes = await gene_loadings.read()
    #pp_bytes = await preprocessed_df.read()
    preprocessed_tmp = tempfile.mkdtemp(prefix="preprocess_")
    preprocessed_path = os.path.join(preprocessed_tmp, "preprocessed_counts.csv")
    s3_key = f"jobs/{job_id}/preprocessed_counts.csv"


    try:
        s3.download_file(BUCKET, s3_key, preprocessed_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not download counts from S3 (Bucket={BUCKET}, Key={s3_key}). Error: {e}"
        )
    #df_expr = pd.read_feather(io.BytesIO(await preprocessed.read()))
    meta_bytes = await metadata.read()

    gene_loadings_df = pd.read_feather(io.BytesIO(gl_bytes))
    df = pd.read_csv(preprocessed_path)
    df = df.set_index("Unnamed: 0")
    df = np.log10(df + 1)
    df = df.reset_index()

    meta = pd.read_feather(io.BytesIO(meta_bytes))

    # Convert annotation_cols string → list
    
    annotation_cols = json.loads(annotation_cols)

    # -----------------------------
    # Extract top X genes per module
    # -----------------------------
    df_long = gene_loadings_df.stack().reset_index()
    df_long.columns = ["module", "gene", "loading"]
    df_long = df_long.drop(0, errors="ignore")

    unique_df = df_long.loc[df_long.groupby("gene")["loading"].idxmax()]

    grouped_genes = (
        unique_df.sort_values(["module", "loading"], ascending=[True, False])
        .groupby("module")
        .head(X)
    )

    module_labels = grouped_genes[["module", "gene"]]
    cluster_boundaries = []

    for i in range(1, len(module_labels)):
        if module_labels.iloc[i, 0] != module_labels.iloc[i - 1, 0]:
            cluster_boundaries.append(i)

    genes_graph = list(grouped_genes["gene"])

    # -----------------------------
    # Build expression matrix
    # -----------------------------
    
    df = df.set_index("Unnamed: 0")
    df = df.T.reset_index()
    df = df.rename(columns={"index": "Geneid"})


    df = df[df["Geneid"].isin(genes_graph)]
    df["Category"] = pd.Categorical(df["Geneid"], categories=genes_graph, ordered=True)
    df_ordered = df.sort_values("Category").drop(columns={"Category"})
    df_plot = df_ordered.set_index("Geneid")

    # -----------------------------
    # Align metadata
    # -----------------------------
    meta = meta.set_index(metadata_index)
    meta_aligned = meta.loc[df_plot.columns.intersection(meta.index)]

    # -----------------------------
    # Build annotation & plot
    # -----------------------------
    col_colors, lut = build_annotations(meta_aligned, annotation_cols)
    g = make_heatmap(df_plot, col_colors, lut, annotation_cols)

    # Add cluster separation lines
    for y in cluster_boundaries:
        g.ax_heatmap.hlines(
            y=y,
            xmin=0,
            xmax=df_plot.shape[1],
            colors="black",
            linewidth=3,
            linestyles="-",
            zorder=10,
        )

    # -----------------------------
    # Return PNG buffer
    # -----------------------------
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    buf.seek(0)

    
    cleanup_after_send(
        dirs=[preprocessed_tmp]
    )

    return StreamingResponse(buf, media_type="image/png")

#this function produces a heatmap that is ordered based on the sample ordering obtained from running hierarchical clustering
async def annotated_heatmap_util(
    module_usages,
    metadata,
    metadata_index,
    leaf_order,
    annotation_cols,
    cluster_labels
):
    leaf_order = json.loads(leaf_order)
    annotation_cols = json.loads(annotation_cols)
    cluster_labels = json.loads(cluster_labels)

    # Load module usage matrix
    df = pd.read_feather(io.BytesIO(await module_usages.read()))
    df.columns = ["Sample"] + list(df.columns[1:])
    df = df.set_index("Sample")

    # Compute sample order from leaf indices
    sample_order = df.index[leaf_order].tolist()
    ordered_df = df.loc[sample_order].T

    # Load metadata aligned to sample order
    meta = pd.read_feather(io.BytesIO(await metadata.read())).set_index(metadata_index)
    meta_aligned = meta.loc[sample_order]

    # Handle Cluster annotation
    if "Cluster" in annotation_cols:

        if len(cluster_labels) != len(df.index):
            raise ValueError("Cluster label length mismatch")

        cluster_labels_ordered = [cluster_labels[i] for i in leaf_order]
        meta_aligned["Cluster"] = cluster_labels_ordered

    # # Build annotations
    # col_colors_df, lut = build_annotations(meta_aligned, annotation_cols)

    # # IMPORTANT: pass DataFrame directly — NO transpose
    # col_colors = col_colors_df

    if len(annotation_cols) == 0:
        col_colors = None
        lut = {}
    else:
        col_colors, lut = build_annotations(meta_aligned, annotation_cols)
    
    print(col_colors)
    print(lut)
    # Render heatmap
    heatmap_png = make_heatmap_png(
        df=ordered_df,
        col_colors=col_colors,
        lut=lut,
    )

    return StreamingResponse(heatmap_png, media_type="image/png")


#this function produces a heatmap that orders the samples based on top expression in each module 
async def heatmap_top_samples_utils(
    module_usages,
    metadata,
    annotation_cols,
    metadata_index
):

    # -----------------------------------------------------
    # Load DataFrames
    # -----------------------------------------------------
    df_usage = pd.read_feather(io.BytesIO(await module_usages.read()))
    df_usage = df_usage.set_index(df_usage.columns[0])

    meta = pd.read_feather(io.BytesIO(await metadata.read()))
    meta = meta.set_index(metadata_index)

    annotation_cols_list = (
        [c for c in annotation_cols.split(",") if c.strip()]
        if annotation_cols else []
    )

    # -----------------------------------------------------
    # TOP MODULE ORDERING
    # -----------------------------------------------------
    sample_assignments = df_usage.idxmax(axis=1)
    df_usage["TopModule"] = sample_assignments

    #Calculate annotation percentages in each module


    def numeric_key(x):
        try:
            return int(x.split("_")[-1])
        except:
            return x

    modules_sorted = sorted(sample_assignments.unique(), key=numeric_key)

    ordered_samples = []
    for mod in modules_sorted:
        subset = df_usage[df_usage["TopModule"] == mod]
        subset = subset.sort_values(by=mod, ascending=False)
        ordered_samples.extend(subset.index.tolist())
    

    df_plot = df_usage.drop(columns=["TopModule"]).T
    df_plot = df_plot[ordered_samples]

    if len(annotation_cols_list) == 0:
        col_colors = None
        lut = {}
    else:
        col_colors, lut = build_annotations(meta.loc[ordered_samples], annotation_cols_list)

    buf = make_heatmap_png(df_plot, col_colors=col_colors, lut=lut)


    # return hex string instead of bytes
    return {
        "heatmap_png": buf.getvalue().hex(),
        "ordered_samples": ordered_samples,
    }

#this function produces a heatmap obtained from re-ordering the heatmap by modules after running hierarchical clustering on
async def module_heatmap_utils(
    module_usages,
    metadata,
    metadata_index,
    sample_order,
    module_leaf_order,
    module_cluster_labels,
    annotation_cols
):

    # ----------------------------------------------------------
    # 1. Decode incoming JSON lists
    # ----------------------------------------------------------
    sample_order = json.loads(sample_order)
    module_leaf_order = [int(i) for i in json.loads(module_leaf_order)]
    module_cluster_labels = [int(i) for i in json.loads(module_cluster_labels)]
    annotation_cols = json.loads(annotation_cols)

    # ----------------------------------------------------------
    # 2. Load module usages
    # ----------------------------------------------------------
    df = pd.read_feather(io.BytesIO(await module_usages.read()))
    df.columns = ["Sample"] + list(df.columns[1:])
    df = df.set_index("Sample")

    # Reorder samples by preserved sample order
    df = df.loc[sample_order]
    df = df.T  # modules × samples

    # ----------------------------------------------------------
    # 3. Reorder modules by module_leaf_order
    # ----------------------------------------------------------
    df_reordered = df.iloc[module_leaf_order, :]

    # Also reorder module cluster labels to match the heatmap
    cluster_labels_ordered = [module_cluster_labels[i] for i in module_leaf_order]

    # ----------------------------------------------------------
    # 4. Load metadata for column annotations
    # ----------------------------------------------------------
    meta = pd.read_feather(io.BytesIO(await metadata.read()))
    meta = meta.set_index(metadata_index)
    meta = meta.loc[sample_order]   # same order as heatmap columns

    return module_annotated_heatmap(df_reordered, meta, annotation_cols, cluster_labels_ordered, module_leaf_order)


async def hypergeom_endpoint_utils(
    module_usages,
    metadata,
    metadata_index,
    cluster_labels,
    cluster_id,
    selected_values
):
    # Decode JSON inputs
    cluster_labels = json.loads(cluster_labels)
    selected_values = json.loads(selected_values)

    # ---------------------------------------------------------
    # Load dataframes
    # ---------------------------------------------------------
    mod = pd.read_feather(io.BytesIO(await module_usages.read()))
    mod.columns = ["Sample"] + list(mod.columns[1:])
    mod = mod.set_index("Sample")

    meta = pd.read_feather(io.BytesIO(await metadata.read()))
    meta = meta.set_index(metadata_index)

    # ---------------------------------------------------------
    # Step 1: Identify samples in selected cluster
    # ---------------------------------------------------------
    cluster_labels = np.array(cluster_labels)
    cluster_mask = (cluster_labels == cluster_id)
    sample_order = mod.index[cluster_mask]   # samples in this cluster

    # ---------------------------------------------------------
    # Step 2: Hypergeometric variables
    # ---------------------------------------------------------
    M = len(mod.index)                 # total samples
    N = len(sample_order)              # samples in selected cluster

    # n = number of samples matching metadata condition
    mask_n = pd.Series(True, index=meta.index)
    for col, val in selected_values.items():
        mask_n &= (meta[col] == val)
    n = mask_n.sum()

    # k = samples matching metadata condition inside the cluster
    meta_filtered = meta.loc[sample_order]
    mask_k = pd.Series(True, index=meta_filtered.index)
    for col, val in selected_values.items():
        mask_k &= (meta_filtered[col] == val)
    k = mask_k.sum()

    # ---------------------------------------------------------
    # Step 3: Compute p-value
    # ---------------------------------------------------------
    p_value = hypergeom.sf(k - 1, M, n, N)

    return JSONResponse({
    "cluster_id": int(cluster_id),
    "M_total_samples": int(M),
    "N_cluster_samples": int(N),
    "n_matching_metadata": int(n),
    "k_cluster_metadata_intersection": int(k),
    "p_value": float(p_value)
})