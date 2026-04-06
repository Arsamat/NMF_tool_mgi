# backend/heatmaps/ Overview

## What Is This?

The `heatmaps/` folder generates all heatmap visualizations and performs hierarchical clustering for the NMF analysis workflow. It takes gene expression matrices and NMF module usage matrices as input and returns PNG images or JSON with encoded images.

---

## Files

```
heatmaps/
├── heatmap_utils.py        # Main async functions called by API endpoints
└── plot_heatmap_helpers.py # Low-level plotting, clustering, and color utilities
```

The pattern: `heatmap_utils.py` handles data loading and orchestration; `plot_heatmap_helpers.py` handles the actual matplotlib/seaborn rendering.

---

## File 1: `heatmap_utils.py`

All functions here are `async` and are called directly by the `/` heatmap API endpoints.

---

### `preview_wide_heatmap_inline(df, meta, metadata_index, annotation_cols, average_groups, sample_order)`
**Called by:** `POST /initial_heatmap_preview/`

**What it does:**
- Renders a quick preview heatmap with metadata annotation bars on top
- Samples are ordered by metadata hierarchy (optionally averaged by group to reduce columns)
- Uses seaborn's `clustermap` with clustering disabled (preserves sample order)

**Returns:** `StreamingResponse` (PNG)

---

### `cluster_samples_utils(module_usages, metadata, metadata_index, k)`
**Called by:** `POST /cluster_samples/`

**What it does:**
1. Reads module usage matrix (Feather format)
2. Computes pairwise Euclidean distances between samples
3. Runs Ward-linkage hierarchical clustering
4. Cuts the dendrogram into `k` clusters and assigns each sample a cluster label
5. Renders a colored dendrogram (branches colored by cluster)
6. Renders a heatmap with samples reordered by the dendrogram leaf order

**Returns:** JSON with:
- `dendrogram_png` (hex-encoded PNG)
- `heatmap_png` (hex-encoded PNG)
- `leaf_order` (sample ordering indices)
- `cluster_labels` (cluster assignment per sample)
- `sample_order` (sample names in new order)

> **This is usually the first step** — its output (`leaf_order`, `cluster_labels`) feeds into `annotated_heatmap_util()` and `module_heatmap_utils()`.

---

### `annotated_heatmap_util(module_usages, metadata, metadata_index, leaf_order, annotation_cols, cluster_labels)`
**Called by:** `POST /annotated_heatmap/`

**What it does:**
- Reorders the module usage matrix by the `leaf_order` from the clustering step
- Injects computed `cluster_labels` as an extra annotation column ("Cluster")
- Renders a heatmap with colored metadata annotation bars

**Returns:** `StreamingResponse` (PNG)

---

### `plot_expression_heatmap(gene_loadings, job_id, metadata, annotation_cols, metadata_index, X)`
**Called by:** `POST /plot_heatmap/`

**What it does:**
- Takes the NMF gene loadings (W matrix) and picks the top `X` genes per module
- Downloads the preprocessed count matrix from S3 (`jobs/{job_id}/preprocessed_counts.csv`)
- Log-transforms expression: `log10(x + 1)`
- Draws a heatmap where rows = genes (grouped by module), columns = samples
- Draws black vertical lines at module cluster boundaries

**Returns:** `StreamingResponse` (PNG)

---

### `heatmap_top_samples_utils(module_usages, metadata, annotation_cols, metadata_index)`
**Called by:** `POST /heatmap_top_samples/`

**What it does:**
- Identifies each sample's "top module" (the module with the highest usage value)
- Sorts modules numerically, then sorts samples within each module by descending usage
- Renders a heatmap in this custom order

**Returns:** JSON with `heatmap_png` (hex) and `ordered_samples` (list)

---

### `module_heatmap_utils(module_usages, metadata, metadata_index, sample_order, module_leaf_order, module_cluster_labels, annotation_cols)`
**Called by:** `POST /module_heatmap/`

**What it does:**
- Takes both sample order (from sample clustering) and module order (from module clustering)
- Reorders both axes simultaneously
- Draws white horizontal lines between module clusters
- Returns the color lookup table (`lut`) so the frontend can render a consistent legend

**Returns:** JSON with `heatmap_png` (hex), `module_leaf_order`, `cluster_labels`, `boundaries`, `lut`

---

### `hypergeom_endpoint_utils(module_usages, metadata, metadata_index, cluster_labels, cluster_id, selected_values)`
**Called by:** `POST /hypergeom/`

**What it does:**
- Runs a hypergeometric test to ask: "Is a given metadata value enriched in a specific sample cluster?"
- Computes the standard M, N, n, k values:
  - M = total samples
  - N = samples in the selected cluster
  - n = samples matching the metadata condition
  - k = samples in the cluster that also match the condition
- Returns the one-tailed p-value: `P(X ≥ k)`

**Returns:** JSON with p-value and the four count values

---

## File 2: `plot_heatmap_helpers.py`

Lower-level functions that deal with actual matplotlib/seaborn figures.

---

### `create_dendrogram(Z, cluster_labels, title)`
- Takes a scipy linkage matrix `Z`
- Colors leaf nodes and branches by cluster assignment (husl palette per cluster)
- Mixed-cluster branches are colored gray
- **Returns:** `(leaf_order list, BytesIO PNG buffer)`

---

### `make_heatmap_png(df, col_colors, lut)`
- Core heatmap renderer: 32×18 inch seaborn clustermap, viridis colormap, no clustering
- Annotation bars drawn on top, labels on the left at 20pt
- Legend positioned to the right
- **Returns:** `BytesIO PNG buffer`

---

### `build_annotations(meta, annotation_cols, use_glasbey=True)`
- Assigns colors to all `(column, value)` pairs across all annotation columns
- Uses a single global **Glasbey palette** (60 colors) so no two values share a color even across different columns
- **Returns:** `(col_colors DataFrame, lut dict {col: {value: color}})`

---

### `dendogram_modules(df, n_clusters)`
- Clusters modules (not samples) using Ward linkage
- Colors leaf labels and branches by module cluster
- **Called by:** `POST /cluster_modules/`
- **Returns:** JSON with `module_leaf_order`, `cluster_labels`, `dendrogram_png` (hex)

---

### `module_annotated_heatmap(df_reordered, meta, annotation_cols, cluster_labels_ordered, module_leaf_order)`
- Full module+sample heatmap with white boundary lines between module clusters
- Returns both the image and the boundary positions so the frontend knows where clusters start/end
- **Returns:** JSON with `heatmap_png`, `module_leaf_order`, `cluster_labels`, `boundaries`, `lut`

---

### `make_heatmap(df_reordered, col_colors, lut, annotation_cols)`
- Gene expression-specific heatmap (red colormap instead of viridis)
- Dynamic figure height based on number of genes
- **Returns:** matplotlib ClusterMap object (not PNG)

---

## Typical Visualization Sequence

Most users follow this sequence:

```
1. POST /initial_heatmap_preview/
   → Quick look at module usage with metadata bars

2. POST /cluster_samples/  (k=0 first for dendrogram, then k=N for clustering)
   → Hierarchical clustering of samples
   → Returns leaf_order + cluster_labels

3. POST /annotated_heatmap/
   → Full heatmap with samples reordered + cluster annotations

4. POST /cluster_modules/
   → Cluster the modules themselves
   → Returns module_leaf_order

5. POST /module_heatmap/
   → Final heatmap with both samples and modules reordered + boundary lines

6. POST /plot_heatmap/       (optional)
   → Show which genes drive each module (from S3 preprocessed counts)

7. POST /hypergeom/          (optional)
   → Test if clusters are enriched for metadata groups
```

---

## Styling Defaults

| Property | Value |
|----------|-------|
| Main colormap | viridis |
| Gene expression colormap | Reds |
| Cluster branch colors | husl palette (8–10 colors) |
| Mixed-cluster branches | gray |
| Annotation colors | Glasbey palette (60 colors) |
| Cluster boundary lines | white, 3pt width |
| Heatmap size | 26×14 inches (module) / 30×dynamic (gene) |
| DPI | 200–400 (higher for gene expression) |
| Annotation labels | left-side, 20pt, horizontal |
| Legend | right-side, no frame, 20pt |
