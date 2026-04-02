# MGI Root-Level Files Overview

## What Lives Here?

The root of the MGI directory contains the **Streamlit application entry point** (`Home.py`), **shared UI components** used across multiple analysis modules, and **configuration files**. These are files that don't belong to a single feature folder but are used broadly across the app.

---

## Files

```
MGI/
├── Home.py                    # App entry point — routing and navigation
├── ui_theme.py                # Global Streamlit page config and CSS theme
├── hypergeometric.py          # Reusable UI: hypergeometric enrichment test widget
├── make_expression_heatmap.py # Reusable UI: gene expression heatmap widget
├── module_clustering.py       # Reusable UI: module clustering (calls backend)
├── module_heatmap.py          # Reusable UI: module heatmap display widget
├── nmf_clustering.py          # Reusable UI: sample/module clustering (local, no backend)
├── preview_heatmap.py         # Reusable UI: heatmap preview (local, no backend)
├── requirements.txt           # Frontend Python dependency list (pinned versions)
└── runtime.txt                # Python runtime version for deployment
```

---

## File Details

---

### `Home.py` — App Entry Point

This is the **first file Streamlit loads**. It acts as the router for the entire application.

**Two modes controlled by `st.session_state["ui_mode"]`:**

**`"home"` mode** — Displays a landing page with a 2-column grid of tool cards:
| Tool | Routes to |
|------|-----------|
| DE Analysis: Browse by Experiment | `deg.experiment_browser.run_experiment_browser()` |
| DE Analysis: De Novo by Samples | `deg.group_selection.run_group_selection()` |
| NMF for Bulk RNA | `nmf_nav.*` subpages |
| NMF for Single-Cell RNA | `nmf_nav_sc.*` subpages |
| Feedback Form | External Microsoft Forms link |

**`"tool"` mode** — Loads the selected tool. For NMF tools, a sidebar `st.radio()` lets the user navigate between subpages (e.g., "Metadata Upload", "Run cNMF", etc.). For DEG tools, the full module is loaded directly.

**Navigation pattern:** Clicking a tool card sets `st.session_state["active_page"]` and `ui_mode = "tool"`, then calls `st.rerun()`. The sidebar "← Back to Home" button resets these to return to the landing page.

**Backend URL:** The DEG tools set `st.session_state["deg_api_url"] = "http://3.141.231.76:8000/"` on load.

---

### `ui_theme.py` — Global Theme

Applies consistent Streamlit page configuration and CSS across the app.

**`apply_custom_theme()`**
- Sets page title to `"NMF Exploration Tool"`, layout to `"wide"`, sidebar initially expanded
- Injects CSS that forces all sidebar text to white (`color: #FFFFFF`)

**Called by:** `Home.py` at the very top before anything else renders.

> Note: `ui_theme.py` contains a top-level `st.set_page_config()` call outside the function. This runs on import and may cause a Streamlit warning if called after the page is already configured.

---

### `hypergeometric.py` — Hypergeometric Test Widget

A reusable Streamlit UI component for running a **hypergeometric enrichment test** — asking: "Is a specific metadata value statistically over-represented in a chosen sample cluster?"

**`hypergeom_ui(meta_bytes, module_usages, cluster_labels)`**

**Parameters:**
- `meta_bytes` — serialized metadata (Feather bytes) to send to the backend
- `module_usages` — module usage DataFrame (samples × modules)
- `cluster_labels` — list of cluster assignments per sample (from a previous clustering step)

**What the user sees:**
1. Dropdown: select which cluster to test
2. Multiselect: pick metadata columns to test
3. For each selected column: dropdown to choose a specific value
4. "Run Hypergeometric Test" button

**On submit:** POSTs to `POST /hypergeom/` with module usages + metadata + cluster info + selected values. Displays the resulting p-value.

**Used by:** `nmf_nav/run_light_nmf.py`, `nmf_nav/run_consensus_nmf.py`, `nmf_nav_sc/run_cnmf_sc.py`

**Session state read:** `st.session_state["meta"]`, `st.session_state["metadata_index"]`, `st.session_state["API_URL"]`

---

### `make_expression_heatmap.py` — Gene Expression Heatmap Widget

A reusable Streamlit form for generating a **gene expression heatmap** showing the top genes per NMF module across all samples.

**`get_expression_heatmap(gene_loadings_df, default_values) → bytes | None`**

**Parameters:**
- `gene_loadings_df` — DataFrame of gene loadings (W matrix from NMF, modules × genes)
- `default_values` — default annotation columns to pre-select in the form

**What the user sees:**
1. Multiselect: choose metadata columns for annotation bars
2. Number input: how many top genes per module (1–20, default 10)
3. "Generate Expression Heatmap" button

**On submit:** POSTs to `POST /plot_heatmap/` with gene loadings + metadata + `job_id` (for the backend to fetch preprocessed counts from S3). Returns raw PNG bytes on success, or displays an error.

**Used by:** `nmf_nav/run_light_nmf.py`, `nmf_nav/run_consensus_nmf.py`, `nmf_nav_sc/run_cnmf_sc.py`

**Session state read:** `st.session_state["meta"]`, `st.session_state["metadata_index"]`, `st.session_state["job_id"]`, `st.session_state["API_URL"]`

---

### `module_clustering.py` — Module Clustering Widget (Backend-Backed)

A reusable function that sends module usage data to the backend to **cluster the NMF modules themselves** (not samples) using hierarchical clustering.

**`m_clustering(module_usages, sample_order, n_clusters_mod, cnmf=False) → BytesIO | None`**

**Parameters:**
- `module_usages` — module usage DataFrame (samples × modules)
- `sample_order` — sample ordering from a previous sample clustering step
- `n_clusters_mod` — number of module clusters; `0` = dendrogram only (no clustering)
- `cnmf` — if `True`, saves results under `cnmf_` prefixed session state keys

**What it does:**
- Serializes module usages to Feather, POSTs to `POST /cluster_modules/`
- If `n_clusters_mod == 0`: returns a dendrogram PNG for preview
- If `n_clusters_mod > 0`: saves `module_leaf_order` and `module_cluster_labels` to session state, returns the dendrogram PNG

**Session state written:**
- `module_leaf_order` / `cnmf_module_leaf_order`
- `module_cluster_labels` / `cnmf_module_cluster_labels`

**Used by:** `nmf_nav/run_light_nmf.py`, `nmf_nav/run_consensus_nmf.py`, `nmf_nav_sc/run_cnmf_sc.py`

---

### `module_heatmap.py` — Module Heatmap Display Widget (Backend-Backed)

A reusable Streamlit form that renders the **final annotated module heatmap** with both samples and modules reordered by clustering.

**`module_heatmap_ui(module_usages, sample_order, module_leaf_order, module_cluster_labels, cnmf=False, default_annotations=[])`**

**What the user sees:**
1. Multiselect: choose metadata annotation columns
2. "Generate Annotated Heatmap" button

**On submit:** POSTs to `POST /module_heatmap/` with module usages + metadata + all ordering/clustering info. On success:
- Saves the PNG to `st.session_state["module_order_heatmap"]` (or `"cnmf_module_order_heatmap"` if `cnmf=True`)
- Displays the image with a download button

Safety checks: displays a warning and returns early if NMF hasn't been run, samples haven't been clustered, or modules haven't been clustered yet.

**Used by:** `nmf_nav/run_light_nmf.py`, `nmf_nav/run_consensus_nmf.py`, `nmf_nav_sc/run_cnmf_sc.py`

---

### `nmf_clustering.py` — Local Hierarchical Clustering (No Backend)

A self-contained clustering module that runs **entirely client-side** using scipy — no backend API call needed.

**`plot_clusters(df_original) → (leaf_order, cluster_labels)`**
- Renders an interactive Streamlit UI for clustering **samples**
- User picks number of clusters (2–10) via a slider
- Runs Ward-linkage hierarchical clustering on Euclidean distances
- Plots a colored dendrogram (branches colored by cluster, mixed clusters go gray)
- Draws a horizontal dashed line at the cutoff height
- **Returns:** `(leaf_order list, cluster_labels Series)`

**`plot_module_clusters(df_modules) → (leaf_order, cluster_labels)`**
- Same flow but clusters **modules** instead of samples
- Gray branches (no color-coding by cluster), but leaf labels are colored by cluster
- **Returns:** `(leaf_order list, cluster_labels Series)`

> This is the local/offline version of what `module_clustering.py` and the backend's `/cluster_samples/` do. It may be used in contexts where a backend call isn't needed or available.

---

### `preview_heatmap.py` — Local Heatmap Preview (No Backend)

A self-contained heatmap preview that runs **entirely client-side** using seaborn — no backend call.

**`preview_wide_heatmap_inline(df, meta, annotation_cols, average_groups, sample_order) → matplotlib.Figure | None`**

**Parameters:**
- `df` — expression/module usage matrix (modules × samples)
- `meta` — sample metadata DataFrame
- `annotation_cols` — which metadata columns to show as color bars
- `average_groups` — if `True`, samples with the same metadata combination are averaged (reduces columns)
- `sample_order` — optional explicit sample ordering; if `None`, samples are ordered by metadata hierarchy

**What it does:**
- Orders samples by metadata hierarchy (or by `sample_order` if provided)
- Optionally groups and averages samples by their metadata combination
- Builds color annotation bars (one color palette per metadata column)
- Renders a seaborn `clustermap` with clustering disabled (preserves sample order)
- Adds a legend for all annotation values

**Returns:** matplotlib Figure object (caller is responsible for displaying it with `st.pyplot()`)

> This is the local version of the backend's `POST /initial_heatmap_preview/`. It's used where an in-process render is preferable.

---

### `requirements.txt` — Frontend Dependencies

Pinned Python packages for the Streamlit frontend. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.48.0 | Web UI framework |
| `pandas` | 2.3.1 | DataFrames |
| `numpy` | 2.0.2 | Numerical arrays |
| `pyarrow` | 21.0.0 | Feather/Parquet serialization |
| `plotly` | 6.3.0 | Interactive charts (DEG module) |
| `matplotlib` | 3.9.4 | Static plots |
| `seaborn` | 0.13.2 | Statistical heatmaps |
| `scipy` | 1.13.1 | Clustering, statistics |
| `scikit-learn` | 1.6.1 | Silhouette scoring |
| `requests` | 2.32.5 | HTTP calls to backend |
| `anthropic` | 0.83.0 | Claude API (natural language features) |
| `openai` | 1.107.1 | OpenAI API (gene descriptions) |
| `streamlit-autorefresh` | 1.0.1 | EC2 wake-up polling |

> This is the **frontend** requirements file. The backend has its own `backend/requirements.txt`.

---

### `runtime.txt` — Deployment Python Version

```
python-3.9
```

Specifies the Python runtime for deployment (e.g., Streamlit Community Cloud or similar PaaS). The actual development environment may use a newer Python version — see `requirements.txt` which notes Python 3.11+ compatibility.

---

## Relationship Between Local and Backend Versions

Several functions exist in two forms — a **local version** (root-level file, runs in the browser process) and a **backend version** (API call to EC2):

| Functionality | Local (root) | Backend (API) |
|--------------|-------------|---------------|
| Heatmap preview | `preview_heatmap.py` | `POST /initial_heatmap_preview/` |
| Sample clustering | `nmf_clustering.plot_clusters()` | `POST /cluster_samples/` |
| Module clustering | `nmf_clustering.plot_module_clusters()` | `POST /cluster_modules/` via `module_clustering.py` |

The local versions are lighter-weight (no network round-trip) but run on the Streamlit server's memory. The backend versions offload computation to EC2 and support larger datasets.
