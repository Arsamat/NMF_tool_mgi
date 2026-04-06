# nmf_nav Overview

## What Is This?

The `nmf_nav` folder contains Streamlit UI pages for **NMF analysis on bulk RNA-seq data**. NMF stands for Non-negative Matrix Factorization — a mathematical technique for breaking a large gene expression matrix into meaningful, interpretable patterns called **modules**.

Each module represents a group of genes that tend to be expressed together, and each sample gets a "usage score" for each module — telling you how strongly that pattern is active in that sample.

---

## The Core Idea (For Beginners)

Imagine you have a matrix **V** where:
- Rows = genes (e.g., 20,000 genes)
- Columns = samples (e.g., 100 samples)

NMF decomposes **V** into two smaller matrices:
- **W** (genes × k): How much each gene contributes to each of the k modules
- **H** (k × samples): How much each sample "uses" each module

You choose **k** — the number of modules. Picking the right k is part of the workflow.

---

## Files at a Glance

```
nmf_nav/
├── home.py                       # Landing page + session state initialization
├── metadata_upload.py            # Step 1: upload sample metadata
├── preprocess2_data.py           # Step 2: normalize and filter the count matrix
├── explore_k_values.py           # Step 3 (optional): find the best k
├── run_light_nmf.py              # Step 4a: quick single NMF run
├── run_consensus_nmf.py          # Step 4b: production-grade consensus NMF
├── gene_descriptions.py          # Step 5: AI summaries of top genes per module
└── upload_counts_helper.py       # Utility: S3 presigned URL upload
```

---

## The Workflow (Step by Step)

### Step 1 — Upload Metadata (`metadata_upload.py`)
Upload a CSV/TSV file with sample names and experimental group assignments.

Example metadata file:
```
SampleName,  Group
Sample_001,  WT
Sample_002,  TDP43
Sample_003,  WT
```

The app also checks if the backend EC2 server is running (and starts it if not).

---

### Step 2 — Upload & Preprocess Counts (`preprocess2_data.py`)
Upload the raw gene count matrix (genes × samples). The backend will:
1. Filter low-expression genes using `edgeR::filterByExpr()`
2. Normalize library sizes (TMM normalization)
3. Log-transform (log CPM)
4. Optionally correct batch effects using `limma::removeBatchEffect()`
5. Select the most variable genes (HVGs — default 2000)
6. Shift values to be non-negative (required for NMF)

The preprocessed matrix is saved to S3 for use in later steps.

**Alternatively**, if you already have a preprocessed matrix, you can skip the R-based pipeline and upload it directly.

---

### Step 3 — Explore K Values (`explore_k_values.py`) *(Optional)*
Not sure how many modules to use? This step runs **light NMF across a range of k values** (e.g., 2 through 10) and computes a **silhouette score** for each — measuring how well the modules separate your sample groups.

A higher silhouette score = the modules better reflect your experimental design.

The output is a scatter plot (one dot per run) with a mean line, helping you pick the "elbow" or peak.

---

### Step 4a — Light NMF (`run_light_nmf.py`)
Run **one NMF decomposition** with a chosen k. This is fast and good for exploration.

After the run, you get:
- **Module usage matrix** (H): how much each sample uses each module
- **Gene loadings matrix** (W): which genes drive each module

**Visualizations available:**
- Preview heatmap
- Sample clustering (dendrogram → pick # clusters → annotated heatmap)
- Module clustering
- Top samples ordering
- Gene expression matrix (top genes per module)
- Hypergeometric enrichment test (are clusters enriched for specific metadata groups?)

---

### Step 4b — Consensus NMF (`run_consensus_nmf.py`)
This is the **production-quality version**. It uses the [Kotliar et al. cNMF method](https://elifesciences.org/articles/43803):
- Runs NMF many times with different random seeds
- Takes a consensus across runs to produce stable, reproducible modules
- Outputs a PDF heatmap + TSV matrices

This runs in the **background** (can take 5–15 minutes). The page auto-refreshes every 5 seconds while waiting. Same visualization tools as light NMF are available after completion.

---

### Step 5 — Gene Descriptions (`gene_descriptions.py`)
Once you have NMF results, you can ask: *"What do the top genes in Module 3 actually do?"*

This step sends the top N genes per module to **OpenAI GPT**, which returns plain-English descriptions of each gene's biological function. Results are shown in a tabbed interface (one tab per module) and are downloadable as CSV.

---

### Step 5 (also) — Module Correlations (`explore_module_correlations.py`)
Calculates **pairwise Spearman correlations** between modules based on their gene loadings. A high correlation means two modules share similar gene expression patterns. The result is displayed as a heatmap and downloadable as CSV.

---

## How Data Flows Through This Module

```
User uploads metadata CSV
        │
        ▼
User uploads raw counts CSV
        │
        ▼
Backend preprocesses (normalize, HVG selection, optional batch correction)
        │ preprocessed matrix saved to S3
        ▼
(Optional) Explore k values → silhouette score plot
        │
        ▼
Run NMF (light or consensus)
        │ W matrix: gene loadings
        │ H matrix: module usages
        ▼
Visualize:
  ├─ Heatmaps (preview, annotated, module, top-samples)
  ├─ Clustered dendrograms
  └─ Gene expression matrices
        │
        ▼
(Optional) Gene descriptions via GPT
(Optional) Module correlation heatmap
```

---

## Key Parameters

| Parameter | What it controls | Typical range |
|-----------|-----------------|---------------|
| **k** | Number of gene modules | 2–20 |
| **hvg** | Highly variable genes to keep | 500–5000 |
| **max_iter** | NMF convergence iterations | 1000–10000 |
| **batch correction** | Remove technical batch effects | On/Off |

---

## Key Backend Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `POST /preprocess_df` | Normalize + filter counts |
| `POST /k_metrics` | Silhouette scores across k values |
| `POST /run_regular_nmf` | Light single NMF run |
| `POST /run_nmf_files` | Consensus NMF |
| `POST /cluster_samples/` | Hierarchical sample clustering |
| `POST /annotated_heatmap/` | Annotated heatmap with metadata bars |
| `POST /initial_heatmap_preview/` | Quick PNG preview |
| `POST /process_gene_loadings/` | AI gene descriptions |
| `POST /explore_correlations` | Module Spearman correlations |

---

## How to Navigate the Code

| If you want to understand... | Start here |
|-----------------------------|-----------|
| What session state exists | `home.py` → `home_page()` |
| The preprocessing pipeline | `preprocess2_data.py` |
| How light NMF runs | `run_light_nmf.py` |
| How consensus NMF runs (background job) | `run_consensus_nmf.py` → `run_cnmf_job()` |
| How S3 uploads work | `upload_counts_helper.py` |
| How gene descriptions work | `gene_descriptions.py` |

---

## Important Notes

- All heavy computation happens on the backend EC2 server — the Streamlit app is just the UI
- Consensus NMF can take 5–15 minutes; do not close the browser tab while it runs
- The backend URL is hardcoded to `http://18.218.84.81:8000/`
- Data is serialized as **Feather format** for fast transfer between frontend and backend
- This module is for **bulk RNA-seq** data; for single-cell data, see `nmf_nav_sc/`
