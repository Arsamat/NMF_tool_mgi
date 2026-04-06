# nmf_nav_sc Overview

## What Is This?

The `nmf_nav_sc` folder contains Streamlit UI pages for **NMF analysis on single-cell (or pseudobulk) RNA-seq data**. It is the single-cell counterpart of `nmf_nav/`.

NMF (Non-negative Matrix Factorization) decomposes a gene expression matrix into **modules** — groups of co-expressed genes — and tells you how strongly each sample "uses" each module.

This module uses the **cNMF algorithm** (Kotliar et al., 2019), which runs NMF many times with different random seeds and takes a consensus to produce stable, reproducible results.

---

## The Core Idea (For Beginners)

You have a gene expression matrix **V**:
- Rows = genes
- Columns = samples (or cells in pseudobulk)

NMF finds **k** gene modules that explain the patterns in the data:
- **W** (genes × k): how much each gene drives each module
- **H** (k × samples): how much each sample "activates" each module

You choose **k**. Higher k = finer-grained modules, but potentially noisier.

---

## How This Differs From `nmf_nav/`

| Feature | `nmf_nav/` (bulk) | `nmf_nav_sc/` (single-cell) |
|---------|------------------|----------------------------|
| Input data | Bulk RNA-seq counts | Single-cell/pseudobulk counts |
| NMF method | Light NMF or Consensus NMF | Consensus NMF only (cNMF) |
| File upload | Standard single upload | **Multipart S3 upload** (supports very large files) |
| Preprocessing | In backend (R) | In backend (Python/scanpy) |
| K selection | Silhouette score page | Not included (pick k directly) |

---

## Files at a Glance

```
nmf_nav_sc/
├── home_sc.py                        # Landing page + session state initialization
├── metadata_upload_sc.py             # Step 1: upload sample metadata
├── upload_counts_helper_sc.py        # Utility: multipart S3 upload for large files
├── run_cnmf_sc.py                    # Step 2: run cNMF + all visualizations
├── gene_descriptions_sc.py           # Step 3: AI summaries of top genes per module
└── explore_module_correlations.py    # Step 4: pairwise module similarity heatmap
```

---

## The Workflow (Step by Step)

### Step 1 — Upload Metadata (`metadata_upload_sc.py`)
Upload a CSV/TSV file with sample names and experimental groups.

Example:
```
SampleName,   Group
Cell_001,     WT
Cell_002,     TDP43
```

Specify:
- Which column contains sample names
- Which column contains the experimental group (design factor)

The app also checks if the backend EC2 server is running and starts it if needed (via AWS Lambda). Expect 1–4 minutes for the server to wake up.

---

### Step 2 — Upload Counts & Run cNMF (`run_cnmf_sc.py`)

**Upload counts:**
The count matrix (genes × samples) is uploaded directly to **AWS S3** using multipart upload — this supports very large files (hundreds of MB) by splitting them into 128 MB chunks and uploading up to 6 chunks in parallel.

**Configure parameters:**
| Parameter | What it controls |
|-----------|-----------------|
| **k** | Number of gene modules (2–50) |
| **hvg** | Highly variable genes to use (100–20,000) |
| **gene_column** | Which column identifies genes (e.g., "GeneID") |
| **batch correction** | Remove technical batch effects from the metadata |

**Run cNMF:**
Click "Run cNMF". The job runs **in the background** on the EC2 server (`POST /cnmf_single_cell`). The page auto-refreshes every 5 seconds while waiting (typically 5–15 minutes).

**Results returned (as a ZIP file):**
- PDF heatmap of module usage across samples
- Module usages matrix TSV (samples × modules)
- Gene loadings matrix TSV (modules × genes)

---

### Visualizations (After cNMF Completes)

All visualizations are built from the two output matrices (W and H):

| Visualization | What it shows |
|--------------|---------------|
| **Preview Heatmap** | Quick PNG of module usages across samples |
| **Sample Clustering** | Hierarchical dendrogram → pick # clusters → annotated heatmap |
| **Module Clustering** | Groups similar modules together |
| **Top Samples Ordering** | Reorders samples so each module's "top users" are grouped |
| **Gene Expression Matrix** | Shows top genes from each module across all samples |
| **Hypergeometric Test** | Are the sample clusters statistically enriched for metadata groups? |

All visualizations are downloadable as PNG or CSV.

---

### Step 3 — Gene Descriptions (`gene_descriptions_sc.py`)
Sends the top N genes per module to the backend, which uses **OpenAI GPT** to generate plain-English biological descriptions.

Results are shown in tabs (one per module) and can be downloaded as CSV.

---

### Step 4 — Module Correlations (`explore_module_correlations.py`)
Calculates **pairwise Spearman correlations** between modules based on their gene loadings. Displayed as a heatmap — high correlation means two modules share similar gene patterns. Downloadable as PNG + CSV.

---

## How Data Flows Through This Module

```
User uploads metadata CSV
        │
        ▼
User uploads raw counts (multipart → S3)
        │
        ▼
Backend runs cNMF (background, 5–15 min)
        │ W matrix: gene loadings (modules × genes)
        │ H matrix: module usages (samples × modules)
        │ PDF heatmap
        ▼
Frontend parses ZIP results
        │
        ▼
Visualize:
  ├─ Preview heatmap
  ├─ Sample clustering dendrograms
  ├─ Annotated heatmaps
  └─ Gene expression matrices
        │
        ▼
(Optional) Gene descriptions via GPT
(Optional) Module correlation heatmap
```

---

## The Multipart Upload (Why It's Different)

Single-cell count matrices can be very large (hundreds of MB to GB). The standard single-file upload would time out. Instead:

1. The app requests an **upload ID** from the backend
2. The file is split into **128 MB chunks**
3. Up to **6 chunks** are uploaded in parallel to S3
4. Each chunk gets an **ETag** from S3
5. Once all chunks finish, the backend **finalizes** the upload

This is all handled transparently via JavaScript embedded in the Streamlit page.

---

## Key Backend Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | Check if server is ready |
| `GET /create_multipart_upload` | Initialize multipart S3 upload |
| `POST /sign_part` | Get presigned URL for each chunk |
| `POST /complete_multipart_upload` | Finalize upload |
| `POST /cnmf_single_cell` | Run cNMF algorithm |
| `POST /initial_heatmap_preview/` | Quick PNG preview |
| `POST /cluster_samples/` | Hierarchical sample clustering |
| `POST /annotated_heatmap/` | Annotated heatmap with metadata bars |
| `POST /heatmap_top_samples/` | Reorder by top module samples |
| `POST /process_gene_loadings/` | AI gene descriptions |
| `POST /explore_correlations` | Module pairwise correlations |

---

## How to Navigate the Code

| If you want to understand... | Start here |
|-----------------------------|-----------|
| Session state variables | `home_sc.py` → `home_page_sc()` |
| The metadata upload flow | `metadata_upload_sc.py` |
| The multipart S3 upload | `upload_counts_helper_sc.py` |
| The cNMF run + all visualizations | `run_cnmf_sc.py` |
| Background job pattern | `run_cnmf_sc.py` → `run_cnmf_job()` |
| Gene descriptions | `gene_descriptions_sc.py` |

---

## Important Notes

- This module is for **single-cell or pseudobulk** RNA-seq data; for bulk RNA-seq, see `nmf_nav/`
- The cNMF run can take 5–15 minutes — do not close the browser tab
- All session state variables end with `_sc` to avoid conflicts with the bulk NMF module
- The backend URL is hardcoded to `http://18.218.84.81:8000/`
- Batch correction uses metadata columns as covariates (passed to scanpy/cNMF)
