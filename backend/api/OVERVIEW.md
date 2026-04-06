# api/ Overview

## What Is This?

The `api/` folder is the **HTTP interface layer** of the backend. It contains two things:
- **`routers/`** — FastAPI route handlers (one file per feature area)
- **`schemas/`** — Pydantic models that validate incoming request data

Nothing computationally heavy happens here. Each router function validates the request, calls the appropriate service module, and returns the result. Think of this as the "front door" of the backend.

---

## Structure

```
api/
├── routers/
│   ├── health.py           # GET /healthz
│   ├── deg.py              # DEG analysis endpoints
│   ├── precomputed_deg.py  # Browse saved DEG results
│   ├── nmf.py              # NMF preprocessing + decomposition
│   ├── heatmaps.py         # Heatmap generation endpoints
│   ├── pathway.py          # Pathway/KEGG analysis + gene descriptions
│   ├── metadata.py         # Sample metadata queries
│   └── uploads.py          # S3 multipart upload management
└── schemas/
    ├── deg.py              # DEGGroupsRequest, DEGResearchRequest
    ├── precomputed_deg.py  # DEGResultsGroupsRequest, DEGResultsTermsRequest, DEGResultsFetchRequest
    └── uploads.py          # SignPartRequest, CompleteMultipartRequest
```

---

## Routers

### `health.py`
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/healthz` | GET | `{"ok": True}` |

Used by the frontend to poll whether the EC2 server is ready.

---

### `deg.py`
| Endpoint | Method | Key Parameters | Returns |
|----------|--------|---------------|---------|
| `/deg_submit_groups/` | POST | `group_a: list[str]`, `group_b: list[str]` | ZIP file (DEG results) |
| `/deg_with_research/` | POST | `deg_table` (CSV upload), `disease_context`, `tissue`, `num_genes`, `comparison_description` | JSON (AI insights) |

- `/deg_submit_groups/` → calls `deg.deg_utils.run_deg_analysis()`
- `/deg_with_research/` → calls `research.research_utils.run_deg_with_research_pipeline()`

---

### `precomputed_deg.py`
| Endpoint | Method | Key Parameters | Returns |
|----------|--------|---------------|---------|
| `/deg_results/experiments` | GET | — | `{"experiments": [...]}` |
| `/deg_results/groups` | POST | `experiment: str` | `{"groups": [...]}` |
| `/deg_results/terms` | POST | `experiment`, `group`, `context` | `{"terms": [...]}` |
| `/deg_results/fetch_csv` | POST | `experiment`, `output_dir`, `de_csv_path`, `gsea_barplot_path` | ZIP (CSV + PNG) |

All call functions in `deg.precomputed_deg_utils`. Results come from MongoDB + S3.

---

### `nmf.py`
| Endpoint | Method | Key Parameters | Returns |
|----------|--------|---------------|---------|
| `/preprocess_df` | POST | metadata upload + gene_column, hvg, batch options, job_id | Feather preview |
| `/k_metrics` | POST | meta, job_id, k_min, k_max, max_iter, design_factor | JSON (silhouette scores) |
| `/run_regular_nmf` | POST | job_id, k, max_iter, design_factor | ZIP (W + H matrices) |
| `/run_nmf_files` | POST | metadata upload + k, hvg, max_iter, design_factor, job_id | ZIP (cNMF results) |
| `/cnmf_single_cell` | POST | metadata upload + k, hvg, job_id, batch_correct, gene_column | ZIP (cNMF results) |
| `/explore_correlations` | POST | gene_loadings (feather upload) | JSON (Spearman pairs) |

All delegate to `nmf.nmf_utils`.

---

### `heatmaps.py`
| Endpoint | Method | Key Parameters | Returns |
|----------|--------|---------------|---------|
| `/initial_heatmap_preview/` | POST | df + metadata (feather), metadata_index, annotation_cols | PNG |
| `/plot_heatmap/` | POST | gene_loadings, job_id, metadata, annotation_cols, X (genes per module) | PNG |
| `/cluster_samples/` | POST | module_usages + metadata (feather), k | JSON (dendrogram PNG + cluster labels + leaf order) |
| `/annotated_heatmap/` | POST | module_usages + metadata, leaf_order, annotation_cols, cluster_labels | PNG |
| `/hypergeom/` | POST | module_usages + metadata, cluster_labels, cluster_id, selected_values | JSON (p-value) |
| `/cluster_modules/` | POST | module_usages (feather), sample_order, n_clusters | JSON (dendrogram PNG + leaf order) |
| `/module_heatmap/` | POST | module_usages + metadata, sample_order, module_leaf_order, etc. | JSON (PNG + metadata) |
| `/heatmap_top_samples/` | POST | module_usages + metadata, annotation_cols | JSON (PNG + sample order) |

All delegate to `heatmaps.heatmap_utils`.

---

### `pathway.py`
| Endpoint | Method | Key Parameters | Returns |
|----------|--------|---------------|---------|
| `/process_gene_loadings/` | POST | file (feather), top_n (int) | JSON (gene descriptions per module) |
| `/run_pathview/` | POST | file (CSV with genes + z-scores), gene_format | ZIP (pathway PNGs + kegg_dataframe.csv) |

- `/process_gene_loadings/` → calls `pathway.gene_loadings_gpt.gpt_utils()`
- `/run_pathview/` → writes file to `/tmp`, runs `pathway_analysis.R` via subprocess, returns ZIP

---

### `metadata.py`
| Endpoint | Method | Key Parameters | Returns |
|----------|--------|---------------|---------|
| `/get_metadata/` | GET | — | JSON (columns + unique values) |
| `/get_samples/` | POST | `filters: dict` | Feather (metadata rows) |
| `/get_counts/` | POST | metadata (feather upload) | ZIP (counts preview + job.json) |
| `/update_db/` | POST | counts_table + metadata (feather uploads) | `{"status": "success"}` |

All delegate to `infra.db_utils`.

---

### `uploads.py`
| Endpoint | Method | Key Parameters | Returns |
|----------|--------|---------------|---------|
| `/create_upload_url` | POST | — | `{job_id, s3_key, upload_url}` |
| `/create_preprocessed_upload_url` | POST | — | `{job_id, s3_key, upload_url}` |
| `/download_preprocessed_data` | GET | `job_id`, `data_type` | Redirect to S3 presigned URL |
| `/create_multipart_upload` | GET | — | `{job_id, key, uploadId}` |
| `/sign_part` | POST | `key`, `uploadId`, `partNumber` | `{url}` |
| `/complete_multipart_upload` | POST | `key`, `uploadId`, `parts` | `{location, etag}` |

S3 bucket: `nmf-tool-bucket` (us-east-2). Small files use single presigned PUT; large files use multipart.

---

## Schemas

### `schemas/deg.py`
```python
DEGGroupsRequest:
    group_a: list[str]   # Sample names for comparison group
    group_b: list[str]   # Sample names for reference group

DEGResearchRequest:
    disease_context: str = "Unknown"
    tissue: str = "Unknown"
    num_genes: int = 10
```

### `schemas/precomputed_deg.py`
```python
DEGResultsGroupsRequest:
    experiment: str

DEGResultsTermsRequest:
    experiment: str
    group: str
    context: str = ""

DEGResultsFetchRequest:
    experiment: str
    output_dir: str
    de_csv_path: str
    gsea_barplot_path: str = ""
```

### `schemas/uploads.py`
```python
SignPartRequest:
    key: str
    uploadId: str
    partNumber: int

CompleteMultipartRequest:
    key: str
    uploadId: str
    parts: list   # [{ETag: str, PartNumber: int}, ...]
```

---

## How Routers Are Registered

Routers are imported and mounted in `fast_api.py` (one level up). Each router has a tag that groups endpoints in the auto-generated API docs (`/docs`).

---

## File Format Reference

Most endpoints transfer data as binary files rather than JSON for efficiency:

| Format | Used for |
|--------|---------|
| **Feather** | DataFrames (metadata, module usages, gene loadings) |
| **ZIP** | Bundled results (DEG ZIP, cNMF ZIP, pathway ZIP) |
| **PNG** | Heatmap images (returned as StreamingResponse or hex in JSON) |
| **JSON** | Small structured results, clustering metadata |
| **CSV** | Gene/pathway tables inside ZIPs |
