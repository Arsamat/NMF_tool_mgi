# backend/deg/ Overview

## What Is This?

The `deg/` folder handles **Differential Expression Gene (DEG) analysis** — the statistical comparison of gene expression between two sample groups to find which genes are significantly up- or down-regulated.

It supports two workflows:
1. **Real-time analysis**: user provides two groups of samples → R script runs limma/edgeR → results returned as ZIP
2. **Precomputed results**: query previously saved results stored in MongoDB + S3

---

## Files

```
deg/
├── deg_utils.py              # Real-time DEG analysis via R subprocess
├── precomputed_deg_utils.py  # Query MongoDB/S3 for saved results
└── deg_analysis.R            # The R script that runs the actual statistics
```

---

## File 1: `deg_utils.py`

### Purpose
Orchestrates real-time DEG analysis. Called by the `/deg_submit_groups/` API endpoint.

### Main Function: `run_deg_analysis(group_a, group_b)`

**Parameters:**
- `group_a: list[str]` — sample names for the comparison group (experimental)
- `group_b: list[str]` — sample names for the reference group (control)

**Returns:** `StreamingResponse` — a ZIP file attachment

**What it does, step by step:**
1. Validates both groups have ≥ 2 samples
2. Calls `infra.db_utils.get_counts_subset(all_samples)` to fetch raw gene counts from S3 (parquet shards)
3. Calls `infra.db_utils.get_run_metadata(all_samples)` to get sequencing batch info from MongoDB
4. Builds a metadata DataFrame: each sample gets a "Group" label (GroupA or GroupB) + "Run" (batch)
5. Validates all requested samples exist in the counts matrix
6. Writes `counts.csv` and `metadata.csv` to a temp directory
7. Runs `deg_analysis.R` via subprocess: `Rscript deg_analysis.R counts.csv metadata.csv output.csv human`
8. Reads the outputs and packages them into a ZIP:
   - `deg_analysis.feather` (main DEG results, always present)
   - `heatmap_matrix.csv` + `heatmap_annotation.csv` (optional)
   - `gsea_results.csv` (optional)
9. Returns the ZIP as a streaming HTTP response

**Errors raised:**
- `ValueError` if groups are empty or have < 2 samples
- `ValueError` if samples are not found in the count matrix
- `FileNotFoundError` if `deg_analysis.R` is missing
- `RuntimeError` if the R subprocess fails

---

## File 2: `precomputed_deg_utils.py`

### Purpose
Queries MongoDB for metadata about pre-run DEG analyses, then fetches the actual result files from S3.

### MongoDB Collection: `brb_seq.deg_mapping`
Each document maps: `experiment → group → term_label → S3 paths`

Key fields per document:
```
experiment, group, model_design, context, n_samples, sample_names
term_label, de_csv_path, de_csv_exists, output_dir
gsea_barplot_path, effect_interaction, reference
```

### Functions

**`get_experiments() → list`**
- Returns sorted list of all unique experiment names in the collection

**`get_groups_for_experiment(experiment) → list`**
- Returns one dict per unique `(group, context)` pair within the experiment
- Deduplicates because multiple MongoDB rows exist per group (one per term_label)
- Each dict contains: `group`, `model_design`, `context`, `n_samples`, `sample_names`

**`get_terms_for_group(experiment, group, context="") → list`**
- Returns available contrast terms for a specific group
- Only includes rows where `de_csv_exists == True`
- Each dict contains: `term_label`, `de_csv_path`, `output_dir`, `gsea_barplot_path`, `effect_interaction`, `reference`

**`fetch_deg_results_from_s3(experiment, output_dir, de_csv_path, gsea_barplot_path) → StreamingResponse`**
- Constructs S3 keys:
  - CSV: `deg_data/{experiment}/{basename(de_csv_path)}`
  - PNG: `deg_data/{experiment}/GSEA/{basename(gsea_barplot_path)}`
- Returns a ZIP containing:
  - `deg_results.csv` (always)
  - `gsea_barplot.png` (only if path is non-empty and file exists on S3)
- Raises HTTP 404 if the DEG CSV is not found

---

## File 3: `deg_analysis.R`

### Purpose
The actual statistical engine. Invoked as a subprocess by `deg_utils.py`.

### Input (via command line args)
1. `counts_file` — CSV with genes as rows, samples as columns (first column = gene IDs)
2. `metadata_file` — CSV with columns: `SampleName`, `Group`, `Run` (batch)
3. `output_file` — Where to write the DEG results CSV
4. `species` — `"human"` or `"mouse"` (default: human)

### Statistical Pipeline
1. **Load + align**: reads counts + metadata, filters to common samples
2. **Group setup**: converts `Group` column to factor; `GroupB` is the reference (so `logFC = GroupA - GroupB`)
3. **edgeR DGEList**: builds count object with sample metadata
4. **Gene filtering**: uses `filterByExpr()` to remove lowly-expressed genes
5. **Normalization**: TMM → RLE → none (fallback chain for robustness)
6. **Design matrix**: `~ Group + Run` (accounts for batch effects via the Run column)
7. **Voom**: converts counts to log₂-CPM with precision weights
8. **Linear model + eBayes**: `lmFit()` + `eBayes()` (empirical Bayes shrinkage)
9. **Results**: `topTable()` on the Group coefficient, BH-adjusted p-values

### Gene Annotation
- Calls `build_gene_table_from_ensembl()` which queries **biomaRt** (Ensembl REST API)
- Adds: `SYMBOL`, `GENENAME`, `ENTREZID`, `BIOTYPE`, `CHROMOSOME` columns
- Tries primary Ensembl mirror first, falls back to asia mirror

### Outputs
| File | Description |
|------|-------------|
| `deg_results.csv` | Columns: gene, SYMBOL, GENENAME, logFC, AveExpr, t, P.Value, adj.P.Val, B |
| `heatmap_matrix.csv` | Top 30 most variable genes (log₂ CPM), hierarchically ordered |
| `heatmap_annotation.csv` | Sample metadata: SampleName, Group |
| `gsea_results.csv` | GSEA hallmark pathway enrichment (if packages available) |

### GSEA (Optional)
- Requires: `clusterProfiler`, `msigdbr`, `org.Hs.eg.db` / `org.Mm.eg.db`
- Maps Ensembl IDs → Entrez IDs; handles duplicate mappings by max |logFC|
- Runs GSEA with hallmark gene sets; pvalueCutoff = 0.25
- Silently skipped if packages are missing or too few genes

---

## Data Flow Summary

```
API: /deg_submit_groups/
    │
    ▼
deg_utils.run_deg_analysis(group_a, group_b)
    │
    ├─ infra.db_utils.get_counts_subset()   → S3 parquet shards
    ├─ infra.db_utils.get_run_metadata()    → MongoDB
    │
    ▼
Write temp counts.csv + metadata.csv
    │
    ▼
subprocess: Rscript deg_analysis.R
    │
    ▼
deg_results.csv  [+ optional heatmap CSVs + gsea_results.csv]
    │
    ▼
Bundle into ZIP → StreamingResponse
```

```
API: /deg_results/fetch_csv
    │
    ▼
precomputed_deg_utils.fetch_deg_results_from_s3()
    │
    ├─ MongoDB: deg_mapping collection → get file paths
    └─ S3: brb-seq-data-storage → download CSV + PNG
    │
    ▼
Bundle into ZIP → StreamingResponse
```
