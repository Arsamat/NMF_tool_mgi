# backend/nmf/ Overview

## What Is This?

The `nmf/` folder contains the **NMF (Non-negative Matrix Factorization) analysis engine**. NMF decomposes a gene expression matrix **V** into:
- **W** (samples × k): how much each sample "uses" each module
- **H** (k × genes): which genes drive each module

This folder handles everything from preprocessing raw counts to running the factorization to statistical quality assessment.

---

## Files

```
nmf/
├── nmf_utils.py               # Async HTTP handlers — bridges API routes to NMF functions
├── NMF.py                     # Core NMF decomposition (PyTorch GPU/CPU) + preprocessing
├── cNMF.py                    # Consensus NMF wrapper using the external cnmf package
├── group_associations.py      # Statistical module-group quality metrics (silhouette, ANOVA)
├── spearman.py                # Pairwise Spearman correlations between modules
└── save_heatmap_pdf_ordered.py # PDF heatmap generator for cNMF output
```

---

## File 1: `nmf_utils.py` — The Orchestrator

This is the main entry point. It's called by the API routers and coordinates S3 I/O, temp file management, and calls to the NMF/cNMF functions.

### Key Functions

---

#### `preprocess_util(metadata, gene_column, metadata_index, design_factor, symbols, hvg, batch, batch_column, batch_include, job_id, single_cell)` *(async)*
**Called by:** `POST /preprocess_df`

1. Downloads raw counts from S3: `jobs/{job_id}/counts.csv`
2. Calls `NMF.preprocess2()` which:
   - Maps gene IDs (Ensembl → Symbol if needed)
   - Runs an R script (`filter_batch.R`) for filtering + optional batch correction
   - Selects `hvg` highly variable genes
3. Uploads preprocessed counts back to S3: `jobs/{job_id}/preprocessed_counts.csv`
4. **Returns:** Feather-encoded 15×15 preview of the preprocessed matrix

---

#### `run_regular_nmf_util(job_id, k, max_iter, design_factor)` *(async)*
**Called by:** `POST /run_regular_nmf`

1. Downloads preprocessed counts from S3
2. Calls `NMF.do_NMF(df, k=k, max_iter=max_iter)`
3. **Returns:** ZIP containing:
   - `nmf_h.feather` — gene loadings (H matrix, k × genes)
   - `nmf_w.feather` — module usages (W matrix, samples × k)
   - `metadata.json`

---

#### `run_cnmf_utils(metadata, k, hvg, max_iter, design_factor, metadata_index, job_id, batch_correct, gene_column)` *(async)*
**Called by:** `POST /run_nmf_files`

1. Downloads preprocessed counts from S3
2. Calls `cNMF.cNMF_consensus()` with the provided parameters
3. **Returns:** ZIP with:
   - PDF heatmap
   - Module usages TSV
   - Gene spectra scores TSV

---

#### `single_cell_util(...)` *(async)*
**Called by:** `POST /cnmf_single_cell`

Same as `run_cnmf_utils` but sets `single_cell=True`, which enables **Harmony batch correction** before NMF.

---

#### `explore_k_utils(meta, job_id, k_min, k_max, max_iter, design_factor, sample_column, n_reps)` *(async)*
**Called by:** `POST /k_metrics`

Runs a grid search across k values to help the user pick the best k:
1. Downloads preprocessed counts once from S3
2. For each `k` in `[k_min, k_max]` and each replicate:
   - Calls `process_single_run()` which runs NMF × 10 seeds
   - Computes a **silhouette score** for each run
3. Uses `ProcessPoolExecutor(max_workers=8)` for parallelism
4. **Returns:** JSON with all `(k, replicate, silhouette_score)` tuples

---

#### `process_single_run(expr_path, meta_path, k, run, max_iter, design_factor, sample_column, tmp_dir)`
- Runs 10 NMF replicates at a given k (different random seeds)
- Calls `group_associations.run_module_group_analysis()` to get silhouette score per run
- Returns a list of `{k, rep, silhouette_score}` dicts

---

## File 2: `NMF.py` — Core Decomposition

### `do_NMF(df_expr, design_factor, max_iter, n_components, seed)` → `(df_h, df_w, converged)`

The actual NMF factorization, implemented in **PyTorch** for GPU acceleration.

**Algorithm:**
1. Drops the design_factor column if present
2. Initializes W and H matrices using **NNDSVDA** (non-negative double SVD) from scikit-learn
3. Runs multiplicative update rule in a loop until convergence or `max_iter`
4. Convergence criterion: Frobenius norm change < 1e-4
5. Auto-detects CUDA; falls back to CPU

**Returns:**
- `df_h` — H matrix: modules × genes (named `Module_1`, `Module_2`, ...)
- `df_w` — W matrix: samples × modules
- `converged` — boolean

---

### `preprocess2(counts_link, metadata_link, metadata_index, design_factor, hvg, gene_column, symbols, batch, batch_column, batch_include)` → `(counts_df, tmp_dir)`

Full preprocessing pipeline:
1. Calls `transform_ids()` to map Ensembl IDs → gene symbols (using a local `gene_maps.csv`)
2. Runs `filter_batch.R` via subprocess which does:
   - `edgeR::filterByExpr()` gene filtering
   - TMM normalization + log-CPM transformation
   - Optional `limma::removeBatchEffect()` batch correction
   - HVG selection (top `hvg` most variable genes)
3. Returns the processed DataFrame ready for NMF

---

### `transform_ids(df_link, gene_column, symbols)` → `(df, new_path)`
- Reads a local `gene_maps.csv` lookup table (Ensembl ID → Symbol)
- Maps gene IDs in the expression matrix
- Reverts any duplicated symbol mappings back to Ensembl IDs to avoid collisions

---

## File 3: `cNMF.py` — Consensus NMF

### `cNMF_consensus(df, metadata, metadata_index, k, hvg, max_iter, design_factor, out_dir, batch_vars, single_cell)`

Wrapper around the external `cnmf` Python package (Kotliar et al. method).

**Two paths based on `single_cell`:**

**If `single_cell=False` (bulk RNA-seq):**
1. Converts DataFrame to AnnData via `convert_ann_obj()`
2. Runs cNMF factorization with 16 parallel workers (`ProcessPoolExecutor`)
3. Consensus step finalizes stable modules across random seeds

**If `single_cell=True`:**
1. Preprocesses with `scanpy` + Harmony batch correction
2. Then runs same cNMF workflow on the corrected counts

**Outputs written to disk (in `out_dir`):**
- Module usages (W matrix)
- Gene spectra scores (H matrix)
- Consensus cluster assignments
- PDF heatmap

---

### `convert_ann_obj(df, metadata, metadata_index, batch_vars)` → `AnnData`
- Converts a pandas DataFrame + metadata into an `AnnData` object
- Uses a sparse CSR matrix for memory efficiency
- Metadata columns attached to `obs` (observation/sample annotations)

---

## File 4: `group_associations.py` — Quality Metrics

### `ModuleGroupAnalyzer` class

Used during k-selection to evaluate how well a given k separates the experimental groups.

**Constructor:** `__init__(usage_matrix, sample_metadata)`
- Validates inputs and aligns samples between the matrix and metadata

**Key methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `calculate_silhouette_score()` | float (-1 to 1) | How well do modules separate groups? Higher = better |
| `calculate_module_anova(alpha)` | DataFrame | Per-module F-statistic, p-value, η² effect size |
| `perform_permanova(n_permutations)` | dict | Multivariate group test (R², p-value) — requires `skbio` |
| `calculate_group_statistics()` | DataFrame | Mean/std/median per module per group |

---

### `run_module_group_analysis(usage_df, metadata, output_dir, design_factor, sample_column)` → `float`
- Top-level function called by `nmf_utils.process_single_run()`
- Creates `ModuleGroupAnalyzer`, runs silhouette score
- **Returns:** silhouette score for that NMF run

---

## File 5: `spearman.py` — Module Correlations

### `run_spearman(df_H)` → `dict`
**Called by:** `POST /explore_correlations`

- Takes the H matrix (modules × genes)
- Computes pairwise Spearman correlation between all module gene rankings
- Returns upper-triangle pairs only (no redundancy)

### `spearman_pairs_from_H(df_h, use_abs)` → `list[dict]`
Each dict contains: `module_i`, `module_j`, `rho`, `abs_rho`, `is_max_for_k`

---

## File 6: `save_heatmap_pdf_ordered.py`

### `save_heatmap_pdf_ordered(usages_path, metadata, index, out_pdf, ...)`
- Renders a wide vector PDF heatmap of cNMF module usages
- Samples ordered by metadata index, all sample labels forced visible
- Used for publication-quality output from consensus NMF

---

## Data Flow

```
POST /preprocess_df
    → nmf_utils.preprocess_util()
        → NMF.preprocess2() → filter_batch.R (R subprocess)
        → Upload preprocessed_counts.csv to S3
        → Return feather preview

POST /k_metrics
    → nmf_utils.explore_k_utils()
        → ProcessPoolExecutor (8 workers)
            → nmf_utils.process_single_run() × (k × n_reps)
                → NMF.do_NMF() × 10 seeds
                → group_associations.run_module_group_analysis()
        → Return JSON {k, rep, silhouette}

POST /run_regular_nmf
    → nmf_utils.run_regular_nmf_util()
        → Download preprocessed counts from S3
        → NMF.do_NMF()
        → Return ZIP (H matrix + W matrix)

POST /run_nmf_files
    → nmf_utils.run_cnmf_utils()
        → Download preprocessed counts from S3
        → cNMF.cNMF_consensus() × 16 workers
        → Return ZIP (usages + gene spectra + PDF)

POST /cnmf_single_cell
    → nmf_utils.single_cell_util()
        → Same as above but single_cell=True (Harmony batch correction)

POST /explore_correlations
    → spearman.run_spearman()
        → Return JSON (Spearman pairs)
```

---

## Key Design Notes

- **GPU support:** `do_NMF()` auto-detects CUDA and falls back to CPU
- **BLAS threading:** Set to 1 thread per process to prevent over-subscription when using multiprocessing
- **Temp file cleanup:** `BackgroundTask(cleanup_after_send, dirs)` deletes temp dirs after HTTP response is sent
- **R subprocess:** Preprocessing uses an R script — requires a working R environment with `edgeR` and `limma` installed
- **S3 job tracking:** Every analysis is tied to a `job_id` (UUID); all intermediate files are stored under `jobs/{job_id}/` in S3
