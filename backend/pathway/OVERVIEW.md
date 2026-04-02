# backend/pathway/ Overview

## What Is This?

The `pathway/` folder handles two distinct but related tasks:

1. **Gene descriptions**: Take the top genes from each NMF module and ask GPT to write a brief biological description of each gene
2. **KEGG pathway analysis**: Given a ranked gene list (with z-scores), run GSEA against KEGG pathways and generate pathway diagrams colored by expression

---

## Files

```
pathway/
├── gene_loadings_gpt.py    # GPT-based gene annotation (async, parallel)
└── pathway_analysis.R      # KEGG pathway enrichment + pathview visualization (R script)
```

---

## File 1: `gene_loadings_gpt.py`

### Purpose
Enriches NMF gene loading results with plain-English gene descriptions from GPT. Runs requests in parallel to handle many genes quickly.

### Called by
`POST /process_gene_loadings/` in `api/routers/pathway.py`

---

### `gpt_utils(file: UploadFile, top_n: int) → JSONResponse` *(async)*

**Parameters:**
- `file` — Feather file containing the gene loadings matrix (modules × genes or genes × modules)
- `top_n` — Number of top-ranked genes to describe per module (default: 2)

**What it does:**
1. Reads the Feather file into a DataFrame
2. Transposes if needed so genes are rows and modules are columns
3. For each module, selects the `top_n` genes with the highest loading value
4. Breaks all genes across all modules into batches of 20
5. Sends each batch to `infra.llm_request_functions.model_request()` in parallel using `ThreadPoolExecutor` (16 workers)
6. GPT returns `{gene_name: description}` per batch
7. Assembles the descriptions back per module

**Returns:** JSON structured as:
```json
{
  "Module_1": [
    {"Gene": "TARDBP", "value": 0.85, "Description": "RNA-binding protein linked to ALS..."},
    ...
  ],
  "Module_2": [...],
  ...
}
```

---

### `break_chunks(genes, size)` → generator
- Simple utility: splits a list into sub-lists of `size` items
- Used to create batches of genes for parallel GPT requests

---

## File 2: `pathway_analysis.R`

### Purpose
GSEA-based pathway enrichment using KEGG gene sets, followed by per-pathway visualizations using the `pathview` package. Invoked as a subprocess from `api/routers/pathway.py`.

### Called by
`POST /run_pathview/` — the router writes the uploaded CSV to `/tmp`, then runs:
```bash
Rscript pathway_analysis.R <input_file> <out_root> [gene_format]
```

### Input File Format
A CSV with two columns:
- Column 1: Gene identifiers (Ensembl IDs like `ENSG...` or gene symbols like `TP53`)
- Column 2: Numeric values (typically z-scores or log fold changes)

### What it does, step by step

**Step 1 — Gene ID Mapping**
- Auto-detects identifier type: Ensembl IDs start with `"ENSG"`, everything else is treated as a symbol
- Uses `clusterProfiler::bitr()` to map both types to **Entrez IDs** (required for KEGG)
- Filters to unique Entrez IDs; logs the mapping success rate

**Step 2 — GSEA on KEGG Pathways** (`gseKEGG()`)
- Creates a named numeric vector: values = z-scores, names = Entrez IDs (sorted descending)
- Runs GSEA against the KEGG human database (`organism = "hsa"`)
- Parameters: 10,000 permutations, gene set size 3–800, p-value cutoff 0.05
- Saves results to `kegg_dataframe.csv`

**Step 3 — Pathway Visualization** (`pathview()`)
- For each significant pathway, generates a PNG with gene nodes colored by z-score:
  - Blue → strongly negative (downregulated)
  - White → neutral
  - Red → strongly positive (upregulated)
- Color scale is symmetric, capped at the 99th percentile of absolute values
- KEGG pathway XML/images are cached at `/home/ec2-user/kegg_cache` to speed up repeated requests

**Step 4 — Bundle Output**
- All pathway PNG images are zipped into `pathway_results.zip`
- PNG files are deleted after zipping
- Final outputs in `out_root/`:
  - `pathway_results.zip` — one PNG per significant pathway
  - `kegg_dataframe.csv` — GSEA statistics for each pathway

### Error Handling
- Custom error handler logs the full R traceback and exits with code 1
- Script stops early if no genes map to Entrez IDs or no pathways reach significance

### R Package Dependencies
| Package | Purpose |
|---------|---------|
| `clusterProfiler` | Gene ID mapping + GSEA |
| `org.Hs.eg.db` | Human gene annotations |
| `pathview` | KEGG pathway diagram rendering |
| `tidyverse` | Data manipulation |
| `zip` | ZIP archive creation |

---

## How These Connect to the API

```
POST /process_gene_loadings/
    │
    ├─ Receives feather upload (gene loadings from NMF)
    ├─ gene_loadings_gpt.gpt_utils(file, top_n)
    │    ├─ ThreadPoolExecutor × 16 workers
    │    └─ infra.llm_request_functions.model_request() per batch
    └─ Returns JSON {module: [{Gene, value, Description}]}

POST /run_pathview/
    │
    ├─ Receives CSV upload (genes + z-scores)
    ├─ Writes file to /tmp
    ├─ subprocess: Rscript pathway_analysis.R /tmp/file out_root
    │    ├─ Gene ID mapping (bitr)
    │    ├─ gseKEGG (GSEA)
    │    └─ pathview (per-pathway PNGs)
    └─ Returns ZIP (pathway_results.zip + kegg_dataframe.csv)
```

---

## Important Notes

- **Gene descriptions** require a valid `OPENAI_API_KEY` in the environment
- **Pathway analysis** requires R with `clusterProfiler`, `pathview`, and `org.Hs.eg.db` installed
- The KEGG cache at `/home/ec2-user/kegg_cache` is specific to the EC2 instance — this path would need to change in a different deployment
- KEGG analysis is human-only (`organism = "hsa"`); the `gene_format` parameter exists but is not currently used in the R script to change organism
