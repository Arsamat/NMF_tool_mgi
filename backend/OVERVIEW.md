# Backend Overview

## What Is This?

The `backend` folder is the **server-side engine** of the MGI platform. It's a Python API (built with FastAPI) that handles all the heavy computation — things like differential gene expression (DEG) analysis, NMF decomposition, heatmap generation, and AI-powered biological insights.

Think of it as the "brain" that the frontend pages talk to. When a user clicks a button on the Streamlit UI, a request is sent here to actually crunch the numbers.

---

## How It's Organized

```
backend/
├── fast_api.py              # The main app — starts the server, registers all routes
├── api/
│   └── routers/             # One file per feature area (DEG, NMF, heatmaps, etc.)
├── deg/                     # Logic for differential expression analysis
├── nmf/                     # Logic for NMF decomposition
├── heatmaps/                # Logic for generating heatmap images
├── pathway/                 # Logic for pathway/KEGG analysis + AI gene descriptions
├── research/                # AI research pipeline (Claude LLM + OpenTargets API)
├── infra/                   # Database (MongoDB) and cloud storage (S3) utilities
└── api/schemas/             # Data models that validate incoming requests
```

---

## The Main Entry Point: `fast_api.py`

This file starts the FastAPI application. It:
- Registers all the API "routers" (feature areas)
- Sets up CORS (so the frontend browser can talk to it)
- Adds a global error handler that uses Claude to generate user-friendly error messages

---

## Feature Areas (Routers)

### 1. Health Check (`/healthz`)
A simple ping endpoint. The frontend polls this to know when the server is ready.

---

### 2. DEG Analysis (`/deg_submit_groups/`, `/deg_with_research/`)
**Differential expression analysis** — compares gene expression between two sample groups.

**How it works:**
1. User provides two lists of sample names (Group A vs Group B)
2. The server fetches raw gene counts from MongoDB
3. An R script (`deg_analysis.R`) runs `limma`/`edgeR` statistics
4. Results (a table of genes with p-values and fold changes) come back as a ZIP file

There's also a `/deg_with_research/` endpoint that takes the DEG results further — it queries the OpenTargets disease database and then sends everything to Claude to generate biological hypotheses.

---

### 3. Precomputed DEG Results (`/deg_results/...`)
If a DEG analysis was already run and saved, these endpoints let you browse and download the results from MongoDB + S3 without re-running the analysis.

---

### 4. NMF (`/preprocess_df`, `/run_nmf_files`, `/cnmf_single_cell`, etc.)
**Non-negative Matrix Factorization** — decomposes a gene expression matrix into "modules" (groups of co-expressed genes).

**Pipeline:**
1. `/preprocess_df` — normalizes counts, selects highly variable genes, optionally corrects batch effects
2. `/k_metrics` — tests different numbers of modules (k) to help pick the best one
3. `/run_nmf_files` or `/cnmf_single_cell` — runs the actual NMF, returns matrices + a PDF heatmap

---

### 5. Heatmaps (`/plot_heatmap/`, `/annotated_heatmap/`, etc.)
Generates visualization images from expression data. Supports clustering, metadata annotations, and different orderings.

---

### 6. Pathway Analysis (`/process_gene_loadings/`, `/run_pathview/`)
- `/process_gene_loadings/`: Takes the top genes per NMF module and asks GPT to describe what they do biologically
- `/run_pathview/`: Runs an R script (`pathway_analysis.R`) to create KEGG pathway diagrams
- Currently not being present in the user interface, but may be made available later

---

### 7. Metadata & Samples (`/get_metadata/`, `/get_samples/`, `/get_counts/`)
Lets the frontend query the MongoDB database:
- What metadata columns exist? What values?
- Which samples match a given filter?
- Give me the count matrix for these samples

---

### 8. Uploads (`/create_upload_url`, `/complete_multipart_upload`, etc.)
Manages uploading large files to AWS S3. Uses presigned URLs so the browser can upload directly to S3 without routing through the server.

---

## Data Storage

| Where | What's stored |
|-------|--------------|
| **MongoDB** | Raw gene counts, sample metadata, precomputed DEG result paths |
| **AWS S3** | Preprocessed count matrices, NMF results, job output files |

---

## AI Integrations

| Service | Used for |
|---------|----------|
| **Claude (Anthropic)** | Biological disease predictions, research hypotheses, error message summaries |
| **OpenAI (GPT)** | Gene function descriptions per NMF module |
| **OpenTargets GraphQL API** | Fetching disease-gene associations (free, no auth needed) |

---

## How a Typical Request Flows

```
Browser (Streamlit UI)
    │
    │  HTTP POST with parameters
    ▼
FastAPI Router (e.g., /deg_submit_groups/)
    │
    │  Validate request schema
    ▼
Service function (e.g., run_deg_analysis())
    │
    ├── Fetch data from MongoDB
    ├── Run R script
    └── Bundle results into ZIP
    │
    ▼
Return StreamingResponse to browser
```

---

## Environment Variables Needed

```
MONGODB_URI          # MongoDB connection string
ANTHROPIC_API_KEY    # Claude API key
OPENAI_API_KEY       # OpenAI API key
AWS_ACCESS_KEY_ID    # AWS credentials
AWS_SECRET_ACCESS_KEY
```

---

## Where to Start Reading

If you want to understand the codebase, start here:
1. **`fast_api.py`** — see what routes exist
2. **`api/routers/deg.py`** — the most important feature endpoint
3. **`deg/deg_utils.py`** — how DEG analysis actually runs
4. **`infra/db_utils.py`** — how data is fetched from MongoDB
