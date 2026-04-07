# DEG Module Overview

## What Is This?

The `deg` folder contains Streamlit UI pages for **Differential Expression Analysis** (DEG). DEG analysis compares gene expression between two groups of samples to find which genes are significantly up- or down-regulated.

For example: "Which genes are expressed differently in TDP43 motor neurons vs. wild-type motor neurons?"

---

## Three Ways to Use This Module

### Path 1 — Experiment Browser
Start from a curated list of published experiments. Select one, see its samples, and then either run a fresh analysis or browse pre-computed results.

### Path 2 — De Novo Analysis (Free-form)
Filter any samples from the database, assign them to two groups manually (or with AI assistance), and run a new DEG analysis.

### Path 3 — Precomputed Results Browser
Browse previously analyzed comparisons that were saved to MongoDB + S3. No re-computation needed.

---

## Folder Structure

The module is organized into three subpackages plus shared canonical files at the root level:

```
deg/
├── browse_brb_by_experiment/           # Entry points for the experiment-first workflow
│   ├── __init__.py
│   ├── experiment_browser.py           # Curated experiment table + routing to sub-flows
│   ├── precomputed_results_browser.py  # Groups → results navigation for precomputed DEG
│   └── precomputed_results_browser_helpers.py  # Sidebar, volcano plot, expression heatmap
│
├── browse_brb_by_metadata/             # Entry points for the free-form metadata workflow
│   ├── __init__.py
│   ├── group_selection.py              # Main orchestrator for de novo analysis
│   ├── experiment_extract_ui.py        # Re-export shim → deg.experiment_extract_ui
│   └── group_selection_data_flow.py    # Re-export shim → deg.group_selection_data_flow
│
├── viz_helpers/                        # All result visualization components
│   ├── __init__.py
│   ├── common.py                       # Shared constants (MIN_LFC, MAX_FDR) + utilities
│   ├── volcano_plot.py                 # Volcano plot expander
│   ├── ma_plot.py                      # MA plot expander
│   ├── gsea_plot.py                    # GSEA barplot expander
│   ├── heatmap.py                      # Expression heatmap expander (calls backend)
│   ├── deg_viz.py                      # Main orchestrator: calls all 5 viz sub-functions
│   └── research_pipeline.py            # AI research pipeline UI
│
├── group_helpers.py                    # Shared helpers: auth, EC2 wake, group management, AI filters
├── group_selection_data_flow.py        # Canonical Steps 1–3: filters, metadata load, download
├── experiment_extract_ui.py            # Download UI scoped to one experiment
├── plot_helpers.py                     # Label placement algorithm for scatter plots
└── viz_helpers.py                      # Legacy shim → delegates to viz_helpers/ subpackage
```

> **Note on subpackage shims:** `browse_brb_by_metadata/experiment_extract_ui.py` and `browse_brb_by_metadata/group_selection_data_flow.py` are thin re-export shims that point to the canonical implementations at the root `deg/` level. This keeps the subpackage import paths clean.

---

## The De Novo Analysis Workflow (Step by Step)

### Step 1 — Filter Samples (Optional)
`group_selection_data_flow.py` → `render_filter_and_data_steps()`

The user can narrow down which samples they want to see, two ways:
- **Manual**: Use dropdown menus to select values for each metadata column (Genotype, Treatment, Timepoint, etc.)
- **AI-assisted**: Type in plain English, e.g. *"WT or TDP43 iMNs treated with Rotenone at 24 or 48 hours"* — Claude converts this to a structured filter automatically via `group_helpers.natural_language_to_filters()`

### Step 2 — Load Metadata
The app sends the filters to the backend (`POST /get_samples`), which returns a table of matching samples as a Feather file. Each row is a sample, each column is a metadata attribute.

### Step 3 — Assign Samples to Groups
The user checks boxes in the table to assign samples to:
- **Comparison Group (A)** — the experimental condition
- **Reference Group (B)** — the control

Again, this can be done manually or with AI: *"Add TDP43 samples to Group A and WT samples to Group B"*

### Step 4 — Run DEG Analysis
Click "Run DEG Analysis". The app sends the two sample lists to the backend (`POST /deg_submit_groups/`), which:
1. Fetches raw counts from S3 (parquet shards)
2. Runs an R script using `limma`/`edgeR`
3. Returns a ZIP file containing:
   - `deg_analysis.feather` — main DEG results table
   - `heatmap_matrix.csv` + `heatmap_annotation.csv` — top-30 variable genes (optional)
   - `gsea_results.csv` — hallmark pathway enrichment (optional)
   - `deg_heatmap.png` — pre-rendered heatmap with MongoDB metadata annotations (optional)

### Step 5 — Explore Results
`viz_helpers/deg_viz.py` → `render_deg_results_and_visualizations()`

Results visualizations are split across five sub-modules:

| Module | Visualization | What it shows |
|--------|--------------|---------------|
| `viz_helpers/volcano_plot.py` | **Volcano Plot** | logFC vs. significance; up/down genes visible at a glance |
| `viz_helpers/ma_plot.py` | **MA Plot** | Average expression vs. logFC |
| `viz_helpers/gsea_plot.py` | **GSEA Barplot** | Enriched hallmark gene sets (top 20), interactive Plotly chart |
| `viz_helpers/heatmap.py` | **Expression Heatmap** | Top genes across samples; calls `POST /deg_heatmap_render/` with annotation column selection |
| `viz_helpers/research_pipeline.py` | **AI Research Pipeline** | Disease associations, drug predictions, novel hypotheses |

---

## Expression Heatmap (New)

The heatmap is no longer rendered by the R script alone. It uses a dedicated two-step flow:

1. **R script** writes `heatmap_matrix.csv` (top-30 most variable genes, log₂-CPM, hierarchically clustered rows)
2. **`viz_helpers/heatmap.py`** sends `heatmap_matrix.csv` + group assignments to `POST /deg_heatmap_render/`
3. **Backend** (`backend/deg/deg_heatmap.py`) merges GroupA/GroupB labels with MongoDB metadata, applies user-selected annotation columns, and returns a publication-quality PNG

For **precomputed results**, a similar flow uses `POST /deg_results/precomputed_heatmap` (no group labels needed — only MongoDB metadata is used for annotations).

---

## The AI Research Pipeline

After viewing DEG results, the user can optionally run the **AI Research Pipeline** (`viz_helpers/research_pipeline.py`):
1. Input disease context, tissue type, and comparison description
2. The app sends the DEG table to `POST /deg_with_research/`
3. The backend queries **OpenTargets** for disease-gene associations
4. **Claude LLM** synthesizes everything and generates:
   - A biological summary of the expression changes
   - Disease association predictions with confidence scores
   - Drug/therapeutic suggestions
   - Novel research hypotheses to validate

---

## Precomputed Results Browser

`browse_brb_by_experiment/precomputed_results_browser.py`

Two-level navigation:
1. **Groups view** (`render_groups_view`): Lists all comparison groups for the selected experiment. Each row is a unique `(group, context)` pair. Click to navigate deeper.
2. **Results view** (`render_results_view`): Loads the DEG table for the selected contrast term. Offers the same visualization suite as de novo results.

`browse_brb_by_experiment/precomputed_results_browser_helpers.py` provides:
- `render_precomputed_deg_term_sidebar()` — sidebar with contrast term buttons grouped by variable (main effects vs. interaction terms)
- `render_table_and_volcano()` — DEG table + volcano plot
- `render_precomputed_expression_heatmap()` — expression heatmap using MongoDB metadata only (calls `POST /deg_results/precomputed_heatmap`)

---

## Experiment Browser

`browse_brb_by_experiment/experiment_browser.py`

Shows a curated table of 8 experiments. On selection:
- Displays a metadata summary (sample counts, genotypes, treatments, timepoints)
- Offers three actions:
  - **Run de novo analysis** → hands off to `browse_brb_by_metadata/group_selection.py`
  - **Browse precomputed results** → enters `precomputed_results_browser.py`
  - **Download metadata + counts** → enters `experiment_extract_ui.py`

---

## AI-Assisted Features

This module uses **Claude Sonnet 4.6** for two natural language tasks:

**1. Natural Language Filtering**
```
User: "WT or TDP43 treated with Rotenone at 24h, not 96h"
Claude: {"Genotype": ["WT", "TDP43"], "Treatment": ["Rotenone"], "Timepoint": [24]}
```

**2. Natural Language Group Assignment**
```
User: "Compare TDP43 samples at 48h against WT controls"
Claude: {group_a: [TDP43 samples], group_b: [WT samples]}
```

---

## Key Backend Endpoints Used

| Endpoint | What it does |
|----------|-------------|
| `GET /get_metadata/` | Fetch available metadata columns and values |
| `POST /get_samples/` | Filter and retrieve matching samples |
| `POST /get_counts/` | Load the count matrix for selected samples |
| `POST /deg_submit_groups/` | Run DEG analysis on two groups |
| `POST /deg_heatmap_render/` | Render heatmap PNG with MongoDB annotations (de novo) |
| `POST /deg_with_research/` | DEG + AI research pipeline |
| `POST /deg_results/groups` | List precomputed groups for an experiment |
| `POST /deg_results/terms` | List available contrast terms |
| `POST /deg_results/fetch_csv` | Download precomputed DEG CSV + GSEA barplot from S3 |
| `POST /deg_results/precomputed_heatmap` | Render heatmap PNG from precomputed DEG results |

---

## How to Navigate the Code

| If you want to understand... | Start here |
|-----------------------------|-----------|
| The overall de novo flow | `browse_brb_by_metadata/group_selection.py` → `run_group_selection()` |
| How filters work (manual + AI) | `group_selection_data_flow.py` → `render_filter_and_data_steps()` |
| How group assignment works | `group_helpers.py` → `add_to_group()`, `natural_language_group_assignment()` |
| How results are visualized | `viz_helpers/deg_viz.py` → `render_deg_results_and_visualizations()` |
| How the heatmap is rendered | `viz_helpers/heatmap.py` → calls `POST /deg_heatmap_render/` |
| How precomputed results work | `browse_brb_by_experiment/precomputed_results_browser.py` |
| How the experiment browser works | `browse_brb_by_experiment/experiment_browser.py` |
| How the AI research pipeline works | `viz_helpers/research_pipeline.py` |

---

## Important Notes

- The backend URL is hardcoded to `http://18.218.84.81:8000/`
- Authentication uses a password stored in Streamlit secrets
- The EC2 server must be "woken up" via a Lambda function when cold (takes 1–4 min)
- Session state keys are prefixed (`deg_novo_` or `deg_exp_`) to avoid collisions between the different flows
- Constants `MIN_LFC = 0.6` and `MAX_FDR = 0.05` (defined in `viz_helpers/common.py`) control significance thresholds across all plots
