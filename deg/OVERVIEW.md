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

## Files at a Glance

```
deg/
├── experiment_browser.py                    # Entry point: pick a curated experiment
├── group_selection.py                       # Main orchestrator for de novo analysis
├── group_helpers.py                         # Shared helpers: auth, group management, AI filters
├── group_selection_data_flow.py             # Steps 1–3: filters, metadata load, download
├── experiment_extract_ui.py                 # Download UI scoped to one experiment
├── precomputed_results_browser.py           # Browse pre-saved DEG results
├── precomputed_results_browser_helpers.py   # Sidebar + volcano plot for precomputed results
├── viz_helpers.py                           # All result visualizations + AI research pipeline
└── plot_helpers.py                          # Label placement algorithm for scatter plots
```

---

## The De Novo Analysis Workflow (Step by Step)

### Step 1 — Filter Samples (Optional)
The user can narrow down which samples they want to see, two ways:
- **Manual**: Use dropdown menus to select values for each metadata column (Genotype, Treatment, Timepoint, etc.)
- **AI-assisted**: Type in plain English, e.g. *"WT or TDP43 iMNs treated with Rotenone at 24 or 48 hours"* — Claude converts this to a structured filter automatically

### Step 2 — Load Metadata
The app sends the filters to the backend (`POST /get_samples`), which returns a table of matching samples. Each row is a sample, each column is a metadata attribute.

### Step 3 — Assign Samples to Groups
The user checks boxes in the table to assign samples to:
- **Comparison Group (A)** — the experimental condition
- **Reference Group (B)** — the control

Again, this can be done manually or with AI: *"Add TDP43 samples to Group A and WT samples to Group B"*

### Step 4 — Run DEG Analysis
Click "Run DEG Analysis". The app sends the two sample lists to the backend (`POST /deg_submit_groups/`), which:
1. Fetches raw counts from MongoDB
2. Runs an R script using `limma`/`edgeR`
3. Returns a ZIP file with the DEG table (genes, p-values, log fold changes), an optional heatmap, and optional GSEA results

### Step 5 — Explore Results
The results page offers multiple visualizations:

| Visualization | What it shows |
|--------------|---------------|
| **DEG Table** | Full list of genes with statistics |
| **Volcano Plot** | logFC vs. significance; up/down genes visible at a glance |
| **MA Plot** | Average expression vs. logFC |
| **Heatmap** | Top genes across all samples, color-coded by expression |
| **GSEA Barplot** | Enriched hallmark gene sets (top 20) |
| **AI Research Pipeline** | Disease associations, drug predictions, novel hypotheses |

---

## The AI Research Pipeline

After viewing DEG results, the user can optionally run the **AI Research Pipeline**:
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

If a comparison was already analyzed and saved:
1. **Select experiment** → **Select group** → **Select contrast term** (e.g., "TDP43 vs WT")
2. The app fetches the saved CSV + optional GSEA plot from S3
3. Same visualization tools are available (volcano plot, table, etc.)

The sidebar dynamically groups contrast terms by variable (Genotype, Treatment, etc.) and separates main effects from interaction terms.

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
| `POST /deg_with_research/` | DEG + AI research pipeline |
| `POST /deg_results/groups` | List precomputed groups for an experiment |
| `POST /deg_results/terms` | List available contrast terms |
| `POST /deg_results/fetch_csv` | Download precomputed DEG CSV from S3 |

---

## How to Navigate the Code

| If you want to understand... | Start here |
|-----------------------------|-----------|
| The overall flow | `group_selection.py` → `run_group_selection()` |
| How filters work (manual + AI) | `group_selection_data_flow.py` |
| How group assignment works | `group_helpers.py` → `add_to_group()`, `natural_language_group_assignment()` |
| How results are visualized | `viz_helpers.py` → `render_deg_results_and_visualizations()` |
| How precomputed results work | `precomputed_results_browser.py` |
| How the experiment browser works | `experiment_browser.py` |

---

## Important Notes

- The backend URL is hardcoded to `http://18.218.84.81:8000/`
- Authentication uses a password stored in Streamlit secrets
- The EC2 server must be "woken up" via a Lambda function when cold (takes 1–4 min)
- Session state keys are prefixed (`deg_novo_` or `deg_exp_`) to avoid collisions between the different flows
