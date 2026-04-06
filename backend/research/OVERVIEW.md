# backend/research/ Overview

## What Is This?

The `research/` folder is the **AI-powered biological insights pipeline**. It takes DEG analysis results and generates:
- Disease association predictions
- Drug response and repurposing suggestions
- Novel research hypotheses

It does this by combining two data sources:
1. **OpenTargets API** — a free public database of disease-gene and drug-gene associations
2. **Claude LLM** — synthesizes the data into structured biological predictions

---

## Files

```
research/
├── research_utils.py         # FastAPI service layer (entry point from API router)
├── research_pipeline.py      # Master orchestrator for the full pipeline
├── deg_parser.py             # Parse DEG results CSVs into structured objects
├── literature_retriever.py   # Retrieve gene context from OpenTargets
├── opentargets_retriever.py  # Low-level GraphQL client for OpenTargets API
└── llm_predictor.py          # Claude LLM calls for predictions
```

---

## How the Pipeline Flows

```
API: POST /deg_with_research/
    │
    ▼
research_utils.run_deg_with_research_pipeline()
    │  (normalizes column names, validates CSV)
    ▼
research_pipeline.ResearchPipeline.full_analysis()
    │
    ├─ deg_parser.DEGAnalysis.from_dataframe()
    │    → Parses genes, log2FC, padj
    │    → Ranks top genes
    │
    ├─ literature_retriever.LiteratureRetriever.generate_context_for_llm()
    │    → For each top gene:
    │        opentargets_retriever.OpenTargetsAPI.get_diseases_for_gene()
    │        opentargets_retriever.OpenTargetsAPI.get_drugs_for_gene()
    │    → Formats everything as a context string
    │
    └─ llm_predictor.BiomedicalPredictor (3 calls to Claude)
         ├─ predict_disease_associations()
         ├─ predict_drug_response()
         └─ generate_research_hypothesis()
    │
    ▼
JSON response with all results
```

---

## File 1: `research_utils.py` — Service Entry Point

### `run_deg_with_research_pipeline(deg_bytes, disease_context, tissue, num_genes, comparison_description) → JSONResponse`

The adapter between the FastAPI router and the research pipeline.

**What it does:**
1. Parses the uploaded CSV bytes with pandas
2. Normalizes column names — supports many naming conventions:
   - Gene: `gene`, `Gene`, `SYMBOL`, `ENSEMBLID`
   - Log2FC: `log2fc`, `logFC`, `log2FoldChange`
   - Adjusted p-value: `padj`, `adj.P.Val`, `adjusted_pvalue`
3. Returns HTTP 400 if required columns are missing
4. Calls `ResearchPipeline().full_analysis()`
5. Returns the result as JSON

---

## File 2: `research_pipeline.py` — Orchestrator

### `ResearchPipeline` class

**Constructor:** Initializes `LiteratureRetriever` and `BiomedicalPredictor`. If `ANTHROPIC_API_KEY` is missing, logs a warning and skips LLM predictions gracefully.

---

### `full_analysis(deg_df, disease_context, tissue, num_genes, comparison_description) → dict`

The main method. Returns a dict with:

| Key | Description |
|-----|-------------|
| `status` | `"success"` or error status |
| `deg_summary` | Human-readable summary of DEG results |
| `total_genes` | Number of DEG genes |
| `deg_data` | Serialized DEGAnalysis object |
| `biological_context` | OpenTargets context string |
| `disease_predictions` | Claude's disease association predictions |
| `drug_predictions` | Claude's drug response predictions |
| `research_hypotheses` | Claude's research hypothesis suggestions |
| `note` or `prediction_error` | Status messages if LLM unavailable |

**Pipeline:**
1. `parse_deg_results()` → builds `DEGAnalysis` object
2. `get_biological_context()` → queries OpenTargets for top `num_genes` genes
3. `predict_disease_associations()` → Claude call #1
4. `predict_drug_response()` → Claude call #2
5. `generate_hypotheses()` → Claude call #3
6. Assembles and returns all results

---

## File 3: `deg_parser.py` — DEG Data Structures

### `DEGResult` class
Represents a single gene's DEG statistics.

**Fields:** `gene`, `log2fc`, `padj`, `pvalue`

**Property:** `direction` → `"upregulated"` if `log2fc > 0`, else `"downregulated"`

---

### `DEGAnalysis` class
Container for a full set of DEG results with filtering and ranking.

**Key methods:**

| Method | Description |
|--------|-------------|
| `add_deg(deg)` | Add a DEGResult |
| `filter_by_padj(threshold=0.05)` | Return genes below p-value threshold |
| `filter_by_lfc(threshold=1.0)` | Return genes with \|log2fc\| > threshold |
| `get_top_genes(n=10, by="padj")` | Return top N genes sorted by `"padj"`, `"log2fc"`, or `"combined"` |
| `from_csv(filepath)` | Load from CSV file |
| `from_dataframe(df)` | Load from pandas DataFrame |
| `summary()` | Human-readable text summary |

**Column name flexibility:** The `from_dataframe()` method tries multiple naming conventions per column (e.g., `logFC`, `log2FoldChange`, `log2fc` all map to log2fc).

---

## File 4: `literature_retriever.py` — Biological Context

### `GeneKnowledge` class
Data container for what is known about a single gene.

**Methods:** `add_function()`, `add_disease()`, `add_drug()`, `add_pathway()`

---

### `LiteratureRetriever` class

**`get_gene_knowledge(gene) → GeneKnowledge`**
- Accepts Ensembl IDs or HGNC symbols
- Queries OpenTargets for disease associations and drug information
- Caches results to avoid redundant API calls

**`batch_retrieve(genes) → dict`**
- Retrieves knowledge for a list of genes

**`generate_context_for_llm(genes) → str`**
- Main method used by the pipeline
- Retrieves knowledge for all genes, formats it as a structured text block
- This string is passed directly to Claude as biological context

---

## File 5: `opentargets_retriever.py` — OpenTargets API Client

Low-level GraphQL client for the OpenTargets Platform API.

**API endpoint:** `https://api.platform.opentargets.org/api/v4/graphql`
**Authentication:** None required (free public API)

### `OpenTargetsAPI` class

**`get_diseases_for_gene(gene_symbol, max_results=100) → list`**
- Resolves gene symbol → Ensembl ID (cached)
- Queries `associatedDiseases` via GraphQL
- **Returns:** list of disease association dicts with names and scores

**`get_drugs_for_gene(gene_symbol, max_results=50) → list`**
- Queries `knownDrugs` via GraphQL
- **Returns:** list of drug dicts with mechanism, status, indication

**`parse_disease_response(diseases_data) → list`**
- Maps association scores to evidence strength labels:

| Score | Strength |
|-------|----------|
| ≥ 0.7 | very strong |
| ≥ 0.5 | strong |
| ≥ 0.3 | moderate |
| ≥ 0.1 | weak |
| < 0.1 | minimal |

**`parse_drug_response(drugs_data) → list`**
- Extracts drug name, mechanism of action, clinical status, indication

**Caching:**
- Responses cached to `opentargets_cache.json`
- Rate limiting: 0.3 second delay between requests

---

## File 6: `llm_predictor.py` — Claude LLM Calls

### `BiomedicalPredictor` class

**Model:** `claude-sonnet-4-6`
**API key:** from `ANTHROPIC_API_KEY` env var

---

### `predict_disease_associations(genes, biological_context, disease_context, tissue, comparison_description) → dict`
- Sends: gene list (with log2fc + padj), OpenTargets context, and study metadata
- Claude returns: 3–4 disease associations with confidence scores, therapeutic approaches, relevant pathways, and validation experiments
- **Returns:** `{"predictions": str, "model": str, "tokens_used": int}`

---

### `predict_drug_response(...) → dict`
- Same inputs as above
- Claude returns: 3–4 drugs (sensitivity/resistance), 2–3 repurposing opportunities, 2 biomarker genes, 1–2 key gene-drug interactions
- **Returns:** same structure

---

### `generate_research_hypothesis(...) → dict`
- Same inputs
- Claude returns: 2–3 novel, testable hypotheses with rationale and experimental validation suggestions
- **Returns:** `{"hypotheses": str, "model": str, "tokens_used": int}`

**Design philosophy:** All three prompts instruct Claude to be concise — fewer, higher-confidence results over exhaustive lists.

---

## Error Handling & Graceful Degradation

- If `ANTHROPIC_API_KEY` is missing → `BiomedicalPredictor` logs a warning; `full_analysis()` skips all three LLM calls and returns results with a `"note"` field
- If OpenTargets API fails for a gene → that gene is skipped silently
- R script timeout: 300 seconds
- All API calls include logging for debugging

---

## Environment Variables Required

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude LLM access |

No auth needed for OpenTargets (free API).
