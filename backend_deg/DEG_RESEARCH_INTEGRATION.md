# DEG Analysis + Research Insights Integration

## Overview

This document describes the integrated pipeline that combines differential expression gene (DEG) analysis with AI-powered research insights. All computation happens on EC2, with the frontend receiving comprehensive results in a single API call.

## Architecture

```
User submits DEG CSV + Context
    ↓
[POST /deg_with_research/]
    ├─ Upload DEG table (CSV file)
    ├─ Parse DEG results (deg_parser.py)
    │   └─ Validates & maps columns
    ├─ Retrieve biological context (literature_retriever.py)
    │   └─ Gene knowledge for top genes
    ├─ Generate LLM predictions (llm_predictor.py)
    │   ├─ Disease associations
    │   ├─ Drug response
    │   └─ Research hypotheses
    └─ Return JSON with all results
```

**Note:** DEG analysis itself is done separately using `/deg_submit_groups/` or external tools. This endpoint focuses on research insights from already-computed DEG results.

## New Files Added

### Core Modules

1. **`deg_parser.py`** - DEG data parsing
   - `DEGResult`: Single gene result class
   - `DEGAnalysis`: Container for multiple results
   - Handles multiple column name formats

2. **`literature_retriever.py`** - Biological context
   - `LiteratureRetriever`: Retrieves gene knowledge
   - `GeneKnowledge`: Gene information container
   - Mock data for ~8 common genes (expandable)

3. **`llm_predictor.py`** - LLM integration
   - `BiomedicalPredictor`: Claude API wrapper
   - Methods for disease, drug, and hypothesis prediction

4. **`research_pipeline.py`** - Orchestration
   - `ResearchPipeline`: Main orchestrator
   - Coordinates DEG analysis → parsing → prediction
   - Runs R script and LLM calls

## New API Endpoint

### POST `/deg_with_research/`

Generate research insights from pre-computed DEG results.

**Request:**
- **File Upload**: `deg_table` - CSV file with DEG results
  - Required columns: `gene`, `log2fc`, `padj`
  - Other columns (pvalue, SYMBOL, etc.) are optional
- **Form Fields**:
  - `disease_context` (optional, default: "Unknown") - Disease/condition being studied
  - `tissue` (optional, default: "Unknown") - Tissue type
  - `num_genes` (optional, default: 10) - Number of top genes to analyze

**Example CSV Format:**
```
gene,log2fc,padj,pvalue,SYMBOL
ENSG00000141510,2.5,0.001,0.00001,TP53
ENSG00000012048,1.8,0.01,0.0001,BRCA1
ENSG00000146648,-2.2,0.005,0.00005,EGFR
```

**Note:** The endpoint accepts DEG tables that have already been computed via `/deg_submit_groups/` or any other DEG analysis tool. It skips the DEG analysis step and goes directly to generating research insights.

**Response:**
```json
{
  "status": "success",
  "deg_summary": "DEG Analysis Summary: ...",
  "total_genes": 15000,
  "disease_context": "Alzheimer's disease",
  "tissue": "Hippocampus",
  "deg_data": {
    "disease_context": "Alzheimer's disease",
    "tissue": "Hippocampus",
    "num_genes": 15000,
    "genes": [
      {
        "gene": "TP53",
        "log2fc": 2.5,
        "padj": 0.001,
        "pvalue": 0.00001,
        "direction": "upregulated"
      }
    ]
  },
  "biological_context": "BIOLOGICAL CONTEXT:\n\nGene: TP53\n  Functions: ...",
  "disease_predictions": {
    "predictions": "Disease Association Analysis:\n...",
    "model": "claude-sonnet-4-6",
    "tokens_used": 1234
  },
  "drug_predictions": {
    "predictions": "Drug Response Analysis:\n...",
    "model": "claude-sonnet-4-6",
    "tokens_used": 1456
  },
  "research_hypotheses": {
    "hypotheses": "Novel Research Hypotheses:\n...",
    "model": "claude-sonnet-4-6",
    "tokens_used": 1200
  }
}
```

## Setup Instructions

### 1. Dependencies

The integration requires the Anthropic API client. Add to requirements.txt if not present:

```
anthropic>=0.24.0
pandas>=2.0.0
```

Install with:
```bash
pip install anthropic
```

### 2. Environment Variables

Set the Anthropic API key on EC2:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

For persistent setup, add to `/etc/environment` or `.bashrc`:

```bash
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Verify Installation

Test the pipeline:

```bash
cd /Users/azamat/Documents/MGI/backend_deg

# Test DEG parser
python3 -c "from deg_parser import DEGAnalysis; print('✓ deg_parser works')"

# Test literature retriever
python3 -c "from literature_retriever import LiteratureRetriever; r = LiteratureRetriever(); print('✓ literature_retriever works')"

# Test LLM predictor (requires ANTHROPIC_API_KEY)
python3 -c "from llm_predictor import BiomedicalPredictor; p = BiomedicalPredictor(); print('✓ llm_predictor works')"

# Test research pipeline
python3 -c "from research_pipeline import ResearchPipeline; r = ResearchPipeline(); print('✓ research_pipeline works')"
```

## Workflow

### 1. DEG analysis (already completed)
- User has already computed DEG results using:
  - `/deg_submit_groups/` endpoint (run DEG on EC2)
  - External DEG tool (edgeR, limma, DESeq2, etc.)
  - Any other differential expression analysis
- Results available as CSV file with columns: gene, log2fc, padj

### 2. User uploads DEG table + context
- Uploads DEG results CSV file
- Provides disease context and tissue type
- Specifies number of top genes to analyze (default: 10)

### 3. Backend parses DEG results
- Reads CSV into DataFrame
- Validates required columns (gene, log2fc, padj)
- Handles case-insensitive column names
- Filters by adjusted p-value

### 4. Retrieve biological context
- Query knowledge base for each gene
- Get functions, diseases, drugs, pathways
- Format as LLM context

### 5. Generate predictions (via Claude API)
- **Disease associations**: What diseases might these genes affect?
- **Drug response**: What drugs might work for this profile?
- **Research hypotheses**: What novel ideas emerge from the data?

### 6. Return comprehensive results
- All results as JSON
- Includes DEG summary, biological context, and research insights
- User can explore results without additional API calls

## Example Usage

### cURL Request

```bash
# Upload DEG CSV file and get research insights
curl -X POST http://localhost:8000/deg_with_research/ \
  -F "deg_table=@deg_results.csv" \
  -F "disease_context=Alzheimer disease" \
  -F "tissue=Hippocampus" \
  -F "num_genes=10" \
  | python3 -m json.tool
```

### Python Client

```python
import requests
import json

url = "http://localhost:8000/deg_with_research/"

# Prepare DEG table file and form data
with open("deg_results.csv", "rb") as f:
    files = {"deg_table": f}
    data = {
        "disease_context": "Alzheimer disease",
        "tissue": "Hippocampus",
        "num_genes": 10
    }

    response = requests.post(url, files=files, data=data)

results = response.json()

# Access results
print(results["deg_summary"])
print(results["disease_predictions"]["predictions"])
print(results["research_hypotheses"]["hypotheses"])
```

### TypeScript/JavaScript

```typescript
// Upload DEG CSV file and get research insights
const fileInput = document.getElementById('degFile') as HTMLInputElement;
const degFile = fileInput.files[0];

const formData = new FormData();
formData.append('deg_table', degFile);
formData.append('disease_context', 'Alzheimer disease');
formData.append('tissue', 'Hippocampus');
formData.append('num_genes', '10');

const response = await fetch('http://localhost:8000/deg_with_research/', {
  method: 'POST',
  body: formData
});

const results = await response.json();

// Access results
console.log(results.deg_summary);
console.log(results.disease_predictions.predictions);
console.log(results.research_hypotheses.hypotheses);
```

## Configuration

### Adjusting Gene Knowledge Base

To add more genes to the knowledge base:

```python
# In literature_retriever.py, add to _initialize_knowledge_base():

new_gene = GeneKnowledge("BRCA2")
new_gene.add_function("DNA repair, particularly homologous recombination")
new_gene.add_disease("breast cancer", "very strong")
new_gene.add_disease("ovarian cancer", "very strong")
new_gene.add_drug("Olaparib", "PARP inhibitor", "approved")
new_gene.add_pathway("Homologous recombination", "KEGG")
kb["BRCA2"] = new_gene
```

### Adjusting LLM Model

To use a different Claude model, modify in `llm_predictor.py`:

```python
self.model = "claude-opus-4-6"  # Change from sonnet to opus
```

### Number of Genes to Analyze

Default is 10 genes. Users can request different numbers via the API:

```json
{
  "num_genes": 20  // Analyze top 20 genes
}
```

## Error Handling

### Missing API Key

If ANTHROPIC_API_KEY is not set, the endpoint will still work but skip LLM predictions:

```json
{
  "status": "success",
  "note": "LLM predictions not available (no API key)",
  "deg_data": { ... },
  "biological_context": "..."
}
```

### Insufficient Samples

If groups have < 2 samples:

```json
{
  "status": "error",
  "detail": "DEG analysis requires at least 2 samples per group. Got: A=1, B=1"
}
```

### Missing Samples

If requested samples don't exist in database:

```json
{
  "status": "error",
  "detail": "Samples not found in counts: {'sample_x', 'sample_y'}"
}
```

## Performance

- **DEG Analysis**: 30-60 seconds depending on gene count
- **Literature Retrieval**: <1 second
- **LLM Predictions**: 10-30 seconds (depends on model and token count)
- **Total**: ~2-3 minutes per request

Recommendations:
- Run async on frontend (don't block UI)
- Implement progress tracking
- Cache results for repeated queries
- Use background jobs for batch analysis

## Extending the Pipeline

### Adding Real API Data

Replace mock data with real APIs:

```python
# literature_retriever.py
def get_gene_knowledge_from_opentargets(gene: str):
    # Query OpenTargets API
    # Parse response
    # Return GeneKnowledge object
    pass
```

### Custom Analysis

Add domain-specific predictions:

```python
# llm_predictor.py
def predict_clinical_trial_outcomes(self, genes, context):
    # Custom prompt for clinical predictions
    # Return predictions
    pass
```

### Multi-tissue Comparison

Extend pipeline for tissue-specific analysis:

```python
def compare_tissues(self, degs_by_tissue, disease_context):
    # Generate tissue-specific insights
    # Compare expression patterns
    pass
```

## Troubleshooting

### "DEG analysis failed"

- Check R script exists: `/Users/azamat/Documents/MGI/backend_deg/deg_analysis.R`
- Check R packages installed: `R -q -e "library(edgeR); library(limma)"`
- Check sample names match database

### "BiomedicalPredictor not available"

- Verify ANTHROPIC_API_KEY is set: `echo $ANTHROPIC_API_KEY`
- Check API key is valid (try manual API call)
- LLM predictions will be skipped, but DEG analysis still works

### "No count data found"

- Verify samples exist in MongoDB: `/get_metadata/` endpoint
- Check sample names match exactly (case-sensitive)
- Verify counts table is populated

## Files Summary

| File | Purpose | Lines |
|------|---------|-------|
| `deg_parser.py` | Parse DEG results | ~140 |
| `literature_retriever.py` | Gene knowledge base | ~280 |
| `llm_predictor.py` | LLM predictions | ~160 |
| `research_pipeline.py` | Orchestration | ~250 |
| `fast_api.py` | API endpoint | +60 (new endpoint) |

**Total Integration**: ~790 lines of Python code

## Testing

### Manual Test

```bash
# Test DEG analysis with research
curl -X POST http://localhost:8000/deg_with_research/ \
  -H "Content-Type: application/json" \
  -d '{
    "group_a": ["sample1", "sample2"],
    "group_b": ["sample3", "sample4"],
    "disease_context": "Test",
    "tissue": "Tissue",
    "num_genes": 5
  }'
```

### Unit Tests

```python
# test_research_pipeline.py
import pandas as pd
from research_pipeline import ResearchPipeline

# Test parsing
pipeline = ResearchPipeline()
test_df = pd.DataFrame({
    "gene": ["TP53", "BRCA1"],
    "log2fc": [2.5, 1.8],
    "padj": [0.001, 0.01],
    "pvalue": [0.0001, 0.001]
})

deg_analysis = pipeline.parse_deg_results(test_df)
assert len(deg_analysis.degs) == 2
assert deg_analysis.degs[0].gene == "TP53"
```

## Support

For issues or questions:
1. Check logs: `/var/log/fastapi.log` (or EC2 console)
2. Review R output: Check stderr from deg_analysis.R
3. Verify environment variables: `env | grep ANTHROPIC`
4. Test API key: Manual Anthropic API call

## License

Same as parent project.
