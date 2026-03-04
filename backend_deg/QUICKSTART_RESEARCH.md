# Quick Start: DEG Research Insights

## Overview

Generate AI-powered research insights from pre-computed DEG results in 3 steps:

1. Get DEG results (CSV file)
2. Upload to `/deg_with_research/` endpoint
3. Get disease, drug, and hypothesis predictions

## Step 1: Get DEG Results

### Option A: Use backend DEG analysis
```bash
# Get DEG results via /deg_submit_groups/
curl -X POST http://localhost:8000/deg_submit_groups/ \
  -H "Content-Type: application/json" \
  -d '{
    "group_a": ["sample1", "sample2"],
    "group_b": ["sample3", "sample4"]
  }' > deg_results.zip

# Extract the CSV
unzip deg_results.zip
# Contains: deg_analysis.feather (convert to CSV if needed)
```

### Option B: Use your own DEG analysis
- Run DEG analysis with any tool (edgeR, limma, DESeq2, etc.)
- Save results as CSV with columns: `gene`, `log2fc`, `padj`
- Example format:
```csv
gene,log2fc,padj
ENSG00000141510,2.5,0.001
ENSG00000012048,1.8,0.01
ENSG00000146648,-2.2,0.005
```

## Step 2: Upload DEG Table

### cURL
```bash
curl -X POST http://localhost:8000/deg_with_research/ \
  -F "deg_table=@deg_results.csv" \
  -F "disease_context=Alzheimer disease" \
  -F "tissue=Hippocampus" \
  -F "num_genes=10"
```

### Python
```python
import requests

with open("deg_results.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/deg_with_research/",
        files={"deg_table": f},
        data={
            "disease_context": "Alzheimer disease",
            "tissue": "Hippocampus",
            "num_genes": 10
        }
    )

results = response.json()
```

### JavaScript/React
```javascript
const formData = new FormData();
formData.append("deg_table", degFile); // from file input
formData.append("disease_context", "Alzheimer disease");
formData.append("tissue", "Hippocampus");
formData.append("num_genes", 10);

const response = await fetch(
  "http://localhost:8000/deg_with_research/",
  { method: "POST", body: formData }
);

const results = await response.json();
```

## Step 3: Access Results

```python
# Disease predictions
print(results["disease_predictions"]["predictions"])

# Drug response
print(results["drug_predictions"]["predictions"])

# Research hypotheses
print(results["research_hypotheses"]["hypotheses"])

# Biological context
print(results["biological_context"])

# DEG summary
print(results["deg_summary"])
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deg_table` | File | Required | CSV with DEG results (gene, log2fc, padj) |
| `disease_context` | String | "Unknown" | Disease/condition being studied |
| `tissue` | String | "Unknown" | Tissue type |
| `num_genes` | Integer | 10 | Top genes to analyze |

## Required Columns in CSV

- `gene` - Gene identifier (symbol or Ensembl ID)
- `log2fc` - Log2 fold change
- `padj` - Adjusted p-value

## Optional Columns

- `pvalue` - Raw p-value
- `SYMBOL` - Gene symbol
- `GENENAME` - Gene description
- Any other metadata

## Error Handling

### Missing columns
```json
{
  "status": "error",
  "detail": "Missing required column: 'log2fc'"
}
```

### No API key (LLM predictions skipped)
```json
{
  "status": "success",
  "note": "LLM predictions not available (no API key)",
  "deg_data": {...},
  "biological_context": "..."
}
```

## What You Get

```json
{
  "status": "success",
  "deg_summary": "DEG Analysis Summary: 15000 genes...",
  "total_genes": 15000,
  "disease_context": "Alzheimer disease",
  "tissue": "Hippocampus",

  "biological_context": "BIOLOGICAL CONTEXT:\n\nGene: TP53\n...",

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

## Workflow

```
DEG Results (CSV)
      ↓
   Upload
      ↓
Backend Processing:
  • Parse CSV
  • Get gene context
  • Query LLM
  • Generate insights
      ↓
Research Insights (JSON)
```

## Tips

1. **Column names are flexible**: `gene`, `Gene`, `SYMBOL` all work
2. **Case-insensitive**: Column names work regardless of case
3. **Large datasets**: Works with 1000s of genes
4. **API key optional**: Works without ANTHROPIC_API_KEY (but skips LLM predictions)
5. **Async recommended**: Use background jobs for large analyses

## Setup

1. Ensure ANTHROPIC_API_KEY is set:
```bash
export ANTHROPIC_API_KEY="sk-..."
```

2. Backend is running:
```bash
uvicorn fast_api:app --host 0.0.0.0 --port 8000
```

3. You have Python requests or similar HTTP client

## Examples

### Example 1: Minimal request
```bash
curl -X POST http://localhost:8000/deg_with_research/ \
  -F "deg_table=@deg.csv"
```

### Example 2: Full request with context
```bash
curl -X POST http://localhost:8000/deg_with_research/ \
  -F "deg_table=@deg.csv" \
  -F "disease_context=Type 2 Diabetes" \
  -F "tissue=Pancreatic islet" \
  -F "num_genes=15"
```

### Example 3: Python workflow
```python
import pandas as pd
import requests

# Load your DEG results
deg_df = pd.read_csv("my_deg_analysis.csv")

# Upload and get insights
with open("my_deg_analysis.csv", "rb") as f:
    response = requests.post(
        "http://ec2-instance:8000/deg_with_research/",
        files={"deg_table": f},
        data={
            "disease_context": "Cancer",
            "tissue": "Tumor",
            "num_genes": 20
        }
    )

insights = response.json()

# Save results
with open("research_insights.json", "w") as f:
    import json
    json.dump(insights, f, indent=2)

print("Hypotheses:")
print(insights["research_hypotheses"]["hypotheses"])
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Missing required column" | Check CSV has `gene`, `log2fc`, `padj` columns |
| "LLM predictions not available" | Set ANTHROPIC_API_KEY environment variable |
| File upload fails | Ensure file is readable CSV, < 100MB |
| Timeout | Large DEG tables take longer; wait or reduce `num_genes` |

## Support

See `DEG_RESEARCH_INTEGRATION.md` for detailed documentation.
