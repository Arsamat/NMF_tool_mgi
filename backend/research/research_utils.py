import io
from typing import Dict, Any

import pandas as pd
from fastapi.responses import JSONResponse

from research.research_pipeline import ResearchPipeline


def run_deg_with_research_pipeline(
    deg_bytes: bytes,
    disease_context: str,
    tissue: str,
    num_genes: int,
    comparison_description: str,
) -> JSONResponse:
    """
    Shared service logic for the /deg_with_research/ endpoint.

    Takes raw CSV bytes for a DEG table, normalizes required columns,
    runs the research pipeline, and returns a JSONResponse.
    """
    deg_df = pd.read_csv(io.BytesIO(deg_bytes))

    # Define column aliases for each required field
    col_aliases: Dict[str, list[str]] = {
        "gene": ["gene", "Gene", "GENE", "SYMBOL", "ENSEMBLID", "ensemblid"],
        "log2fc": [
            "log2fc",
            "logFC",
            "log2FC",
            "Log2FC",
            "log2FoldChange",
            "logFoldChange",
        ],
        "padj": [
            "padj",
            "adj.P.Val",
            "adjustedPvalue",
            "adjusted_pvalue",
            "p.adjust",
        ],
    }

    available_cols = set(deg_df.columns)
    col_mapping: Dict[str, str] = {}

    # Find matching columns for each required field
    for req_col, aliases in col_aliases.items():
        found = False
        for alias in aliases:
            if alias in available_cols:
                col_mapping[alias] = req_col
                found = True
                break
        if not found:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "detail": (
                        f"Missing required column for '{req_col}'. "
                        f"Expected one of: {aliases}. "
                        f"Available columns: {list(available_cols)}"
                    ),
                },
            )

    # Rename columns to standard names if needed
    if col_mapping:
        deg_df = deg_df.rename(columns=col_mapping)

    # Initialize and run research pipeline
    pipeline = ResearchPipeline()
    results: Dict[str, Any] = pipeline.full_analysis(
        deg_df=deg_df,
        disease_context=disease_context,
        tissue=tissue,
        num_genes=num_genes,
        comparison_description=comparison_description,
    )

    return JSONResponse(
        status_code=200,
        content=results,
    )

