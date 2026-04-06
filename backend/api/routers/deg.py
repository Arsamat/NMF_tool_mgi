import io
import json
import traceback

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from api.schemas.deg import DEGGroupsRequest
from deg.deg_heatmap import render_deg_heatmap_png_bytes
from deg.deg_utils import run_deg_analysis
from infra.db_utils import get_metadata_for_samples
from research.research_utils import run_deg_with_research_pipeline

router = APIRouter(tags=["deg"])


@router.post("/deg_submit_groups/")
async def deg_submit_groups(payload: DEGGroupsRequest):
    """
    Accept group_a and group_b sample lists, run DEG analysis (limma/voom),
    and return the DEG results table.
    """
    group_a = payload.group_a or []
    group_b = payload.group_b or []
    if not group_a or not group_b:
        return JSONResponse(
            status_code=400,
            content={"detail": "Both group_a and group_b must contain at least one sample."},
        )
    try:
        return run_deg_analysis(group_a, group_b)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "status": "error"},
        )


@router.post("/deg_with_research/")
async def deg_with_research(
    deg_table: UploadFile = File(...),
    disease_context: str = Form("Unknown"),
    tissue: str = Form("Unknown"),
    num_genes: int = Form(10),
    comparison_description: str = Form("Unknown"),
):
    """
    Generate research insights from pre-computed DEG results.
    """
    try:
        print("Starting research insights pipeline...")
        print(f"  Disease: {disease_context}")
        print(f"  Tissue: {tissue}")
        print(f"  Top genes to analyze: {num_genes}")

        deg_bytes = await deg_table.read()

        response = run_deg_with_research_pipeline(
            deg_bytes=deg_bytes,
            disease_context=disease_context,
            tissue=tissue,
            num_genes=num_genes,
            comparison_description=comparison_description,
        )

        print("Analysis complete.")
        return response
    except Exception as e:
        error_msg = str(e)
        print(f"Error in /deg_with_research: {error_msg}")
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": error_msg,
                "error_type": type(e).__name__,
            },
        )


@router.post("/deg_heatmap_render/")
async def deg_heatmap_render(
    heatmap_csv: UploadFile = File(...),
    group_a_json: str = Form(...),
    group_b_json: str = Form(...),
    annotation_cols_json: str = Form(default="[]"),
):
    """
    Render the DEG top-gene heatmap as PNG using heatmap_matrix.csv bytes and
    fresh metadata from MongoDB for the samples in that matrix.
    """
    try:
        group_a = json.loads(group_a_json)
        group_b = json.loads(group_b_json)
        annotation_cols = json.loads(annotation_cols_json)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"detail": f"Invalid JSON form fields: {e}"})

    if not isinstance(group_a, list) or not isinstance(group_b, list):
        return JSONResponse(status_code=400, content={"detail": "group_a and group_b must be JSON arrays."})
    if annotation_cols is not None and not isinstance(annotation_cols, list):
        return JSONResponse(status_code=400, content={"detail": "annotation_cols must be a JSON array."})

    raw = await heatmap_csv.read()
    try:
        hm_df = pd.read_csv(io.BytesIO(raw), index_col=0)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Invalid heatmap CSV: {e}"})

    if hm_df.empty:
        return JSONResponse(status_code=400, content={"detail": "Heatmap matrix is empty."})

    cols = annotation_cols if annotation_cols else None
    try:
        mongo_meta = get_metadata_for_samples([str(c) for c in hm_df.columns])
        png_bytes = render_deg_heatmap_png_bytes(
            hm_df,
            group_a,
            group_b,
            mongo_meta,
            annotation_cols=cols,
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e), "status": "error"})

    return Response(content=png_bytes, media_type="image/png")
