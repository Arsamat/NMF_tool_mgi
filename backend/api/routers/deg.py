import traceback

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from api.schemas.deg import DEGGroupsRequest
from deg.deg_utils import run_deg_analysis
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
