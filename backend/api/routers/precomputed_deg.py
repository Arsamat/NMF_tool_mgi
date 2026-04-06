import json
import traceback

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from api.schemas.precomputed_deg import (
    DEGResultsFetchRequest,
    DEGResultsGroupsRequest,
    DEGResultsTermsRequest,
)
from deg.precomputed_deg_utils import (
    fetch_deg_results_from_s3,
    get_experiments,
    get_groups_for_experiment,
    get_terms_for_group,
)
from deg.precomputed_heatmap import build_precomputed_deg_heatmap_png_bytes_from_csv_bytes

router = APIRouter(tags=["precomputed-deg"])


@router.get("/deg_results/experiments")
def list_deg_experiments():
    """Return sorted list of unique experiments in the precomputed results map."""
    return {"experiments": get_experiments()}


@router.post("/deg_results/groups")
def list_deg_groups(req: DEGResultsGroupsRequest):
    """Return one summary row per unique group for the given experiment."""
    return {"groups": get_groups_for_experiment(req.experiment)}


@router.post("/deg_results/terms")
def list_deg_terms(req: DEGResultsTermsRequest):
    """Return available terms for an experiment + group (+ optional context)."""
    return {"terms": get_terms_for_group(req.experiment, req.group, req.context)}


@router.post("/deg_results/fetch_csv")
def fetch_precomputed_deg_results(req: DEGResultsFetchRequest):
    """Fetch DEG CSV + GSEA barplot PNG from S3, returned as a ZIP."""
    return fetch_deg_results_from_s3(
        req.experiment, req.output_dir, req.de_csv_path, req.gsea_barplot_path
    )


@router.post("/deg_results/precomputed_heatmap")
async def precomputed_deg_heatmap(
    deg_csv: UploadFile = File(...),
    samples_json: str = Form(...),
    annotation_cols_json: str = Form(default="[]"),
):
    """
    Top-30-by-p-value genes × samples log₂(CPM+1) heatmap with Mongo metadata bars
    (CellType, Treatment, Genotype, Timepoint). No Group bar.
    """
    try:
        samples = json.loads(samples_json)
        annotation_cols = json.loads(annotation_cols_json)
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=400, content={"detail": f"Invalid JSON: {e}"})

    if not isinstance(samples, list) or not all(isinstance(s, str) for s in samples):
        return JSONResponse(status_code=400, content={"detail": "samples_json must be a JSON array of strings."})
    if annotation_cols is not None and not isinstance(annotation_cols, list):
        return JSONResponse(status_code=400, content={"detail": "annotation_cols_json must be a JSON array."})

    raw = await deg_csv.read()
    cols = annotation_cols if annotation_cols else None
    try:
        png = build_precomputed_deg_heatmap_png_bytes_from_csv_bytes(
            raw,
            samples,
            annotation_cols=cols,
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e), "status": "error"})

    return Response(content=png, media_type="image/png")
