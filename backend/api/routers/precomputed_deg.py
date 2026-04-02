from fastapi import APIRouter

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
