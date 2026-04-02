"""
Utility functions for querying the deg_results_map collection in MongoDB.

The collection stores rows from the Webtool_Results_Map CSV file:
  experiment, group, model_design, context, n_samples, sample_names,
  term_label, de_csv_path, de_csv_exists, output_dir, ...

S3 layout for DEG CSVs:
  Bucket: DEG_RESULTS_BUCKET env var (default: brb-seq-data-storage)
  Key:    {output_dir}/{de_csv_path}
"""
import io
import os
import zipfile

import boto3
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------------------------
MONGO_URI = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("Set MONGODB_URI (or MONGO_URI) in the environment")

client = MongoClient(MONGO_URI)
db = client["brb_seq"]
results_col = db["deg_mapping"]

# ---------------------------------------------------------------------------
# S3 connection
# ---------------------------------------------------------------------------
DEG_RESULTS_BUCKET = os.environ.get("DEG_RESULTS_BUCKET", "brb-seq-data-storage")
s3 = boto3.client("s3", region_name=os.environ.get("DEG_RESULTS_REGION", "us-east-2"))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def csv_exists(val) -> bool:
    """Normalise de_csv_exists stored as bool or string 'True'/'False'."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def get_experiments() -> list:
    """Sorted list of unique experiment names in the results map."""
    exps = results_col.distinct("experiment")
    return sorted(str(e) for e in exps if e)


def get_groups_for_experiment(experiment: str) -> list:
    """
    Return one summary dict per unique (group, context) within an experiment.
    Fields: group, model_design, context, n_samples, sample_names.
    Multiple rows per group/context exist (one per term_label); we deduplicate.
    """
    docs = list(results_col.find({"experiment": experiment}, {"_id": 0}))
    if not docs:
        return []

    seen = {}
    for doc in docs:
        grp = doc.get("group", "")
        ctx = doc.get("context", "")
        key = (grp, ctx)
        if key not in seen:
            seen[key] = {
                "group": grp,
                "model_design": doc.get("model_design", ""),
                "context": ctx,
                "n_samples": doc.get("n_samples", ""),
                "sample_names": doc.get("sample_names", ""),
            }

    return list(seen.values())


def get_terms_for_group(experiment: str, group: str, context: str = "") -> list:
    """
    Return available term_labels for a given experiment + group (+ optional context).
    Only includes entries where de_csv_exists is True.
    """
    query = {"experiment": experiment, "group": group}
    if context:
        query["context"] = context

    docs = list(results_col.find(
        query,
        {
            "_id": 0,
            "term_label": 1, "de_csv_path": 1, "de_csv_exists": 1, "output_dir": 1,
            "gsea_barplot_path": 1, "gsea_barplot_exists": 1, "effect_interaction": 1, 
            "reference": 1,
        },
    ))

    return [
        {
            "term_label": doc.get("term_label", ""),
            "de_csv_path": doc.get("de_csv_path", ""),
            "output_dir": doc.get("output_dir", ""),
            "gsea_barplot_path": doc.get("gsea_barplot_path", ""),
            "gsea_barplot_exists": csv_exists(doc.get("gsea_barplot_exists", False)),
            "effect_interaction": doc.get("effect_interaction", ""),
            "reference": doc.get("reference", ""),
        }
        for doc in docs
        if csv_exists(doc.get("de_csv_exists", False))
    ]


def fetch_deg_results_from_s3(
    experiment: str, output_dir: str, de_csv_path: str, gsea_barplot_path: str
) -> StreamingResponse:
    """
    Fetch DEG CSV + GSEA barplot PNG from S3 and return as a ZIP.
    ZIP always contains deg_results.csv.
    gsea_barplot.png is included only when gsea_barplot_path is non-empty AND
    the file is found in S3 — missing or empty barplot is silently skipped.
    """
    # 1. DEG CSV (required — raise 404 if not found)
    csv_key = f"deg_data/{experiment}/{os.path.basename(de_csv_path)}"
    try:
        csv_content = s3.get_object(Bucket=DEG_RESULTS_BUCKET, Key=csv_key)["Body"].read()
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"DEG CSV not found in S3 (bucket={DEG_RESULTS_BUCKET}, key={csv_key}): {exc}",
        )

    # 2. GSEA barplot PNG (optional — silently skip if path is empty or S3 fetch fails)
    barplot_content = None
    if gsea_barplot_path:
        barplot_key = f"deg_data/{experiment}/GSEA/{os.path.basename(gsea_barplot_path)}"
        try:
            barplot_content = s3.get_object(Bucket=DEG_RESULTS_BUCKET, Key=barplot_key)["Body"].read()
        except Exception:
            barplot_content = None

    # 3. Bundle into ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("deg_results.csv", csv_content)
        if barplot_content is not None:
            z.writestr("gsea_barplot.png", barplot_content)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="deg_results.zip"'},
    )
