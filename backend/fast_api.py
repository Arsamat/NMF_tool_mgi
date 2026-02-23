import os

# limit BLAS threads per worker to avoid deadlocks
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from fastapi import FastAPI, UploadFile, Form, File, BackgroundTasks, Request
from fastapi.responses import FileResponse
import pandas as pd
import os
from spearman import run_spearman  # <-- your existing find_k function
import io
import zipfile
from fastapi.responses import JSONResponse
import json
import subprocess
import boto3
from llm_request_functions import model_request, summarize_traceback
import traceback, uuid, datetime, logging
from plot_heatmap_helpers import create_dendrogram, make_heatmap_png, build_annotations, dendogram_modules, module_annotated_heatmap
from utils.heatmap_utils import plot_expression_heatmap, annotated_heatmap_util, heatmap_top_samples_utils, module_heatmap_utils, hypergeom_endpoint_utils, cluster_samples_utils, preview_wide_heatmap_inline
from utils.db_utils import get_all_metadata_values, query_counts, query_metadata, update_database
from utils.nmf_utils import run_regular_nmf_util, run_cnmf_utils, cleanup_after_send, preprocess_util, explore_k_utils, single_cell_util
from utils.extra_utils import gpt_utils
from utils.s3_utils import create_url, create_preprocessed_url, download_data_util
from fastapi import Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "https://nmftoolmgi.streamlit.app",
    # or your custom domain if you have one
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # use ["*"] only for quick testing
    allow_credentials=False,         # keep False unless you truly need cookies/auth
    allow_methods=["*"],             # includes OPTIONS
    allow_headers=["*"],             # includes Content-Type, Authorization, etc.
)

LOG_FILE = "error.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions, log them, and return a friendly AI-generated summary."""
    error_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().isoformat()
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Log full traceback
    log_message = (
        f"ID: {error_id}\n"
        f"Time: {timestamp}\n"
        f"Path: {request.url.path}\n"
        f"Error: {exc}\n"
        f"Traceback:\n{tb}\n"
        f"{'-'*60}\n"
    )
    logging.error(log_message)

    # Call OpenAI synchronously to summarize
    summary = summarize_traceback(tb)

    # Return structured JSON response (include raw exception for debugging)
    return JSONResponse(
        status_code=500,
        content={
            "error_id": error_id,
            "message": summary,
            "details": "This issue has been logged for review.",
            "raw_error": str(exc),
        },
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}

# Create a CloudWatch client (uses IAM Role attached to EC2 automatically)
cloudwatch = boto3.client("cloudwatch", region_name="us-east-2")

# Middleware to log every request into CloudWatch
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip internal health checks
    if request.url.path in ("/"):
        return await call_next(request)

    try:
        cloudwatch.put_metric_data(
            Namespace="FastAPIApp",
            MetricData=[
                {
                    "MetricName": "RequestCount",
                    "Dimensions": [
                        {"Name": "Service", "Value": "FastAPI"}
                    ],
                    "Value": 1,
                    "Unit": "Count"
                }
            ]
        )
    except Exception as e:
        print(f"Failed to push metric: {e}")

    return await call_next(request)

@app.post("/preprocess_df")
async def preprocess_data(
    metadata: UploadFile,
    gene_column: str = Form(...),
    metadata_index: str = Form(...),
    design_factor: str = Form(...),
    symbols: bool = Form(False),
    hvg: int = Form(5000),
    batch: bool = Form(False),
    batch_column: str = Form(""),
    batch_include: str = Form(""),
    job_id: str = Form(...),
    single_cell: bool = Form(False)
):
    print(single_cell)
    return await preprocess_util(
        metadata, gene_column, metadata_index, design_factor, symbols, hvg, batch, batch_column, batch_include, job_id, single_cell
    )
    
    

@app.post("/explore_correlations")
async def explore_corr(gene_loadings: UploadFile):
    import traceback
    try:
        content = await gene_loadings.read()
        df_h = pd.read_feather(io.BytesIO(content))
        print("Received df_h shape:", df_h.shape)

        pairs_by_k = run_spearman(df_H=df_h)
        print("pairs_by_k:", pairs_by_k)

        return JSONResponse(content={"pairs_by_k": pairs_by_k})

    except Exception as e:
        tb = traceback.format_exc()
        print("Error in /explore_correlations:", tb)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb}
        )


@app.post("/k_metrics")
async def find_k_metrics(
    #preprocessed_feather: bytes = File(...),
    meta: bytes = File(...),
    job_id: str = Form(...),
    k_min: int = Form(...),
    k_max: int = Form(...),
    max_iter: int = Form(...),
    design_factor: str = Form(...),
    sample_column: str = Form(...),
    n_reps: int = 1  # number of replicates per K
):
    return await explore_k_utils(
        #preprocessed_feather,
        meta,
        job_id,
        k_min,
        k_max,
        max_iter,
        design_factor,
        sample_column,
        n_reps
    )
    
    

@app.post("/run_regular_nmf")
async def run_nmf_files(
    #preprocessed: UploadFile,
    job_id: str = Form(...),
    k: int = Form(...),
    max_iter: int = Form(...),
    design_factor: str = Form(...)
):
    
    return await run_regular_nmf_util(#preprocessed,
    job_id,
    k,
    max_iter,
    design_factor)


#Function to run consensus NMF 
@app.post("/run_nmf_files")
async def run_cnmf_files(
    #preprocessed: UploadFile,
    metadata: UploadFile,
    k: int = Form(...),
    hvg: int = Form(...),
    max_iter: int = Form(...),
    design_factor: str = Form(...),
    metadata_index: str = Form(...),
    job_id: str = Form(...),
    batch_correct: str = Form(""),
    gene_column: str = Form(...)
):
    
    return await run_cnmf_utils(
        #preprocessed,
        metadata,
        k,
        hvg,
        max_iter,
        design_factor,
        metadata_index,
        job_id,
        batch_correct,
        gene_column
)

@app.post("/cnmf_single_cell")
async def run_cnmf_single_cell(
    #preprocessed: UploadFile,
    metadata: UploadFile,
    k: int = Form(...),
    hvg: int = Form(...),
    #max_iter: int = Form(...),
    design_factor: str = Form(...),
    metadata_index: str = Form(...),
    job_id: str = Form(...),
    batch_correct: str = Form(""),
    gene_column: str = Form(...),
    gene_symbols: bool = Form(True)
):
    
    return await single_cell_util(
        #preprocessed,
        metadata,
        k,
        hvg,
        #max_iter,
        design_factor,
        metadata_index,
        job_id,
        batch_correct,
        gene_column, 
        gene_symbols
)

#Make request to ChatGPT to summarise gene functions
@app.post("/process_gene_loadings/")
async def process_gene_loadings(
    file: UploadFile,
    top_n: int = Form(2)
):
    
    return await gpt_utils(file, top_n)


#Run Pathway Analysis
@app.post("/run_pathview/")
async def run_pathview(file: UploadFile = File(...), 
                gene_format: str = Form(None)):
    # 1. Save uploaded file temporarily
    input_path = f"/tmp/{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    out_root = "/tmp/pathview_output"
    os.makedirs(out_root, exist_ok=True)
    print("Input path:", input_path)
    print("Output dir:", out_root)
    print("Gene format:", gene_format)
    # 2. Call R script
    result = subprocess.run(
        ["Rscript", "pathway_analysis.R", input_path, out_root, gene_format],
        #capture_output=True,
        #text=True,
        check=True
    )
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    zip_path = out_root + "/pathway_results.zip"   # R printed the zip path

    # task = BackgroundTask(
    #         cleanup_after_send,
    #         files=[zip_path, input_path],
    #         dirs=[out_root]
    #     )
    # 3. Return zip file to user
    zip_path = "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(out_root + "/pathway_results.zip",  arcname="pathway_results.zip")
        zipf.write(out_root + "/kegg_dataframe.csv",  arcname="kegg_dataframe.csv")
    return FileResponse(zip_path, media_type="application/zip", filename="bundle.zip")


# ----------------------------------------------
# Main API Endpoint
# ----------------------------------------------

@app.post("/initial_heatmap_preview/")
async def plot_heatmap(
    df: UploadFile,
    metadata: UploadFile,
    metadata_index: str = Form(...),
    annotation_cols: str = Form(...),
    average_groups: str = Form(...),
):
    """
    Returns a PNG heatmap computed fully in-memory.
    """

    df_bytes = await df.read()
    meta_bytes = await metadata.read()
    

    df = pd.read_feather(io.BytesIO(df_bytes))
    df = df.set_index("index")
    meta = pd.read_feather(io.BytesIO(meta_bytes))
    annotation_cols = json.loads(annotation_cols)

    
    average_groups = average_groups.lower() == "true"

    return await preview_wide_heatmap_inline(
        df,
        meta,
        metadata_index,
        annotation_cols,
        average_groups
    )

@app.post("/plot_heatmap/")
async def plot_heatmap(
    gene_loadings: UploadFile = File(...),
    #preprocessed_df: UploadFile = File(...),
    job_id: str = Form(...),
    metadata: UploadFile = File(...),
    annotation_cols: str = Form(...),     # JSON string list
    metadata_index: str = Form(...),
    X: int = Form(10),
):
    """
    Returns a PNG heatmap computed fully in-memory.
    """
    return await plot_expression_heatmap(gene_loadings,
    #preprocessed_df,
    job_id,
    metadata,
    annotation_cols,
    metadata_index,
    X)

#Hierarchical Clustering Heatmap
@app.post("/cluster_samples/")
async def cluster_samples(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    k: int = Form(3)
):
    
    return await cluster_samples_utils(module_usages,
    metadata,
    metadata_index,
    k)

@app.post("/annotated_heatmap/")
async def annotated_heatmap(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    leaf_order: str = Form(...),
    annotation_cols: str = Form("[]"),
    cluster_labels: str = Form("[]")
):
    return await annotated_heatmap_util(
        module_usages,
        metadata,
        metadata_index,
        leaf_order,
        annotation_cols,
        cluster_labels
    )

@app.post("/hypergeom/")
async def hypergeom_endpoint(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    cluster_labels: str = Form(...),
    cluster_id: int = Form(...),
    selected_values: str = Form(...)   # {"Genotype":"TDP43", "Dose":"EC10"}
):
    
    return await hypergeom_endpoint_utils(
        module_usages,
        metadata,
        metadata_index,
        cluster_labels,
        cluster_id,
        selected_values
    )


@app.post("/cluster_modules/")
async def cluster_modules(
    module_usages: UploadFile = File(...),
    sample_order: str = Form(...),
    n_clusters: int = Form(...)
):
    # ----------------------------------------------------------
    # 1. Decode sample ordering (list of sample names)
    # ----------------------------------------------------------
    sample_order = json.loads(sample_order)
    # ----------------------------------------------------------
    # 2. Load module usage matrix
    # ----------------------------------------------------------
    df = pd.read_feather(io.BytesIO(await module_usages.read()))
    df.columns = ["Sample"] + list(df.columns[1:])
    df = df.set_index("Sample")
    # ----------------------------------------------------------
    # 3. Reorder samples using sample clustering order
    # ----------------------------------------------------------
    df = df.loc[sample_order]      # samples now in correct order
    df = df.T                      # modules × samples

    
    return dendogram_modules(df, n_clusters)

@app.post("/module_heatmap/")
async def module_heatmap(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    sample_order: str = Form(...),
    module_leaf_order: str = Form(...),
    module_cluster_labels: str = Form(...),
    annotation_cols: str = Form("[]")
):
    
    return await module_heatmap_utils(
        module_usages, metadata, metadata_index, sample_order, module_leaf_order, module_cluster_labels, annotation_cols
    )

@app.post("/heatmap_top_samples/")
async def heatmap_top_samples(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    annotation_cols: str = Form(""),
    metadata_index: str = Form(...)
):
    
    return await heatmap_top_samples_utils(
        module_usages,
        metadata,
        annotation_cols,
        metadata_index
    )
#Obtain all columns present in the metadata data base
@app.get("/get_metadata/")
async def get_metadata_values():
    return get_all_metadata_values()

#Get metadata for samples that match a filtering condition provided by the user
@app.post("/get_samples/")
async def get_sample_counts(request: Request):
    filters = await request.json()

    result = query_metadata(filters)

    return result



@app.post("/get_counts/")
async def get_sample_counts(
    metadata: UploadFile = File(...)
):
    return await query_counts(metadata)
    
@app.post("/update_db/")
async def get_sample_counts(
    counts_table: UploadFile = File(...),
    metadata: UploadFile = File(...)
):
    result = await update_database(counts_table, metadata)
    return result

@app.post("/create_upload_url")
def create_upload_url():
    return create_url()
 
@app.post("/create_preprocessed_upload_url")
def create_preprocessed_upload():
    return create_preprocessed_url()

@app.get("/download_preprocessed_data")
def download_data(job_id: str = Query(...), data_type: str = Query(...)):
    return download_data_util(job_id, data_type)

BUCKET = "nmf-tool-bucket"
REGION = "us-east-2"

s3 = boto3.client("s3", region_name=REGION)

class CreateMultipartRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"

@app.get("/create_multipart_upload")
def create_multipart_upload():
    job_id = str(uuid.uuid4())
    key = f"jobs/{job_id}/counts.csv"

    resp = s3.create_multipart_upload(
        Bucket=BUCKET,
        Key=key,
        #ContentType=req.content_type,
    )
    return {"job_id": job_id, "key": key, "uploadId": resp["UploadId"]}

class SignPartRequest(BaseModel):
    key: str
    uploadId: str
    partNumber: int

@app.post("/sign_part")
def sign_part(req: SignPartRequest):
    url = s3.generate_presigned_url(
        ClientMethod="upload_part",
        Params={
            "Bucket": BUCKET,
            "Key": req.key,
            "UploadId": req.uploadId,
            "PartNumber": req.partNumber,
        },
        ExpiresIn=3600,
    )
    return {"url": url}

class CompleteMultipartRequest(BaseModel):
    key: str
    uploadId: str
    parts: list  # [{ "ETag": "...", "PartNumber": 1 }, ...]

@app.post("/complete_multipart_upload")
def complete_multipart_upload(req: CompleteMultipartRequest):
    resp = s3.complete_multipart_upload(
        Bucket=BUCKET,
        Key=req.key,
        UploadId=req.uploadId,
        MultipartUpload={"Parts": req.parts},
    )
    return {"location": resp.get("Location"), "etag": resp.get("ETag")}
