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
import shutil
import os
from NMF import do_NMF,  preprocess2  # <-- your existing pipeline function
from spearman import run_spearman  # <-- your existing find_k function
import io
import zipfile
from fastapi.responses import StreamingResponse
from typing import Optional
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import seaborn as sns
from cNMF import cNMF_consensus
from save_heatmap_pdf_ordered import save_heatmap_pdf_ordered
from starlette.background import BackgroundTask
from group_associations import run_module_group_analysis
import numpy as np
from pathlib import Path
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from fastapi.encoders import jsonable_encoder
import re
import subprocess
import boto3
import tempfile
from llm_request_functions import model_request, summarize_traceback
import traceback, uuid, datetime, logging

app = FastAPI()

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

    # Return structured JSON response
    return JSONResponse(
        status_code=500,
        content={
            "error_id": error_id,
            "message": summary,
            "details": "This issue has been logged for review."
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
    if request.url.path in ("/healthz", "/"):
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
    counts: UploadFile,
    metadata: UploadFile,
    gene_column: str = Form(...),
    metadata_index: str = Form(...),
    design_factor: str = Form(...),
    symbols: bool = Form(False),
    hvg: int = Form(5000),
    batch: bool = Form(False),
    batch_column: str = Form(""),
    batch_include: str = Form("")
):
    
    tmp_dir = tempfile.mkdtemp(prefix="preprocess_")
    counts_path = os.path.join(tmp_dir, counts.filename)
    metadata_path = os.path.join(tmp_dir, metadata.filename)
    out_dir = None

    try:
        # Save uploaded files to disk
        with open(counts_path, "wb") as f:
            shutil.copyfileobj(counts.file, f)

        with open(metadata_path, "wb") as f:
            shutil.copyfileobj(metadata.file, f)
        
        batch_include_tmp = batch_include.split(",")

        # Call your existing preprocessing function
        df_expr, out_dir = preprocess2(
            counts_link=counts_path,
            metadata_link=metadata_path,
            gene_column=gene_column,
            metadata_index=metadata_index,
            design_factor=design_factor,
            symbols=symbols,
            hvg=hvg,
            batch=batch,
            batch_column=batch_column,
            batch_include=batch_include_tmp
        )

        # serialize DF to feather in-memory
        df_buf = io.BytesIO()
        df_expr.to_feather(df_buf)
        df_buf.seek(0)

        return StreamingResponse(
            df_buf,
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="preprocessed.feather"'}
        )
    
    finally:
        # Clean up temporary files
        for path in [counts_path, metadata_path, out_dir]:
            if path == out_dir:
                shutil.rmtree(out_dir, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)

@app.post("/explore_correlations")
async def explore_corr(
    preprocessed: UploadFile,
    design_factor: str = Form(...),
    k: int = Form(...),
):
    df_expr = pd.read_feather(io.BytesIO(await preprocessed.read()))
    #df_expr = df_expr.set_index("SampleName")
    try:
        pairs_by_k = run_spearman(
            df_expr=df_expr,
            k=k,
            design_factor=design_factor,
        )
        print(pairs_by_k)

        return JSONResponse(content={"pairs_by_k": pairs_by_k})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# def _safe_obj(x):
#     """Recursively cast to JSON-serializable Python types."""
#     if x is None:
#         return None
#     if isinstance(x, (str, int, float, bool)):
#         return x
#     if isinstance(x, (np.integer,)):   return int(x)
#     if isinstance(x, (np.floating,)):  return float(x)
#     if isinstance(x, (np.bool_,)):     return bool(x)
#     if isinstance(x, (np.ndarray,)):   return x.tolist()
#     if isinstance(x, Path):            return str(x)
#     if isinstance(x, pd.Timestamp):    return x.isoformat()
#     if isinstance(x, dict):            return {str(k): _safe_obj(v) for k, v in x.items()}
#     if isinstance(x, (list, tuple, set)): return [_safe_obj(v) for v in x]
#     if isinstance(x, pd.DataFrame):
#         df = x.where(pd.notnull(x), None)  # NaN -> None
#         recs = df.to_dict(orient="records")
#         return [_safe_obj(r) for r in recs]
#     if isinstance(x, pd.Series):
#         s = x.where(pd.notnull(x), None).to_dict()
#         return {str(k): _safe_obj(v) for k, v in s.items()}
#     # Fallback: stringize
#     return str(x)


def process_single_run(expr_path, meta_path, k, run, max_iter, design_factor, sample_column, tmp_dir):

    #samples = df_expr["SampleName"].copy()
    #expr = df_expr.drop(columns=["SampleName"])
    df_expr = pd.read_feather(expr_path)
    df_meta = pd.read_csv(meta_path)
    meta = df_meta[["SampleName", design_factor]]
    scores = []

    for i in range(1, 11):
        pid = os.getpid()
        start = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Start K={k}, run={i}, PID={pid}")

        seed = random.randint(1, 42)
        _, df_w = do_NMF(df_expr, design_factor, n_components=k, max_iter=max_iter, seed=seed)

        out = os.path.join(tmp_dir, f"across_k_{k}/run_{run}")
        os.makedirs(out, exist_ok=True)

        #df_w["SampleName"] = samples.loc[df_w.index]
        
        #df_w = df_w.drop(columns=[design_factor])

        print("finding score")
        score = run_module_group_analysis(
            usage_df=df_w,
            metadata=meta,
            #transpose_matrix=True,
            #matrix_sep=None,
            output_dir=out,
            example_data=False,
            design_factor=design_factor,
            sample_column=sample_column
        )
        end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Done  K={k}, run={i}, PID={pid}, took {end-start:.2f}s")

        scores.append({
            "k": k,
            "rep": i,
            "silhouette": float(score)  # cast the value, not the dict
        })

    return scores


@app.post("/k_metrics")
async def find_k_metrics(
    preprocessed_feather: bytes = File(...),
    meta: bytes = File(...),
    k_min: int = Form(...),
    k_max: int = Form(...),
    max_iter: int = Form(...),
    design_factor: str = Form(...),
    sample_column: str = Form(...),
    n_reps: int = 1  # number of replicates per K
):
    tmp_dir = tempfile.mkdtemp(prefix="kmetrics_")
    expr_path = os.path.join(tmp_dir, "expr.feather")
    meta_path = os.path.join(tmp_dir, "meta.csv")
    start = time.time()

    try:
        # Save expression + metadata once
        pd.read_feather(io.BytesIO(preprocessed_feather)).to_feather(expr_path)
        try:
            df_meta = pd.read_csv(io.BytesIO(meta), sep=None, engine="python")
        except Exception:
            df_meta = pd.read_csv(io.BytesIO(meta))
        df_meta.to_csv(meta_path, index=False)

        # Load once (to pass directly to workers instead of re-reading per job)
       # df_expr = pd.read_feather(expr_path)
        results = []

        # Launch jobs: (k, run)
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_single_run, expr_path, meta_path, k, run, max_iter, design_factor, sample_column, tmp_dir): (k, run)
                for k in range(k_min, k_max + 1)
                for run in range(1, n_reps + 1)
            }

            for future in as_completed(futures):
                k_done, run_done = futures[future]
                #try:
                scores_list = future.result()  # ✅ now this is just the silhouette score
                results.extend(scores_list)
                # except Exception as e:
                #     print(f"Error with K={k_done}, run={run_done}: {e}")

        end = time.time()
        print(f"All jobs done in {end - start:.1f} seconds")

        # Combine results across Ks/runs
        #combine("/tmp")

        # return FileResponse(
        #     path="/tmp/permanova_summary_all.tsv",
        #     filename="permanova_summary_all.tsv",
        #     media_type="text/tab-separated-values"
        # )
        return JSONResponse(content={"results": results})

    finally:
        # Cleanup temp dir with inputs
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        # Cleanup across_k directories
        # for k in range(k_min, k_max + 1):
        #     dir_path = f"/tmp/across_k_{k}"
        #     if os.path.exists(dir_path):
        #         shutil.rmtree(dir_path)
    

def safe_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def cleanup_after_send(files=None, dirs=None):
    files = files or []
    dirs  = dirs or []
    for p in files:
        safe_remove(p)
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

@app.post("/run_regular_nmf")
async def run_nmf_files(
    preprocessed: UploadFile,
    k: int = Form(...),
    max_iter: int = Form(...),
    design_factor: str = Form(...)
):
    df_expr = pd.read_feather(io.BytesIO(await preprocessed.read()))
    try:
        df_h, df_w = do_NMF(
            n_components = k,
            df_expr = df_expr,
            max_iter = max_iter,
            design_factor = design_factor 
        )
        #df_buf = io.BytesIO()
        #df_w.to_feather(df_buf)
        #df_buf.seek(0)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
            h_buf = io.BytesIO()
            df_h.to_feather(h_buf)
            zip_file.writestr("nmf_h.feather", h_buf.getvalue())

            w_buf = io.BytesIO()
            df_w.to_feather(w_buf)
            zip_file.writestr("nmf_w.feather", w_buf.getvalue())

        zip_buf.seek(0)
        return StreamingResponse(
            iter([zip_buf.getvalue()]),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="nmf_results.zip"'}
        )

    finally:
        print("Done")


#Function to run consensus NMF 
@app.post("/run_nmf_files")
async def run_nmf_files(
    preprocessed: UploadFile,
    metadata: UploadFile,
    k: int = Form(...),
    hvg: int = Form(...),
    max_iter: int = Form(...),
    design_factor: str = Form(...),
    metadata_index: str = Form(...)
):
    
    df_expr = pd.read_feather(io.BytesIO(await preprocessed.read()))
    if "Geneid" in df_expr.columns:
        df_expr = df_expr.set_index("Geneid")
    elif "index" in df_expr.columns:   # fallback if column is called 'index'
        df_expr = df_expr.set_index("index")

    out_dir = tempfile.mkdtemp(prefix="cnmf_")
    df_expr.index.name = None
    #metadata = pd.read_feather(io.BytesIO(await metadata.read()))
    metadata = pd.read_csv(io.BytesIO(await metadata.read()), sep="\t")
    #df_expr = df_expr.set_index("SampleName")
    try:
        # Call your existing NMF function
        cNMF_consensus(
            k=k,
            hvg=hvg,
            df_expr=df_expr,
            max_iter=max_iter,
            design_factor=design_factor,
            out_dir=out_dir
        )

        usages_path = os.path.join(out_dir, f"example_CNMF/cNMF/cNMF.usages.k_{k}.dt_0_2.consensus.txt")

        # unique temp file to avoid clashes
        pdf_path = os.path.join(out_dir, f"nmf_heatmap_{k}.pdf")
        save_heatmap_pdf_ordered(usages_path=usages_path, metadata=metadata, out_pdf = pdf_path, index=metadata_index)
        module_usage_path = os.path.join(out_dir, f"example_CNMF/cNMF/cNMF.usages.k_{k}.dt_0_2.consensus.txt")
        gene_zscore_path = os.path.join(out_dir, f"example_CNMF/cNMF/cNMF.gene_spectra_score.k_{k}.dt_0_2.txt")

        out_zip=os.path.join(out_dir, "cnmf_bundle.zip")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            # add the PDF by path
            z.write(pdf_path, arcname="nmf_heatmap.pdf")  # rename inside zip if you want
            # add your text files
            for p in [module_usage_path, gene_zscore_path]:
                if os.path.isfile(p):
                    z.write(p, arcname=os.path.basename(p))
        
        task = BackgroundTask(
            cleanup_after_send,
            files=[out_zip, pdf_path],
            dirs=[out_dir]
        )

        return FileResponse(
            out_zip,
            media_type="application/zip",
            filename=f"cnmf_k{k}_bundle.zip",
            background=task,   # <- Starlette BackgroundTask
        )
    finally:
        print("Done:")

#break data into chunks to send to chatGPT
def break_chunks(genes, size):
    for i in range(0, len(genes), size):
        yield genes[i: i + size]

#Make request to ChatGPT to summarise gene functions
@app.post("/process_gene_loadings/")
async def process_gene_loadings(
    file: UploadFile,
    top_n: int = Form(2)
):
    # Read feather file into DataFrame
    content = await file.read()
    df = pd.read_feather(io.BytesIO(content))
    if df is not None:
        if df.columns[0] == "Module":
            tmp = list(df["Module"])
            modules = ["Module_" + str(x) for x in tmp]
            df = df.drop("Module", axis= 1)
            df = df.T
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "Gene"})
            df.columns = ["Gene"] + modules

        else:
            modules = list(df.index.astype(str)) 
            df = df.T
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "Gene"})
            df.columns = ["Gene"] + modules
    
    print(df.head)


    #if "Gene" not in df.columns:
     #   return JSONResponse(status_code=400, content={"error": "DataFrame must have a 'Gene' column"})

    hmap = {}
    all_top_genes = set()
    for module in df.columns:
        if module == "Gene":
            continue
        tmp = df[["Gene", module]].copy()
        tmp = tmp.sort_values(by=module, ascending=False).head(top_n)
        hmap[module] = tmp
        all_top_genes.update(tmp["Gene"])

    # Parallel API calls
    output = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(model_request, batch) for batch in break_chunks(list(all_top_genes), 20)]
        for future in as_completed(futures):
            try:
                result = future.result()
                output.update(result)
            except Exception as e:
                print("API error:", e)

    # Attach descriptions back to each module
    results = {}
    for module, tmp in hmap.items():
        tmp = tmp.copy()
        tmp["Description"] = tmp["Gene"].map(output)
        results[module] = tmp.to_dict(orient="records")

    return JSONResponse(content=jsonable_encoder(results))


# #Run Pathway Analysis
# @app.post("/run_pathview/")
# async def run_pathview(file: UploadFile = File(...)):
#     # 1. Save uploaded file temporarily
#     input_path = f"/tmp/{file.filename}"
#     with open(input_path, "wb") as f:
#         f.write(await file.read())
    
#     out_root = "/tmp/pathview_output"
#     os.makedirs(out_root, exist_ok=True)

#     # 2. Call R script
#     result = subprocess.run(
#         ["Rscript", "pathway_analysis.R", input_path, out_root],
#         #capture_output=True,
#         #text=True,
#         check=True
#     )
#     print("STDOUT:\n", result.stdout)
#     print("STDERR:\n", result.stderr)

#     zip_path = out_root + "/pathway_results.zip"   # R printed the zip path
    
#     # 3. Return zip file to user
#     response = FileResponse(
#         zip_path,
#         filename="pathway_results.zip",
#         media_type="application/zip"
#     )


#     # 4. Cleanup after sending
#     @response.background
#     def cleanup():
#         os.remove(zip_path)
#         os.remove(input_path)
#         os.rmdir(out_root)

#     return response