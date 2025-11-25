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
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import hypergeom
from plot_heatmap_helpers import create_dendrogram, make_heatmap_png, build_annotations, dendogram_modules, module_annotated_heatmap
import base64
import matplotlib.gridspec as gridspec

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
        df_h, df_w, _ = do_NMF(df_expr, design_factor, n_components=k, max_iter=max_iter, seed=seed)

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
        df_h, df_w, converged = do_NMF(
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

            metadata = {"converged": bool(converged)}
            zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))

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

def build_color_annotations(meta_aligned, annotation_cols):
    palettes = [
        "Set1", "Set2", "Paired", "Pastel1", "Pastel2",
        "Dark2", "Accent", "tab10", "tab20"
    ]

    col_colors = pd.DataFrame(index=meta_aligned.index)
    lut = {}

    if annotation_cols:
        for i, col in enumerate(annotation_cols):
            if col not in meta_aligned.columns:
                continue

            unique_vals = pd.unique(meta_aligned[col].dropna())
            palette = sns.color_palette(palettes[i % len(palettes)], n_colors=len(unique_vals))
            lut[col] = dict(zip(unique_vals, palette))

            col_colors[col] = meta_aligned[col].astype(object).map(lut[col])

    return col_colors, lut


def make_heatmap(df_reordered, col_colors, lut, annotation_cols):
    num_genes = df_reordered.shape[0]
    fig_height = max(10, num_genes * 0.2)

    g = sns.clustermap(
        df_reordered,
        cmap=sns.color_palette("Reds", as_cmap=True),
        figsize=(30, fig_height),
        col_colors=col_colors if col_colors is not None and not col_colors.empty else None,
        col_cluster=False,
        row_cluster=False,
        cbar_kws={"orientation": "vertical", "shrink": 0.5},
        xticklabels=False,
        yticklabels=True
    )

    # Move colorbar left
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.ax.set_position([0.08, 0.25, 0.02, 0.4])
    cbar.set_label("Expression Level", fontsize=12)

    # Fix labels
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=12)
    g.ax_heatmap.tick_params(axis="y", labelsize=11, pad=10)
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_ylabel("Genes", fontsize=12)

    # Add annotation legends
    if annotation_cols and lut and col_colors is not None and not col_colors.empty:
        legend_entries = []
        for col in annotation_cols:
            if col not in lut:
                continue
            for label, color in lut[col].items():
                legend_entries.append(mpatches.Patch(color=color, label=f"{col}: {label}"))

        if legend_entries:
            g.ax_col_dendrogram.legend(
                handles=legend_entries,
                loc="center",
                ncol=2,
                frameon=False,
                prop={'size': 14},
            )

    plt.title("Top Genes Expression Heatmap", fontsize=14, pad=10)
    return g


# ----------------------------------------------
# Main API Endpoint
# ----------------------------------------------
@app.post("/plot_heatmap/")
async def plot_heatmap(
    gene_loadings: UploadFile = File(...),
    preprocessed_df: UploadFile = File(...),
    metadata: UploadFile = File(...),
    annotation_cols: str = Form(...),     # JSON string list
    metadata_index: str = Form(...),
    X: int = Form(10),
):
    """
    Returns a PNG heatmap computed fully in-memory.
    """
    # -----------------------------
    # Load feather inputs
    # -----------------------------
    gl_bytes = await gene_loadings.read()
    pp_bytes = await preprocessed_df.read()
    meta_bytes = await metadata.read()

    gene_loadings_df = pd.read_feather(io.BytesIO(gl_bytes))
    df = pd.read_feather(io.BytesIO(pp_bytes))
    meta = pd.read_feather(io.BytesIO(meta_bytes))

    # Convert annotation_cols string → list
    
    annotation_cols = json.loads(annotation_cols)

    # -----------------------------
    # Extract top X genes per module
    # -----------------------------
    df_long = gene_loadings_df.stack().reset_index()
    df_long.columns = ["module", "gene", "loading"]
    df_long = df_long.drop(0, errors="ignore")

    unique_df = df_long.loc[df_long.groupby("gene")["loading"].idxmax()]

    grouped_genes = (
        unique_df.sort_values(["module", "loading"], ascending=[True, False])
        .groupby("module")
        .head(X)
    )

    module_labels = grouped_genes[["module", "gene"]]
    cluster_boundaries = []

    for i in range(1, len(module_labels)):
        if module_labels.iloc[i, 0] != module_labels.iloc[i - 1, 0]:
            cluster_boundaries.append(i)

    genes_graph = list(grouped_genes["gene"])

    # -----------------------------
    # Build expression matrix
    # -----------------------------
    df = df.T.reset_index()
    df = df[df["Geneid"].isin(genes_graph)]
    df["Category"] = pd.Categorical(df["Geneid"], categories=genes_graph, ordered=True)
    df_ordered = df.sort_values("Category").drop(columns={"Category"})
    df_plot = df_ordered.set_index("Geneid")

    # -----------------------------
    # Align metadata
    # -----------------------------
    meta = meta.set_index(metadata_index)
    meta_aligned = meta.loc[df_plot.columns.intersection(meta.index)]

    # -----------------------------
    # Build annotation & plot
    # -----------------------------
    col_colors, lut = build_color_annotations(meta_aligned, annotation_cols)
    g = make_heatmap(df_plot, col_colors, lut, annotation_cols)

    # Add cluster separation lines
    for y in cluster_boundaries:
        g.ax_heatmap.hlines(
            y=y,
            xmin=0,
            xmax=df_plot.shape[1],
            colors="black",
            linewidth=3,
            linestyles="-",
            zorder=10,
        )

    # -----------------------------
    # Return PNG buffer
    # -----------------------------
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

#Hierarchical Clustering Heatmap
@app.post("/cluster_samples/")
async def cluster_samples(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    k: int = Form(3)
):
    df = pd.read_feather(io.BytesIO(await module_usages.read()))
    df.columns = ["Sample"] + list(df.columns[1:])
    df = df.set_index("Sample")

    meta = pd.read_feather(io.BytesIO(await metadata.read())).set_index(metadata_index)

    # --- clustering
    dist = pdist(df, metric="euclidean")
    Z = linkage(dist, method="ward")
    cluster_labels = fcluster(Z, k, criterion="maxclust")

    # --- dendrogram
    leaf_order, dendro_png = create_dendrogram(Z, cluster_labels, "Sample Clustering")

    # --- reorder df
    ordered_cols = df.index[leaf_order]
    heatmap_png = make_heatmap_png(df.loc[ordered_cols].T)

    return {
        "leaf_order": leaf_order,
        "cluster_labels": cluster_labels.tolist(),
        "dendrogram_png": dendro_png.getvalue().hex(),
        "heatmap_png": heatmap_png.getvalue().hex(),
    }

@app.post("/annotated_heatmap/")
async def annotated_heatmap(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    leaf_order: str = Form(...),
    annotation_cols: str = Form("[]"),
    cluster_labels: str = Form("[]")
):
    leaf_order = json.loads(leaf_order)
    annotation_cols = json.loads(annotation_cols)
    cluster_labels = json.loads(cluster_labels)

    # Load module usage matrix
    df = pd.read_feather(io.BytesIO(await module_usages.read()))
    df.columns = ["Sample"] + list(df.columns[1:])
    df = df.set_index("Sample")

    # Compute sample order from leaf indices
    sample_order = df.index[leaf_order].tolist()
    ordered_df = df.loc[sample_order].T

    # Load metadata aligned to sample order
    meta = pd.read_feather(io.BytesIO(await metadata.read())).set_index(metadata_index)
    meta_aligned = meta.loc[sample_order]

    # Handle Cluster annotation
    if "Cluster" in annotation_cols:

        if len(cluster_labels) != len(df.index):
            raise ValueError("Cluster label length mismatch")

        cluster_labels_ordered = [cluster_labels[i] for i in leaf_order]
        meta_aligned["Cluster"] = cluster_labels_ordered

    # Build annotations
    col_colors_df, lut = build_annotations(meta_aligned, annotation_cols)

    # IMPORTANT: pass DataFrame directly — NO transpose
    col_colors = col_colors_df

    if len(annotation_cols) == 0:
        col_colors = None
        lut = {}
    else:
        col_colors, lut = build_annotations(meta_aligned, annotation_cols)

    # Render heatmap
    heatmap_png = make_heatmap_png(
        df=ordered_df,
        col_colors=col_colors,
        lut=lut,
    )

    return StreamingResponse(heatmap_png, media_type="image/png")

@app.post("/hypergeom/")
async def hypergeom_endpoint(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    cluster_labels: str = Form(...),
    cluster_id: int = Form(...),
    selected_values: str = Form(...)   # {"Genotype":"TDP43", "Dose":"EC10"}
):
    # Decode JSON inputs
    cluster_labels = json.loads(cluster_labels)
    selected_values = json.loads(selected_values)

    # ---------------------------------------------------------
    # Load dataframes
    # ---------------------------------------------------------
    mod = pd.read_feather(io.BytesIO(await module_usages.read()))
    mod.columns = ["Sample"] + list(mod.columns[1:])
    mod = mod.set_index("Sample")

    meta = pd.read_feather(io.BytesIO(await metadata.read()))
    meta = meta.set_index(metadata_index)

    # ---------------------------------------------------------
    # Step 1: Identify samples in selected cluster
    # ---------------------------------------------------------
    cluster_labels = np.array(cluster_labels)
    cluster_mask = (cluster_labels == cluster_id)
    sample_order = mod.index[cluster_mask]   # samples in this cluster

    # ---------------------------------------------------------
    # Step 2: Hypergeometric variables
    # ---------------------------------------------------------
    M = len(mod.index)                 # total samples
    N = len(sample_order)              # samples in selected cluster

    # n = number of samples matching metadata condition
    mask_n = pd.Series(True, index=meta.index)
    for col, val in selected_values.items():
        mask_n &= (meta[col] == val)
    n = mask_n.sum()

    # k = samples matching metadata condition inside the cluster
    meta_filtered = meta.loc[sample_order]
    mask_k = pd.Series(True, index=meta_filtered.index)
    for col, val in selected_values.items():
        mask_k &= (meta_filtered[col] == val)
    k = mask_k.sum()

    # ---------------------------------------------------------
    # Step 3: Compute p-value
    # ---------------------------------------------------------
    p_value = hypergeom.sf(k - 1, M, n, N)

    return JSONResponse({
    "cluster_id": int(cluster_id),
    "M_total_samples": int(M),
    "N_cluster_samples": int(N),
    "n_matching_metadata": int(n),
    "k_cluster_metadata_intersection": int(k),
    "p_value": float(p_value)
})

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

    # ----------------------------------------------------------
    # 1. Decode incoming JSON lists
    # ----------------------------------------------------------
    sample_order = json.loads(sample_order)
    module_leaf_order = [int(i) for i in json.loads(module_leaf_order)]
    module_cluster_labels = [int(i) for i in json.loads(module_cluster_labels)]
    annotation_cols = json.loads(annotation_cols)

    # ----------------------------------------------------------
    # 2. Load module usages
    # ----------------------------------------------------------
    df = pd.read_feather(io.BytesIO(await module_usages.read()))
    df.columns = ["Sample"] + list(df.columns[1:])
    df = df.set_index("Sample")

    # Reorder samples by preserved sample order
    df = df.loc[sample_order]
    df = df.T  # modules × samples

    # ----------------------------------------------------------
    # 3. Reorder modules by module_leaf_order
    # ----------------------------------------------------------
    df_reordered = df.iloc[module_leaf_order, :]

    # Also reorder module cluster labels to match the heatmap
    cluster_labels_ordered = [module_cluster_labels[i] for i in module_leaf_order]

    # ----------------------------------------------------------
    # 4. Load metadata for column annotations
    # ----------------------------------------------------------
    meta = pd.read_feather(io.BytesIO(await metadata.read()))
    meta = meta.set_index(metadata_index)
    meta = meta.loc[sample_order]   # same order as heatmap columns

    return module_annotated_heatmap(df_reordered, meta, annotation_cols, cluster_labels_ordered, module_leaf_order)

def fig_to_hex_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    fig.clf()
    buf.seek(0)
    return buf.read().hex()

@app.post("/heatmap_top_samples/")
async def heatmap_top_samples(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    annotation_cols: str = Form(""),
    metadata_index: str = Form(...)
):

    # -----------------------------------------------------
    # Load DataFrames
    # -----------------------------------------------------
    df_usage = pd.read_feather(io.BytesIO(await module_usages.read()))
    df_usage = df_usage.set_index(df_usage.columns[0])

    meta = pd.read_feather(io.BytesIO(await metadata.read()))
    meta = meta.set_index(metadata_index)

    annotation_cols_list = (
        [c for c in annotation_cols.split(",") if c.strip()]
        if annotation_cols else []
    )

    # -----------------------------------------------------
    # TOP MODULE ORDERING
    # -----------------------------------------------------
    sample_assignments = df_usage.idxmax(axis=1)
    df_usage["TopModule"] = sample_assignments

    def numeric_key(x):
        try:
            return int(x.split("_")[-1])
        except:
            return x

    modules_sorted = sorted(sample_assignments.unique(), key=numeric_key)

    ordered_samples = []
    for mod in modules_sorted:
        subset = df_usage[df_usage["TopModule"] == mod]
        subset = subset.sort_values(by=mod, ascending=False)
        ordered_samples.extend(subset.index.tolist())

    df_plot = df_usage.drop(columns=["TopModule"]).T
    df_plot = df_plot[ordered_samples]

    if len(annotation_cols_list) == 0:
        col_colors = None
        lut = {}
    else:
        col_colors, lut = build_annotations(meta.loc[ordered_samples], annotation_cols_list)

    buf = make_heatmap_png(df_plot, col_colors=col_colors, lut=lut)

    # return hex string instead of bytes
    return {
        "heatmap_png": buf.getvalue().hex(),
        "ordered_samples": ordered_samples,
    }

