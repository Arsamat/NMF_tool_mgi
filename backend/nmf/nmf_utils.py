import pandas as pd
from scipy.stats import hypergeom
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import io
from fastapi.responses import JSONResponse
import numpy as np
from fastapi.responses import StreamingResponse
import zipfile
from fastapi import FastAPI, UploadFile, Form, File, BackgroundTasks, Request
from fastapi.responses import FileResponse
import os
from nmf.NMF import do_NMF
import tempfile
from nmf.cNMF import cNMF_consensus
from nmf.save_heatmap_pdf_ordered import save_heatmap_pdf_ordered
from starlette.background import BackgroundTask
import shutil
from nmf.NMF import do_NMF, preprocess2
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
import time
import random
from nmf.group_associations import run_module_group_analysis
import boto3
from fastapi import HTTPException
from pybiomart import Dataset
import mygene

BUCKET = "nmf-tool-bucket"
REGION = "us-east-2"

s3 = boto3.client("s3", region_name=REGION)


async def run_regular_nmf_util(#preprocessed,
    job_id, 
    k,
    max_iter,
    design_factor):

    preprocessed_tmp = tempfile.mkdtemp(prefix="preprocess_")
    preprocessed_path = os.path.join(preprocessed_tmp, "preprocessed_counts.csv")
    s3_key = f"jobs/{job_id}/preprocessed_counts.csv"

    try:
        s3.download_file(BUCKET, s3_key, preprocessed_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not download counts from S3 (Bucket={BUCKET}, Key={s3_key}). Error: {e}"
        )
    #df_expr = pd.read_feather(io.BytesIO(await preprocessed.read()))
    df_expr = pd.read_csv(preprocessed_path)
    df_expr = df_expr.set_index("Unnamed: 0")
    print(df_expr)

    try:
        df_h, df_w, converged = do_NMF(
            n_components = k,
            df_expr = df_expr,
            max_iter = max_iter,
            design_factor = design_factor 
        )
        print(df_h.head())
        print(df_w.head())

        #df_buf = io.BytesIO()
        #df_w.to_feather(df_buf)
        #df_buf.seek(0)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
            h_buf = io.BytesIO()
            df_h.reset_index(names="module").to_feather(h_buf)
            zip_file.writestr("nmf_h.feather", h_buf.getvalue())

            w_buf = io.BytesIO()
            df_w.reset_index(names="sample").to_feather(w_buf)
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
        try:
            if preprocessed_tmp and os.path.isdir(preprocessed_tmp):
                shutil.rmtree(preprocessed_tmp, ignore_errors=True)
        except Exception:
            # avoid masking the main exception / response
            pass

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

async def run_cnmf_utils(
    #    preprocessed,
        metadata,
        k,
        hvg,
        max_iter,
        design_factor,
        metadata_index,
        job_id, 
        batch_correct,
        gene_column
):
    preprocessed_tmp = tempfile.mkdtemp(prefix="preprocess_")
    preprocessed_path = os.path.join(preprocessed_tmp, "preprocessed_counts.csv")
    s3_key = f"jobs/{job_id}/preprocessed_counts.csv"
    batch_vars = []
    if batch_correct != "":
        batch_vars = batch_correct.split(",")
    
    try:
        s3.download_file(BUCKET, s3_key, preprocessed_path)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Could not download counts from S3 (Bucket={BUCKET}, Key={s3_key}). Error: {e}"
        )
    
    #df_expr = pd.read_feather(io.BytesIO(await preprocessed.read()))
    df_expr = pd.read_csv(preprocessed_path)
    sample_size = len(df_expr.columns.tolist())
    print(sample_size)
    
    if gene_column in df_expr.columns:
       df_expr = df_expr.set_index(gene_column) 
    elif "Geneid" in df_expr.columns:
        df_expr = df_expr.set_index("Geneid")
    elif "gene_name" in df_expr.columns:   
        df_expr = df_expr.set_index("gene_name")
    elif "Unnamed: 0" in df_expr.columns:
        df_expr = df_expr.set_index("Unnamed: 0")   

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
            df=df_expr,
            metadata=metadata,
            metadata_index=metadata_index,
            max_iter=max_iter,
            design_factor=design_factor,
            out_dir=out_dir,
            batch_vars=batch_vars,
            single_cell=False
        )

        # unique temp file to avoid clashes
        pdf_path = os.path.join(out_dir, f"nmf_heatmap_{k}.pdf")
        #if sample_size < 500:
         #   save_heatmap_pdf_ordered(usages_path=usages_path, metadata=metadata, out_pdf = pdf_path, index=metadata_index)
        
        module_usage_path = os.path.join(out_dir, "cNMF", f"cNMF.usages.k_{k}.dt_0_2.consensus.txt")
        gene_zscore_path = os.path.join(out_dir, "cNMF", f"cNMF.gene_spectra_score.k_{k}.dt_0_2.txt")


        modules = pd.read_table(module_usage_path, nrows=20)
        print(modules)

        out_zip=os.path.join(out_dir, "cnmf_bundle.zip")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            #if sample_size < 500:
                # add the PDF by path
            #    z.write(pdf_path, arcname="nmf_heatmap.pdf")  # rename inside zip if you want
            # add your text files
            for p in [module_usage_path, gene_zscore_path]:
                if os.path.isfile(p):
                    z.write(p, arcname=os.path.basename(p))
        
        task = BackgroundTask(
            cleanup_after_send,
            files=[out_zip, pdf_path],
            dirs=[out_dir, preprocessed_tmp]
        )

        return FileResponse(
            out_zip,
            media_type="application/zip",
            filename=f"cnmf_k{k}_bundle.zip",
            background=task,   # <- Starlette BackgroundTask
        )
    finally:
        print("Done:")

#preprocessing function

async def preprocess_util(
    metadata,
    gene_column,
    metadata_index,
    design_factor,
    symbols,
    hvg,
    batch,
    batch_column,
    batch_include,
    job_id,
    single_cell
):
    tmp_dir = tempfile.mkdtemp(prefix="preprocess_")
    print(tmp_dir)

    # derive where the counts file lives in S3 from job_id
    s3_key = f"jobs/{job_id}/counts.csv"
    print(s3_key)

    # local temp paths on EC2
    counts_path = os.path.join(tmp_dir, "counts.csv")
    metadata_path = os.path.join(tmp_dir, metadata.filename)

    out_dir = None

    try:
        # 1) Download counts from S3 -> EC2 temp path
        try:
            s3.download_file(BUCKET, s3_key, counts_path)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Could not download counts from S3 (Bucket={BUCKET}, Key={s3_key}). Error: {e}"
            )

        # 2) Save uploaded metadata to disk (same as before)
        with open(metadata_path, "wb") as f:
            shutil.copyfileobj(metadata.file, f)

        batch_include_tmp = batch_include.split(",") if batch_include else []

        # 3) Run your preprocessing using local file paths
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

        csv_buf = io.StringIO()
        df_expr.to_csv(csv_buf, index=True)
        s3.put_object(
            Bucket=BUCKET,
            Key=f"jobs/{job_id}/preprocessed_counts.csv",
            Body=csv_buf.getvalue().encode("utf-8"),
            ContentType="text/csv"
        )

        df_return = df_expr.iloc[:15, : 15]

        # 4) Serialize DF to feather in-memory + stream back
        # df_return = df_expr.head()
        df_buf = io.BytesIO()
        df_return.reset_index().to_feather(df_buf)
        df_buf.seek(0)

        return StreamingResponse(
            df_buf,
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="preprocessed.feather"'}
        )

    finally:
        # Clean up EC2 local storage (do NOT delete from S3)
        try:
            if out_dir and os.path.isdir(out_dir):
                shutil.rmtree(out_dir, ignore_errors=True)

            for p in (counts_path, metadata_path):
                if p and os.path.exists(p):
                    os.remove(p)

            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            # avoid masking the main exception / response
            pass


def process_single_run(expr_path, meta_path, k, run, max_iter, design_factor, sample_column, tmp_dir):

    #samples = df_expr["SampleName"].copy()
    #expr = df_expr.drop(columns=["SampleName"])
    df_expr = pd.read_csv(expr_path)
    df_expr = df_expr.set_index("Unnamed: 0")
    df_meta = pd.read_csv(meta_path)
    meta = df_meta[[sample_column, design_factor]]
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


async def explore_k_utils(#preprocessed_feather,
    meta,
    job_id,
    k_min,
    k_max,
    max_iter,
    design_factor,
    sample_column,
    n_reps  # number of replicates per K
):
    tmp_dir = tempfile.mkdtemp(prefix="kmetrics_")
    expr_path = os.path.join(tmp_dir, "expr.feather")
    meta_path = os.path.join(tmp_dir, "meta.csv")
    start = time.time()
    s3_key = f"jobs/{job_id}/preprocessed_counts.csv"

    try:
        # Save expression + metadata once
        # 1) Download counts from S3 -> EC2 temp path
        try:
            s3.download_file(BUCKET, s3_key, expr_path)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Could not download counts from S3 (Bucket={BUCKET}, Key={s3_key}). Error: {e}"
            )
        
        #save metadata
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


def map_ensembl_to_symbol(ensembl_ids):
    mg = mygene.MyGeneInfo()

    res = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
        returnall=False,
        verbose=False,
    )

    # res index = queried id, but may contain duplicates.
    # Keep the first non-null symbol per queried id.
    if isinstance(res, pd.DataFrame):
        sym = res["symbol"]
    else:
        # just in case mygene returns something unexpected
        sym = pd.Series(res)

    sym = sym.dropna()

    # ✅ enforce unique index (critical)
    sym = sym[~sym.index.duplicated(keep="first")]

    return sym


def transform_ids(df, gene_column):
    # work on just the gene ids (Series), not the whole matrix
    df = df.reset_index()
    if "index" in df.columns:
        gene_column = "index"
    gene_ids = df[gene_column].astype(str)
    

    # common in scRNA: ENSG... with version like ENSG000001.12
    gene_ids_nover = gene_ids.str.replace(r"\.\d+$", "", regex=True)

    # IMPORTANT: query only the unique IDs you actually have (avoid downloading full biomart table)
    uniq = pd.Index(gene_ids_nover.unique())

    sym = map_ensembl_to_symbol(uniq.tolist())

    mapped = gene_ids_nover.map(sym)
    mapped = mapped.where(mapped.notna() & (mapped.astype(str) != ""), gene_ids_nover)

    
    # enforce uniqueness like your "seen" logic:
    # if a symbol repeats, revert those repeats back to the original Ensembl ID
    dup_mask = mapped.duplicated(keep="first")
    mapped = mapped.where(~dup_mask, gene_ids_nover)

    # drop old gene column BEFORE transpose
    df = df.drop(columns=[gene_column])

    df = df.T

    df.columns = mapped.values

    return df


async def single_cell_util(
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
):
    batch_vars = []
    if batch_correct != "":
        batch_vars = batch_correct.split(",")

    tmp_dir = tempfile.mkdtemp(prefix="preprocess_")
    print(tmp_dir)

    # derive where the counts file lives in S3 from job_id
    s3_key = f"jobs/{job_id}/counts.csv"
    print(s3_key)

    # local temp paths on EC2
    counts_path = os.path.join(tmp_dir, "counts.csv")

    # 1) Download counts from S3 -> EC2 temp path
    try:
        s3.download_file(BUCKET, s3_key, counts_path)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Could not download counts from S3 (Bucket={BUCKET}, Key={s3_key}). Error: {e}"
        )
    
    min_umis_per_cell = 1000
    gene_detect_frac_den = 500

    counts = pd.read_csv(counts_path)
    counts = counts.set_index(gene_column)
    
    #print(columns[1:100])
    umi_per_cell = counts.sum(axis=0)  # sum over genes -> per cell
    #print(umi_per_cell)
    keep_cells = umi_per_cell >= min_umis_per_cell
    counts_f1 = counts.loc[:, keep_cells]

    # 2) Filter genes by detection in >= ceil(N/500) cells
    n_cells = counts_f1.shape[1]
    min_cells_detected = int(np.ceil(n_cells / gene_detect_frac_den))

    detected_cells_per_gene = (counts_f1 > 0).sum(axis=1)
    keep_genes = detected_cells_per_gene >= min_cells_detected
    df_expr = counts_f1.loc[keep_genes, :]
   # df_expr = counts_f2.T

    if not gene_symbols:
        df_expr = transform_ids(df_expr, gene_column)
    print(df_expr.columns)

    if gene_column in df_expr.columns:
       df_expr = df_expr.set_index(gene_column) 
    elif "Geneid" in df_expr.columns:
        df_expr = df_expr.set_index("Geneid")
    elif "gene_name" in df_expr.columns:   
        df_expr = df_expr.set_index("gene_name")
    elif "Unnamed: 0" in df_expr.columns:
        df_expr = df_expr.set_index("Unnamed: 0")   

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
            df=df_expr,
            metadata=metadata,
            metadata_index=metadata_index,
            #max_iter=max_iter,
            design_factor=design_factor,
            out_dir=out_dir,
            batch_vars=batch_vars,
            single_cell=True
        )

        # unique temp file to avoid clashes
        pdf_path = os.path.join(out_dir, f"nmf_heatmap_{k}.pdf")
        #if sample_size < 500:
         #   save_heatmap_pdf_ordered(usages_path=usages_path, metadata=metadata, out_pdf = pdf_path, index=metadata_index)
         
        module_usage_path = os.path.join(out_dir, "BatchCorrected", f"BatchCorrected.usages.k_{k}.dt_0_2.consensus.txt")
        gene_zscore_path  = os.path.join(out_dir, "BatchCorrected", f"BatchCorrected.gene_spectra_score.k_{k}.dt_0_2.txt")

        modules = pd.read_table(module_usage_path, nrows=20)
        print(modules)

        out_zip=os.path.join(out_dir, "cnmf_bundle.zip")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            #if sample_size < 500:
                # add the PDF by path
            #    z.write(pdf_path, arcname="nmf_heatmap.pdf")  # rename inside zip if you want
            # add your text files
            for p in [module_usage_path, gene_zscore_path]:
                if os.path.isfile(p):
                    z.write(p, arcname=os.path.basename(p))
        
        task = BackgroundTask(
            cleanup_after_send,
            files=[out_zip, pdf_path],
            dirs=[out_dir, tmp_dir]
        )

        return FileResponse(
            out_zip,
            media_type="application/zip",
            filename=f"cnmf_k{k}_bundle.zip",
            background=task,   # <- Starlette BackgroundTask
        )
    finally:
        print("Done:")


