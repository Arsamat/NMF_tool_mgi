import io
import traceback

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from nmf.spearman import run_spearman
from nmf.nmf_utils import (
    explore_k_utils,
    preprocess_util,
    run_cnmf_utils,
    run_regular_nmf_util,
    single_cell_util,
)

router = APIRouter(tags=["nmf"])


@router.post("/preprocess_df")
async def preprocess_data(
    metadata: UploadFile,
    gene_column: str = Form(...),
    metadata_index: str = Form(...),
    design_factor: str = Form(...),
    symbols: bool = Form(False),
    mouse: bool = Form(False),
    hvg: int = Form(5000),
    batch: bool = Form(False),
    batch_column: str = Form(""),
    batch_include: str = Form(""),
    job_id: str = Form(...),
    single_cell: bool = Form(False),
):
    print(single_cell)
    return await preprocess_util(
        metadata,
        gene_column,
        metadata_index,
        design_factor,
        symbols,
        mouse,
        hvg,
        batch,
        batch_column,
        batch_include,
        job_id,
        single_cell,
    )


@router.post("/explore_correlations")
async def explore_corr(gene_loadings: UploadFile):
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
            content={"error": str(e), "traceback": tb},
        )


@router.post("/k_metrics")
async def find_k_metrics(
    meta: bytes = File(...),
    job_id: str = Form(...),
    k_min: int = Form(...),
    k_max: int = Form(...),
    max_iter: int = Form(...),
    design_factor: str = Form(...),
    sample_column: str = Form(...),
    n_reps: int = 1,
):
    return await explore_k_utils(
        meta,
        job_id,
        k_min,
        k_max,
        max_iter,
        design_factor,
        sample_column,
        n_reps,
    )


@router.post("/run_regular_nmf")
async def run_nmf_files(
    job_id: str = Form(...),
    k: int = Form(...),
    max_iter: int = Form(...),
    design_factor: str = Form(...),
):
    return await run_regular_nmf_util(
        job_id,
        k,
        max_iter,
        design_factor,
    )


@router.post("/run_nmf_files")
async def run_cnmf_files(
    metadata: UploadFile,
    k: int = Form(...),
    hvg: int = Form(...),
    max_iter: int = Form(...),
    design_factor: str = Form(...),
    metadata_index: str = Form(...),
    job_id: str = Form(...),
    batch_correct: str = Form(""),
    gene_column: str = Form(...),
):
    return await run_cnmf_utils(
        metadata,
        k,
        hvg,
        max_iter,
        design_factor,
        metadata_index,
        job_id,
        batch_correct,
        gene_column,
    )


@router.post("/cnmf_single_cell")
async def run_cnmf_single_cell(
    metadata: UploadFile,
    k: int = Form(...),
    hvg: int = Form(...),
    design_factor: str = Form(...),
    metadata_index: str = Form(...),
    job_id: str = Form(...),
    batch_correct: str = Form(""),
    gene_column: str = Form(...),
    gene_symbols: bool = Form(True),
):
    return await single_cell_util(
        metadata,
        k,
        hvg,
        design_factor,
        metadata_index,
        job_id,
        batch_correct,
        gene_column,
        gene_symbols,
    )
