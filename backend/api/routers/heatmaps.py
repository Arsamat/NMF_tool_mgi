import io
import json

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile

from heatmaps.plot_heatmap_helpers import dendogram_modules
from heatmaps.heatmap_utils import (
    annotated_heatmap_util,
    cluster_samples_utils,
    heatmap_top_samples_utils,
    hypergeom_endpoint_utils,
    module_heatmap_utils,
    plot_expression_heatmap,
    preview_wide_heatmap_inline,
)

router = APIRouter(tags=["heatmaps"])


@router.post("/initial_heatmap_preview/")
async def initial_heatmap_preview(
    df: UploadFile,
    metadata: UploadFile,
    metadata_index: str = Form(...),
    annotation_cols: str = Form(...),
    average_groups: str = Form(...),
):
    """Returns a PNG heatmap computed fully in-memory."""
    df_bytes = await df.read()
    meta_bytes = await metadata.read()

    df_loaded = pd.read_feather(io.BytesIO(df_bytes))
    df_loaded = df_loaded.set_index("index")
    meta = pd.read_feather(io.BytesIO(meta_bytes))
    annotation_cols_parsed = json.loads(annotation_cols)

    average_groups_flag = average_groups.lower() == "true"

    return await preview_wide_heatmap_inline(
        df_loaded,
        meta,
        metadata_index,
        annotation_cols_parsed,
        average_groups_flag,
    )


@router.post("/plot_heatmap/")
async def plot_heatmap_expression(
    gene_loadings: UploadFile = File(...),
    job_id: str = Form(...),
    metadata: UploadFile = File(...),
    annotation_cols: str = Form(...),
    metadata_index: str = Form(...),
    X: int = Form(10),
):
    """Returns a PNG heatmap computed fully in-memory."""
    return await plot_expression_heatmap(
        gene_loadings,
        job_id,
        metadata,
        annotation_cols,
        metadata_index,
        X,
    )


@router.post("/cluster_samples/")
async def cluster_samples(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    k: int = Form(3),
):
    return await cluster_samples_utils(
        module_usages,
        metadata,
        metadata_index,
        k,
    )


@router.post("/annotated_heatmap/")
async def annotated_heatmap(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    leaf_order: str = Form(...),
    annotation_cols: str = Form("[]"),
    cluster_labels: str = Form("[]"),
):
    return await annotated_heatmap_util(
        module_usages,
        metadata,
        metadata_index,
        leaf_order,
        annotation_cols,
        cluster_labels,
    )


@router.post("/hypergeom/")
async def hypergeom_endpoint(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    cluster_labels: str = Form(...),
    cluster_id: int = Form(...),
    selected_values: str = Form(...),
):
    return await hypergeom_endpoint_utils(
        module_usages,
        metadata,
        metadata_index,
        cluster_labels,
        cluster_id,
        selected_values,
    )


@router.post("/cluster_modules/")
async def cluster_modules(
    module_usages: UploadFile = File(...),
    sample_order: str = Form(...),
    n_clusters: int = Form(...),
):
    sample_order_list = json.loads(sample_order)
    df = pd.read_feather(io.BytesIO(await module_usages.read()))
    df.columns = ["Sample"] + list(df.columns[1:])
    df = df.set_index("Sample")
    df = df.loc[sample_order_list]
    df = df.T

    return dendogram_modules(df, n_clusters)


@router.post("/module_heatmap/")
async def module_heatmap(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    metadata_index: str = Form(...),
    sample_order: str = Form(...),
    module_leaf_order: str = Form(...),
    module_cluster_labels: str = Form(...),
    annotation_cols: str = Form("[]"),
):
    return await module_heatmap_utils(
        module_usages,
        metadata,
        metadata_index,
        sample_order,
        module_leaf_order,
        module_cluster_labels,
        annotation_cols,
    )


@router.post("/heatmap_top_samples/")
async def heatmap_top_samples(
    module_usages: UploadFile = File(...),
    metadata: UploadFile = File(...),
    annotation_cols: str = Form(""),
    metadata_index: str = Form(...),
):
    return await heatmap_top_samples_utils(
        module_usages,
        metadata,
        annotation_cols,
        metadata_index,
    )
