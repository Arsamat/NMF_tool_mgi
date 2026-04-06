"""
Top-gene expression heatmap for precomputed DEG results (browse-by-experiment).

Uses counts from S3 parquet, MongoDB metadata for annotation bars (CellType, Treatment,
Genotype, Timepoint — no Group). Genes: top 30 by ascending P-value from the uploaded DEG table.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from deg.deg_heatmap import render_precomputed_heatmap_png_bytes
from infra.db_utils import get_counts_subset, get_metadata_for_samples


def _p_value_column(deg_df: pd.DataFrame) -> Optional[str]:
    for c in ("P.Value", "PValue", "pvalue", "P_value", "p.value"):
        if c in deg_df.columns:
            return c
    if "adj.P.Val" in deg_df.columns:
        return "adj.P.Val"
    if "padj" in deg_df.columns:
        return "padj"
    return None


def _gene_id_column(deg_df: pd.DataFrame) -> Optional[str]:
    for c in ("gene", "Gene", "ensembl_gene_id", "gene_id", "Geneid", "ENSEMBL", "ensembl"):
        if c in deg_df.columns:
            return c
    return None


def _symbol_column(deg_df: pd.DataFrame) -> Optional[str]:
    for c in ("SYMBOL", "gene_symbol", "symbol", "gene_name", "Gene.name"):
        if c in deg_df.columns:
            return c
    return None


def _norm_ensembl(x) -> str:
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    return s.split(".")[0]


def top_n_deg_genes_for_heatmap(deg_df: pd.DataFrame, n: int = 30) -> tuple[list[str], list[str]]:
    """
    Return (ensembl_ids_for_counts, row_labels_for_display) for up to n rows,
    ordered by significance then original order.
    """
    pcol = _p_value_column(deg_df)
    gcol = _gene_id_column(deg_df)
    if not pcol or not gcol:
        raise ValueError(
            "DEG table must contain a gene id column (e.g. gene, ensembl_gene_id) and a p-value column (e.g. P.Value)."
        )
    sym_col = _symbol_column(deg_df)

    work = deg_df.copy()
    work["_p"] = pd.to_numeric(work[pcol], errors="coerce")
    work = work.dropna(subset=["_p"]).sort_values("_p", kind="stable")
    work = work.head(n)

    ids: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    for _, row in work.iterrows():
        gid = _norm_ensembl(row[gcol])
        if not gid or gid in seen:
            continue
        seen.add(gid)
        ids.append(gid)
        if sym_col and pd.notna(row.get(sym_col)) and str(row[sym_col]).strip():
            lab = str(row[sym_col]).strip()
        else:
            lab = gid
        labels.append(lab if lab else gid)

    if not ids:
        raise ValueError("No valid gene ids found in DEG table for heatmap.")
    return ids, labels


def _log2_cpm(counts_mat: pd.DataFrame) -> pd.DataFrame:
    """Samples = columns, genes = rows. Simple log2(CPM+1) using column totals."""
    mat = counts_mat.astype(float)
    lib = mat.sum(axis=0).replace(0, np.nan)
    cpm = mat.div(lib, axis=1) * 1e6
    return np.log2(cpm + 1.0)


def _cluster_rows(mat: pd.DataFrame) -> pd.DataFrame:
    if mat.shape[0] < 2:
        return mat
    dist = pdist(mat.values, metric="euclidean")
    Z = linkage(dist, method="complete")
    order = leaves_list(Z)
    return mat.iloc[order]


def build_precomputed_deg_heatmap_png_bytes(
    deg_df: pd.DataFrame,
    sample_names: list[str],
    annotation_cols: list[str] | None = None,
    top_n: int = 30,
) -> bytes:
    """
    Build log2-CPM heatmap for top_n genes (by p-value) × samples, Mongo annotations, PNG bytes.
    """
    samples = [str(s).strip() for s in sample_names if str(s).strip()]
    if len(samples) < 2:
        raise ValueError("Need at least two sample names for a heatmap.")

    gene_ids, row_labels = top_n_deg_genes_for_heatmap(deg_df, n=top_n)

    counts_df = get_counts_subset(samples)
    if counts_df is None or counts_df.empty:
        raise ValueError("No count data found for the specified samples.")

    if "Geneid" not in counts_df.columns:
        raise ValueError("Counts table missing Geneid column.")

    counts_df = counts_df.set_index("Geneid")
    index_norm = {_norm_ensembl(i): i for i in counts_df.index.astype(str)}
    resolved: list[str] = []
    label_by_raw: list[tuple[str, str]] = []
    seen_raw: set[str] = set()
    for gid, lab in zip(gene_ids, row_labels):
        key = _norm_ensembl(gid)
        raw = index_norm.get(key)
        if raw is not None and raw not in seen_raw:
            seen_raw.add(raw)
            resolved.append(raw)
            label_by_raw.append((raw, lab))

    if not resolved:
        raise ValueError(
            "None of the top DEG genes matched Geneid in the counts matrix. "
            "Check that gene ids in the DE file match Ensembl ids in counts."
        )

    missing_cols = [c for c in samples if c not in counts_df.columns]
    if missing_cols:
        raise ValueError(f"Samples missing from counts data: {missing_cols[:5]}…")

    sub = counts_df.loc[resolved, samples]
    log_cpm = _log2_cpm(sub)

    seen_labs: dict[str, int] = {}
    display_labels: list[str] = []
    for raw, lab in label_by_raw:
        base = lab if lab else str(raw)
        n = seen_labs.get(base, 0)
        seen_labs[base] = n + 1
        display_labels.append(base if n == 0 else f"{base} ({n + 1})")
    log_cpm.index = display_labels

    log_cpm = _cluster_rows(log_cpm)

    mongo_meta = get_metadata_for_samples(samples)
    return render_precomputed_heatmap_png_bytes(log_cpm, mongo_meta, annotation_cols=annotation_cols)


def build_precomputed_deg_heatmap_png_bytes_from_csv_bytes(
    deg_csv_bytes: bytes,
    sample_names: list[str],
    annotation_cols: list[str] | None = None,
    top_n: int = 30,
) -> bytes:
    deg_df = pd.read_csv(io.BytesIO(deg_csv_bytes))
    return build_precomputed_deg_heatmap_png_bytes(deg_df, sample_names, annotation_cols, top_n)
