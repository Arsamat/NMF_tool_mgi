"""
DEG top-gene expression heatmap with metadata annotation bars.

Uses MongoDB-fetched sample metadata plus Group (GroupA/GroupB) assignment.
Plotting follows the same seaborn clustermap + glasbey pattern as heatmaps/heatmap_utils.py.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from heatmaps.plot_heatmap_helpers import build_annotations, make_heatmap_png

# Optional annotation / sort columns (after Group) when present in MongoDB.
METADATA_ANNOTATION_KEYS = ("CellType", "Treatment", "Genotype", "Timepoint")


def merge_mongo_only_metadata(
    sample_names: list[str],
    mongo_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    One row per sample, index = SampleName. Mongo fields only (no Group).
    Used for precomputed / browse-by-experiment heatmaps.
    """
    idx = pd.Index(sample_names, name="SampleName")
    if mongo_df is not None and not mongo_df.empty and "SampleName" in mongo_df.columns:
        sub = mongo_df.drop_duplicates(subset=["SampleName"], keep="first").set_index("SampleName")
        sub = sub.reindex(idx)
    else:
        sub = pd.DataFrame(index=idx)
    sub.index.name = "SampleName"
    return sub


def default_precomputed_annotation_columns(meta: pd.DataFrame) -> list[str]:
    """Annotation bar columns when Group is not defined (Mongo only)."""
    return [c for c in METADATA_ANNOTATION_KEYS if c in meta.columns]


def render_precomputed_heatmap_png_bytes(
    hm_df: pd.DataFrame,
    mongo_df: pd.DataFrame | None,
    annotation_cols: list[str] | None = None,
) -> bytes:
    """
    Genes × samples log-expression heatmap with metadata bars (no Group).
    """
    if hm_df.empty:
        raise ValueError("Heatmap matrix is empty")

    samples = [str(c) for c in hm_df.columns]
    hm_df = hm_df.copy()
    hm_df.columns = samples

    meta = merge_mongo_only_metadata(samples, mongo_df)
    if annotation_cols:
        cols = [c for c in annotation_cols if c in meta.columns]
    else:
        cols = default_precomputed_annotation_columns(meta)

    if not cols:
        buf = make_heatmap_png(hm_df, col_colors=None, lut=None)
        return buf.getvalue()

    hm_ord, meta_ord = sort_hm_by_annotation_columns(hm_df, meta, cols)
    col_colors, lut = build_annotations(meta_ord, cols, use_glasbey=True)
    buf = make_heatmap_png(
        hm_ord,
        col_colors=col_colors if not col_colors.empty else None,
        lut=lut or None,
    )
    return buf.getvalue()


def merge_deg_heatmap_metadata(
    sample_names: list[str],
    group_a: list[str],
    group_b: list[str],
    mongo_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    One row per sample, index = SampleName.
    Merges full Mongo metadata with a synthetic Group column from the DEG comparison.
    """
    gmap = {**{s: "GroupA" for s in group_a}, **{s: "GroupB" for s in group_b}}
    idx = pd.Index(sample_names, name="SampleName")

    if mongo_df is not None and not mongo_df.empty and "SampleName" in mongo_df.columns:
        sub = mongo_df.drop_duplicates(subset=["SampleName"], keep="first").set_index("SampleName")
        sub = sub.reindex(idx)
    else:
        sub = pd.DataFrame(index=idx)

    sub = sub.copy()
    sub["Group"] = [gmap.get(s) for s in sample_names]
    sub.index.name = "SampleName"
    return sub


def sort_hm_by_annotation_columns(
    hm_df: pd.DataFrame,
    meta: pd.DataFrame,
    annotation_cols: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    hm_df: genes × samples. meta: index = SampleName (same as hm_df.columns).
    Stable sort by annotation_cols (primary first). Returns reordered hm and meta rows.
    """
    samples = list(hm_df.columns)
    use_cols = [c for c in annotation_cols if c in meta.columns]
    if not use_cols:
        return hm_df.copy(), meta.reindex(samples)

    m = meta.reindex(samples)
    m = m.sort_values(by=use_cols, kind="stable", na_position="last")
    order = list(m.index)
    hm_out = hm_df[[c for c in order if c in hm_df.columns]]
    return hm_out, m.loc[hm_out.columns]


def render_deg_heatmap_png_bytes(
    hm_df: pd.DataFrame,
    group_a: list[str],
    group_b: list[str],
    mongo_df: pd.DataFrame | None,
    annotation_cols: list[str] | None = None,
) -> bytes:
    """
    Build a PNG (genes × samples, viridis) with optional column annotation bars.
    If annotation_cols is None, uses default columns available in merged metadata.
    """
    if hm_df.empty:
        raise ValueError("Heatmap matrix is empty")

    samples = [str(c) for c in hm_df.columns]
    hm_df = hm_df.copy()
    hm_df.columns = samples

    meta = merge_deg_heatmap_metadata(samples, group_a, group_b, mongo_df)
    if annotation_cols:
        cols = [c for c in annotation_cols if c in meta.columns]
    else:
        #cols = default_annotation_columns(meta)
        cols = ["Group"]
    hm_ord, meta_ord = sort_hm_by_annotation_columns(hm_df, meta, cols)

    col_colors, lut = build_annotations(meta_ord, cols, use_glasbey=True)
    buf = make_heatmap_png(hm_ord, col_colors=col_colors if not col_colors.empty else None, lut=lut or None)
    return buf.getvalue()
