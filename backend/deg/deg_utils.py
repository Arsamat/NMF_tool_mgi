"""
Run two-group differential expression analysis via R (edgeR/limma).
Fetches counts from S3, runs deg_analysis.R, returns StreamingResponse (ZIP containing
feather + optional heatmap and GSEA CSVs).
"""
import io
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from fastapi.responses import StreamingResponse

from infra.db_utils import get_counts_subset, get_run_metadata


def run_deg_analysis(group_a: list[str], group_b: list[str]) -> StreamingResponse:
    """
    Run DEG analysis for group_a vs group_b.
    group_a, group_b: lists of sample names (SampleName from metadata).
    Returns: StreamingResponse with ZIP (deg_analysis.feather + optional heatmap + GSEA).
    """
    if not group_a or not group_b:
        raise ValueError("Both group_a and group_b must contain at least one sample")

    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError(
            "DEG analysis requires at least 2 samples per group. "
            f"Group A: {len(group_a)}, Group B: {len(group_b)}. Add more samples to each group."
        )

    sample_names = list(group_a) + list(group_b)
    counts_df = get_counts_subset(sample_names)
    batch_metadata = get_run_metadata(sample_names)

    if counts_df is None or (isinstance(counts_df, dict) and not counts_df):
        raise ValueError("No count data found for the specified samples")

    if isinstance(counts_df, dict):
        counts_df = pd.DataFrame(counts_df)

    meta_rows = []
    for s in group_a:
        meta_rows.append({"SampleName": s, "Group": "GroupA", "Run": batch_metadata.loc[batch_metadata["SampleName"] == s, "Run"].values[0]})
    for s in group_b:
        meta_rows.append({"SampleName": s, "Group": "GroupB", "Run": batch_metadata.loc[batch_metadata["SampleName"] == s, "Run"].values[0]})
    metadata_df = pd.DataFrame(meta_rows)

    gene_col = "Geneid"

    available_samples = [c for c in counts_df.columns if c != gene_col]
    requested = set(metadata_df["SampleName"])
    missing = requested - set(available_samples)
    if missing:
        raise ValueError(f"Samples not found in counts: {missing}")

    count_cols = [gene_col] + list(metadata_df["SampleName"])
    counts_subset = counts_df[[c for c in count_cols if c in counts_df.columns]].copy()
    counts_subset = counts_subset.rename(columns={gene_col: "gene"})

    script_dir = Path(__file__).resolve().parent
    r_script = script_dir / "deg_analysis.R"
    if not r_script.is_file():
        raise FileNotFoundError(f"DEG R script not found: {r_script}")

    with tempfile.TemporaryDirectory() as tmpdir:
        count_path = os.path.join(tmpdir, "counts.csv")
        meta_path = os.path.join(tmpdir, "metadata.csv")
        out_path = os.path.join(tmpdir, "deg_results.csv")

        counts_subset.to_csv(count_path, index=False)
        metadata_df.to_csv(meta_path, index=False)

        r_args = [str(r_script), count_path, meta_path, out_path, "human"]

        result = subprocess.run(
            r_args,
            cwd=str(script_dir),
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError("DEG analysis failed. Check EC2 console for R output.")

        if not os.path.isfile(out_path):
            raise RuntimeError("DEG script did not produce output file")

        deg_df = pd.read_csv(out_path)
        print(deg_df.head())
        heatmap_path = os.path.join(tmpdir, "heatmap_matrix.csv")
        heatmap_anno_path = os.path.join(tmpdir, "heatmap_annotation.csv")
        gsea_path = os.path.join(tmpdir, "gsea_results.csv")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            df_buf = io.BytesIO()
            deg_df.to_feather(df_buf)
            df_buf.seek(0)
            zf.writestr("deg_analysis.feather", df_buf.getvalue())
            if os.path.isfile(heatmap_path):
                with open(heatmap_path, "rb") as f:
                    zf.writestr("heatmap_matrix.csv", f.read())
            if os.path.isfile(heatmap_anno_path):
                with open(heatmap_anno_path, "rb") as f:
                    zf.writestr("heatmap_annotation.csv", f.read())
            if os.path.isfile(gsea_path):
                with open(gsea_path, "rb") as f:
                    zf.writestr("gsea_results.csv", f.read())
        zip_buf.seek(0)
        return StreamingResponse(
            iter([zip_buf.getvalue()]),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="deg_results.zip"'},
        )
