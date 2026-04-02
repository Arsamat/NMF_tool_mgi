import os
import subprocess
import zipfile
from pathlib import Path

import pathway as pathway_pkg
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse

from pathway.gene_loadings_gpt import gpt_utils

_PATHWAY_DIR = Path(pathway_pkg.__file__).resolve().parent

router = APIRouter(tags=["pathway"])


@router.post("/process_gene_loadings/")
async def process_gene_loadings(
    file: UploadFile,
    top_n: int = Form(2),
):
    return await gpt_utils(file, top_n)


@router.post("/run_pathview/")
async def run_pathview(
    file: UploadFile = File(...),
    gene_format: str = Form(None),
):
    input_path = f"/tmp/{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    out_root = "/tmp/pathview_output"
    os.makedirs(out_root, exist_ok=True)
    print("Input path:", input_path)
    print("Output dir:", out_root)
    gf = gene_format if gene_format is not None else ""
    print("Gene format:", gf)
    result = subprocess.run(
        ["Rscript", str(_PATHWAY_DIR / "pathway_analysis.R"), input_path, out_root, gf],
        check=True,
        cwd=str(_PATHWAY_DIR),
    )
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)

    zip_path = "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(out_root + "/pathway_results.zip", arcname="pathway_results.zip")
        zipf.write(out_root + "/kegg_dataframe.csv", arcname="kegg_dataframe.csv")
    return FileResponse(zip_path, media_type="application/zip", filename="bundle.zip")
