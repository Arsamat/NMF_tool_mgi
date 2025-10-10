# cNMF.py
import os, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from cnmf import cNMF
from time import time

# Keep BLAS from oversubscribing when we spawn processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# -------- TOP-LEVEL WORKER (must not be nested) --------
def _factorize_worker(output_dir, run_name, worker_i, total_workers):
    w = cNMF(output_dir=output_dir, name=run_name)
    w.factorize(worker_i=worker_i, total_workers=total_workers)
    return worker_i

# -------- MAIN API (only these args) --------
def cNMF_consensus(df_expr, k=7, hvg=5000, max_iter=5000, design_factor="Group", out_dir="tmp"):
    # Internal defaults
    start = time()
    output_directory = os.path.join(out_dir, "example_CNMF")
    run_name = "cNMF"
    counts_tsv = os.path.join(out_dir, "filtered_counts.tsv")
    #total_workers = max(1, min(4, (os.cpu_count() or 1)))  # auto-pick up to 4
    total_workers = 16
    n_iter = 50
    seed = 14
    density_threshold = 0.2
    components = np.array([k])  # e.g., k-2..k+2

    # Prep counts file from df_expr
    df = df_expr.copy()
    if design_factor in df.columns:
        df = df.drop(columns=[design_factor])
    # df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # if (df.values < 0).any():
    #     df = df.clip(lower=0.0)
    df.to_csv(counts_tsv, sep="\t", index=True)

    # Setup output dirs
    os.makedirs(output_directory, exist_ok=True)
    tmp_dir = os.path.join(output_directory, run_name, "cnmf_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # Prepare once (parent)
    cnmf_obj = cNMF(output_dir=output_directory, name=run_name)
    cnmf_obj.prepare(
        counts_fn=counts_tsv,
        components=components,
        n_iter=n_iter,
        num_highvar_genes=hvg,
        max_NMF_iter=max_iter,
        seed=seed,
    )

    # Parallel factorize across processes
    with ProcessPoolExecutor(max_workers=total_workers) as ex:
        futures = [
            ex.submit(_factorize_worker, output_directory, run_name, i, total_workers)
            for i in range(total_workers)
        ]
        for f in as_completed(futures):
            f.result()  # raise if any worker failed

    # Stitch + consensus at chosen k
    cnmf_obj.combine()
    cnmf_obj.consensus(
        k=int(k),
        density_threshold=density_threshold,
        show_clustering=False,
        close_clustergram_fig=False,
    )
    end = time()
    print("Total execution time: ", end - start)
