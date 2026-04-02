# cNMF.py
import os, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
#from cnmf import cNMF
from time import time
#from cnmf import cNMF, Preprocess
from cnmf import cNMF
from cnmf import Preprocess
import scanpy as sc
import scipy.sparse as sp
import anndata as ad

# Keep BLAS from oversubscribing when we spawn processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# -------- TOP-LEVEL WORKER (must not be nested) --------
def _factorize_worker(w, output_dir, run_name, worker_i, total_workers):
   # w = cNMF(output_dir=output_dir, name=run_name)
    w.factorize(worker_i=worker_i, total_workers=total_workers)
    return worker_i

def convert_ann_obj(df, metadata, metadata_index, batch_vars):
    metadata = metadata.reindex(df.index)
    #metadata = metadata.set_index(metadata_index)
    to_keep = [metadata_index]
    for var in batch_vars:
        to_keep.append(var)

    metadata = metadata[to_keep]

    var_df = pd.DataFrame(index=df.columns)
    var_df.index.name = "gene"

    # ----------------------------
    # 4) Create AnnData
    # ----------------------------
    # If expr_df is dense and large, you may want sparse to save RAM:
    X = sp.csr_matrix(df.values)

    # Optional: convert to sparse if many zeros
    # X = sparse.csr_matrix(X)

    adata = ad.AnnData(
        X=X,
        obs=metadata.copy(),
        var=var_df
    )

    for val in to_keep:
        adata.obs[val] = adata.obs[val].astype(str).astype("category")

    # adata.obs["sample"] = adata.obs["sample"].astype(str).astype("category")
    # adata.obs["Cell_ID"] = adata.obs["Cell_ID"].astype(str).astype("category")
    print(adata)
    return adata


# -------- MAIN API (only these args) --------
def cNMF_consensus(df, metadata, metadata_index, k=7, hvg=5000, max_iter=5000, design_factor="Group", out_dir="tmp", batch_vars=[], single_cell=False):
    # Internal defaults
    start = time()
    #output_dir = os.path.join(out_dir, "example_CNMF")
    output_dir = out_dir
    run_name = "cNMF"
    counts_tsv = os.path.join(out_dir, "filtered_counts.tsv")
    #total_workers = max(1, min(4, (os.cpu_count() or 1)))  # auto-pick up to 4
    total_workers = 16
    n_iter = 50
    seed = 14
    density_threshold = 0.2
    components = np.array([k])  # e.g., k-2..k+2
    print(batch_vars)

    # Prep counts file from df_expr
    #df = df_expr.copy()
    if design_factor in df.columns:
        df = df.drop(columns=[design_factor])
    # df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # if (df.values < 0).any():
    #     df = df.clip(lower=0.0)
    #df.to_csv(counts_tsv, sep="\t", index=True)

    # Setup output dirs
    # tmp_dir = os.path.join(output_directory, run_name, "cnmf_tmp")

    # if os.path.exists(tmp_dir):
    #     shutil.rmtree(tmp_dir)

    # os.makedirs(tmp_dir, exist_ok=True)
    save_base = os.path.join(output_dir, "batchcorrect")
    os.makedirs(os.path.dirname(save_base), exist_ok=True)    
    print(single_cell)  
    
    if single_cell:
        ann_obj = convert_ann_obj(df, metadata, metadata_index, batch_vars)
        p = Preprocess(random_seed=14)

        #Batch correct the data and save the corrected high-variance gene data to adata_c, and the TPM normalized data to adata_tpm 
        (adata_c, adata_tpm, hvgs) = p.preprocess_for_cnmf(ann_obj, harmony_vars=batch_vars, makeplots=False, n_top_rna_genes = 2000, librarysize_targetsum= 1e6,
                                                            save_output_base=output_dir + '/batchcorrect')

        #Then run cNMF passing in the corrected counts file, tpm_fn, and HVGs as inputs
        cnmf_obj = cNMF(output_dir=output_dir, name='BatchCorrected')
        cnmf_obj.prepare(counts_fn=output_dir + '/batchcorrect.Corrected.HVG.Varnorm.h5ad',
                                tpm_fn=output_dir + '/batchcorrect.TP10K.h5ad',
                                genes_file=output_dir + '/batchcorrect.Corrected.HVGs.txt',
                                components=components, n_iter=20, seed=14, num_highvar_genes=hvg)

        cnmf_obj.factorize_multi_process(16)

        # Stitch + consensus at chosen k
        cnmf_obj.combine()
        cnmf_obj.consensus(
            k=int(k),
            density_threshold=density_threshold,
            show_clustering=False,
            close_clustergram_fig=False,
        )
    else:
        #Prepare once (parent)
        df.to_csv(counts_tsv, sep="\t", index=True)
        cnmf_obj = cNMF(output_dir=output_dir, name=run_name)
        # cnmf_obj.prepare(
        #     counts_fn=counts_tsv,
        #     components=components,
        #     n_iter=n_iter,
        #     num_highvar_genes=hvg,
        #     max_NMF_iter=max_iter,
        #     seed=seed,
        # )

        cnmf_obj.prepare_from_preprocessed(
            X=df,
            tpm=df,                     # if you don't have TPM, this is fine
            components=[k],
            n_iter=50,
            seed=0,
            max_NMF_iter=max_iter
        )

        # Parallel factorize across processes
        # with ProcessPoolExecutor(max_workers=total_workers) as ex:
        #     futures = [
        #         ex.submit(_factorize_worker, cnmf_obj, output_directory, run_name, i, total_workers)
        #         for i in range(total_workers)
        #     ]
        #     for f in as_completed(futures):
        #         f.result()  # raise if any worker failed
        cnmf_obj.factorize_multi_process(16)

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
