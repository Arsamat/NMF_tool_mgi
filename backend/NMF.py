import os

# limit BLAS threads per worker to avoid deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import seaborn as sns
import matplotlib.pyplot as plt
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from scipy.stats import pearsonr
import time
import csv
from pybiomart import Dataset
import subprocess
import torch
from sklearn.utils.extmath import randomized_svd
import math
import tempfile


#function to transform ensembl ids to symbol ids
def transform_ids(df_link, gene_column):
    dataset = Dataset(name='hsapiens_gene_ensembl',
                      host='http://www.ensembl.org')
    delimiter = detect_delimiter(df_link)

    df = pd.read_csv(df_link, delimiter=delimiter)
    df = df.set_index(gene_column)

    # Get Ensembl → symbol mapping
    genes = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    mapping = dict(zip(genes['Gene stable ID'], genes['Gene name']))

    seen = set()
    result = []
    for val in df.index:
        mapped = mapping.get(val, val)
        if not mapped or str(mapped).lower() == "nan":
            mapped = val
        if mapped in seen:  # avoid duplicates
            mapped = val
        seen.add(mapped)
        result.append(mapped)

    df = df.copy()
    df.insert(0, "Geneid", result)  # put Geneid first
    df.reset_index(drop=True, inplace=True)
    print(df.columns)
    # Always save as CSV (overwrites input file path, regardless of extension)
    new_path = df_link.replace("txt", "csv")
    df.to_csv(new_path, sep=",", index=False)

    return df, new_path


#Function to determine which delimiter to use to read the file
def detect_delimiter(file_path):
    with open(file_path, 'r', newline='') as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            return None

def preprocess2(counts_link='250729_renamed_counts.txt',
                          metadata_link='Mapping.txt',
                          metadata_index="SampleName",
                          design_factor="Group",
                          hvg=5000,
                          gene_column='X',
                          symbols=False,
                          batch=False,
                          batch_column="",
                          batch_include=[]):
    
    if not symbols:
        counts, new_path = transform_ids(counts_link, gene_column)
    

    tmp_dir = tempfile.mkdtemp(prefix="preprocess_")
    out_dir = os.path.join(tmp_dir, "filtered_batched_counts.tsv")

    cmd = [
        "Rscript",
        "./filter_batch.R",  # path to your R script
        new_path,
        metadata_link,
        out_dir,
        design_factor,
        metadata_index,
        str(hvg),
        str(batch).upper(),
        batch_column,
        *map(str, batch_include)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("R script STDOUT:\n", result.stdout)
    print("R script STDERR:\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"R script failed:\n{result.stderr}")
    
    print("R filtering script finished successfully.")

    counts = pd.read_csv(out_dir, delimiter="\t")

    counts = counts.set_index('Geneid')

    counts = counts.T 
    #remove duplicates
    #counts = counts.loc[:, ~counts.columns.duplicated()].copy()
    #teanspose samples x genes
    #counts = counts.T 
    
    return counts, tmp_dir


def do_NMF(df_expr, 
           design_factor="Group",
            max_iter=1000, 
            n_components=7,
            seed = 42):
    
    start = time.time()
    # log2 transform
    if design_factor in list(df_expr.columns):
        Group = df_expr[design_factor].copy()
        df_expr = df_expr.drop(columns=[design_factor])
    device=None
    tol=1e-4
    init = "nndsvda"
    verbose=False

    df_expr = np.log2(df_expr + 1)
    
    nmf_start = time.time()

    X = torch.tensor(df_expr.values, dtype=torch.float32)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")
    X = X.to(device)

    n_samples, n_genes = X.shape

    # --- Initialization ---
    if init in ("nndsvd", "nndsvda"):
        # SVD decomposition
        U, S, Vt = randomized_svd(df_expr.values, n_components, random_state=seed)
        W_init = np.zeros((n_samples, n_components))
        H_init = np.zeros((n_components, n_genes))

        # First component
        W_init[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H_init[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])

        # Other components
        for j in range(1, n_components):
            x = U[:, j]
            y = Vt[j, :]
            xp, xn = np.maximum(x, 0), np.maximum(-x, 0)
            yp, yn = np.maximum(y, 0), np.maximum(-y, 0)

            xpnorm, ypnorm = np.linalg.norm(xp), np.linalg.norm(yp)
            xnnorm, ynnorm = np.linalg.norm(xn), np.linalg.norm(yn)

            mp, mn = xpnorm * ypnorm, xnnorm * ynnorm
            if mp > mn:
                u = xp / (xpnorm + 1e-8)
                v = yp / (ypnorm + 1e-8)
                sigma = mp
            else:
                u = xn / (xnnorm + 1e-8)
                v = yn / (ynnorm + 1e-8)
                sigma = mn

            W_init[:, j] = np.sqrt(S[j] * sigma) * u
            H_init[j, :] = np.sqrt(S[j] * sigma) * v

        if init == "nndsvda":
            avg = df_expr.values.mean()
            rng = np.random.default_rng(seed)  # reproducible randoms
            # replace zeros with small random positive values
            W_init[W_init == 0] = rng.random(np.sum(W_init == 0)) * avg / 100
            H_init[H_init == 0] = rng.random(np.sum(H_init == 0)) * avg / 100

        W = torch.tensor(W_init, dtype=torch.float32, device=device)
        H = torch.tensor(H_init, dtype=torch.float32, device=device)

    elif init == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        W = torch.rand((n_samples, n_components), dtype=torch.float32, device=device, generator=gen)
        H = torch.rand((n_components, n_genes), dtype=torch.float32, device=device, generator=gen)

    else:
        raise ValueError(f"Unknown init method: {init}")

    # --- Multiplicative updates ---
    errors = []
    converged = False
    for i in range(max_iter):
        WH = W @ H
        H *= (W.T @ X) / (W.T @ WH + 1e-8)
        WH = W @ H
        W *= (X @ H.T) / (W @ (H @ H.T) + 1e-8)

        # Track error
        if i % 10 == 0 or i == max_iter - 1:
            err = torch.norm(X - W @ H, p="fro").item()
            errors.append(err)
            if verbose:
                print(f"Iter {i}: error={err:.4f}")
            if len(errors) > 1 and abs(errors[-2] - errors[-1]) < tol:
                converged = True
                print(f"NMF converged at iteration {i} with error={err:.4f}")
                break

    if not converged:
        print(f"⚠️ NMF did not converge after {max_iter} iterations (last error={errors[-1]:.4f})")
    print("NMF execution time:", time.time() - nmf_start)

    # Convert back to pandas
    W = W.detach().cpu().numpy()
    H = H.detach().cpu().numpy()

    df_w = pd.DataFrame(W, index=df_expr.index,
                        columns=[f"Module_{i+1}" for i in range(n_components)])
    df_h = pd.DataFrame(H, index=[f"Module_{i+1}" for i in range(n_components)],
                        columns=df_expr.columns)

    return df_h, df_w, converged

#if __name__ == "__main__":
#    do_NMF(counts_link="250729_ensemblid_counts.tsv")