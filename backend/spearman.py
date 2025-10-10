# spearman_engine.py
import numpy as np
import pandas as pd
from NMF import do_NMF

def spearman_pairs_from_H(df_h: pd.DataFrame, use_abs: bool = False):
    """
    Return a list[dict] of pairwise Spearman correlations between modules for one k.
    df_h: rows=modules, cols=genes (NMF H)
    """
    corr = df_h.T.corr(method="spearman")  # modules x modules
    mods = list(corr.index)
    rows = []
    # upper triangle only (i<j)
    for i in range(len(mods)):
        for j in range(i+1, len(mods)):
            rho = float(corr.iat[i, j])
            rows.append({
                "module_i": mods[i],
                "module_j": mods[j],
                "rho": rho,
                "abs_rho": abs(rho)
            })
    # Mark the max pair (by abs_rho if use_abs, else by rho)
    if rows:
        if use_abs:
            idx = int(np.argmax([r["abs_rho"] for r in rows]))
        else:
            idx = int(np.argmax([r["rho"] for r in rows]))
        rows[idx]["is_max_for_k"] = True
    # default False for others
    for r in rows:
        r.setdefault("is_max_for_k", False)
    return rows

def _pairwise_upper_triangle(corr: pd.DataFrame) -> np.ndarray:
    """All off-diagonal, upper-triangle correlation values as 1D array."""
    n = corr.shape[0]
    if n < 2:
        return np.array([], dtype=float)
    idx = np.triu_indices(n, k=1)
    return corr.values[idx]

def spearman_values_from_H(df_h: pd.DataFrame, use_abs: bool = False) -> np.ndarray:
    """
    H is (modules x genes). We want module–module Spearman across genes,
    so transpose to make modules=columns.
    """
    corr = df_h.T.corr(method="spearman")      # (modules x modules)
    vals = _pairwise_upper_triangle(corr)      # 1D array of pairwise values
    if use_abs:
        vals = np.abs(vals)
    return vals

def run_spearman(
    df_expr: pd.DataFrame,
    k,
    design_factor: str,
):
    """
    Loops k in [k_min, k_max], runs NMF -> H, computes Spearman correlations,
    returns:
      - max_map: {k: max correlation}
      - corr_by_k: {k: [all off-diagonal correlations]}
    """
   # max_map: dict[int, float] = {}
   # corr_by_k: dict[int, list[float]] = {}

    pairs_by_k = {}
    max_iter = 5000

    
    df_H, _ = do_NMF(
        n_components=k,
        max_iter=max_iter,
        design_factor=design_factor,
        df_expr=df_expr
    )

    #vals = spearman_values_from_H(df_H)   # np.ndarray
    #corr_by_k[k] = [float(x) for x in vals]                # JSON-safe
    #max_map[k] = float(np.max(vals)) if vals.size else float("nan")

    pairs = spearman_pairs_from_H(df_H)
    pairs_by_k[k] = pairs

    return pairs_by_k

    