def flatten_payload(payload: dict):
    """Turn API JSON (dict keyed by k OR {'per_k': [...]}) into tidy DataFrames."""
    import numpy as np
    import pandas as pd

    rows_eta, rows_p, perm_rows = [], [], []

    # normalize to a list of (k, resdict)
    if "per_k" in payload and isinstance(payload["per_k"], list):
        # legacy/alternate shape
        for item in payload["per_k"]:
            k = int(item["k"])
            # eta_squared and p_value_corrected expected as flat lists
            for v in item.get("eta_squared", []):
                rows_eta.append({"k": k, "eta_squared": float(v) if v is not None else np.nan})
            for p in item.get("p_value_corrected", []):
                rows_p.append({"k": k, "p_value_corrected": float(p) if p is not None else np.nan})
            perm_rows.append({
                "k": k,
                "permanova_p": float(item.get("permanova_p")) if item.get("permanova_p") is not None else np.nan
            })
    else:
        # current server shape: { "2": { "anova_results": [...], "permanova_results": {...} }, ... }
        for k_key, res in payload.items():
            k = int(k_key)
            anova = res.get("anova_results") or []
            for row in anova:
                eta = row.get("eta_squared")
                padj = row.get("p_value_corrected")
                rows_eta.append({"k": k, "eta_squared": float(eta) if eta is not None else np.nan})
                rows_p.append({"k": k, "p_value_corrected": float(padj) if padj is not None else np.nan})
            perm = res.get("permanova_results") or {}
            perm_p = perm.get("p_value")
            perm_rows.append({"k": k, "permanova_p": float(perm_p) if perm_p is not None else np.nan})

    df_eta = pd.DataFrame(rows_eta)
    df_p   = pd.DataFrame(rows_p)
    df_perm = pd.DataFrame(perm_rows).sort_values("k").reset_index(drop=True)

    # k list for plotting order
    k_sorted = sorted(set(df_eta.get("k", pd.Series([], dtype=int))) |
                      set(df_p.get("k", pd.Series([], dtype=int))) |
                      set(df_perm.get("k", pd.Series([], dtype=int))))

    return df_eta, df_p, df_perm, k_sorted