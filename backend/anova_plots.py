import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ---------- helpers: plotting (matplotlib only) ----------

def make_eta_boxplot(df_eta: pd.DataFrame, k_sorted):
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df_eta.loc[df_eta["k"] == k, "eta_squared"].dropna().values for k in k_sorted]
    if any(len(arr) for arr in data):
        ax.boxplot(data, labels=k_sorted)
    else:
        ax.text(0.5, 0.5, "No η² data", ha="center", va="center")
    ax.set_xlabel("k")
    ax.set_ylabel("Effect size (η²)")
    ax.set_title("Distribution of η² across k")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig

def make_p_boxplot(df_p: pd.DataFrame, k_sorted, alpha: float, use_log10: bool = False):
    fig, ax = plt.subplots(figsize=(9, 5))
    if use_log10:
        pvals = df_p["p_value_corrected"].clip(lower=1e-300)
        df_use = df_p.assign(neglog10=-np.log10(pvals))
        data = [df_use.loc[df_use["k"] == k, "neglog10"].dropna().values for k in k_sorted]
        if any(len(arr) for arr in data):
            ax.boxplot(data, labels=k_sorted)
        else:
            ax.text(0.5, 0.5, "No p-value data", ha="center", va="center")
        ax.set_ylabel(r"$-\log_{10}$(FDR-adjusted p)")
        ax.axhline(-np.log10(max(alpha, 1e-300)), ls="--", alpha=0.6)
        ax.set_title("Distribution of −log10(FDR p) across k")
    else:
        data = [df_p.loc[df_p["k"] == k, "p_value_corrected"].dropna().values for k in k_sorted]
        if any(len(arr) for arr in data):
            ax.boxplot(data, labels=k_sorted)
        else:
            ax.text(0.5, 0.5, "No p-value data", ha="center", va="center")
        ax.set_ylabel("FDR-adjusted p-value")
        ax.axhline(alpha, ls="--", alpha=0.6)
        ax.set_title("Distribution of FDR p across k")
    ax.set_xlabel("k")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig

def make_permanova_boxplot(df_perm, k_sorted):
    """
    Boxplot + scatter overlay of PERMANOVA r2_adj across k values,
    with a line connecting the mean values for each k.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    if "k" not in df_perm.columns or "r2_adj" not in df_perm.columns:
        ax.text(0.5, 0.5,
                "No PERMANOVA data (columns 'k' and 'r2_adj' not found)",
                ha="center", va="center")
        return fig

    # Collect values for each k
    data = [df_perm.loc[df_perm["k"] == k, "r2_adj"].dropna().values for k in k_sorted]

    if any(len(arr) for arr in data):
        ax.boxplot(data, labels=k_sorted)

        # Overlay scatter
        for i, arr in enumerate(data, start=1):
            ax.scatter([i] * len(arr), arr, alpha=0.6, s=20, color="blue")

        # Compute mean for each k and connect with a line
        means = [arr.mean() if len(arr) > 0 else None for arr in data]
        valid_idx = [i for i, m in enumerate(means, start=1) if m is not None]
        valid_means = [m for m in means if m is not None]

        ax.plot(valid_idx, valid_means, color="red", marker="o", linestyle="-",
                linewidth=2, label="Mean r2_adj")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No PERMANOVA values found",
                ha="center", va="center")

    ax.set_xlabel("k")
    ax.set_ylabel("PERMANOVA Adj. R²")
    ax.set_title("Distribution of PERMANOVA r2_adj across k")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig
