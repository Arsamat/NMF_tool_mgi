import pandas as pd
import streamlit as st
from scipy.stats import hypergeom
import numpy as np

def hypergeometric(module_scores, meta, cluster_labels):
    cluster = st.selectbox("Select cluster for the test", set(cluster_labels))
    indexes = np.where(cluster_labels == cluster)[0]
    sample_order = module_scores.index[indexes]

    
    selected_cols = st.multiselect(
        "Select metadata columns to use for enrichment:",
        options=meta.columns.tolist()
    )
    
    # --- Step 2: For each selected column, allow choosing unique values ---
    selected_values = {}

    if selected_cols:
        st.markdown("### Choose values for each selected column")

        for col in selected_cols:
            unique_vals = sorted(meta[col].dropna().unique())
            chosen = st.selectbox(
                f"Select values from `{col}`:",
                options=unique_vals,
                key=f"value_selector_{col}"
            )
            selected_values[col] = chosen

        # --- Given variables ---
        # module_scores: pd.DataFrame with a 'Sample' column and index
        # meta: pd.DataFrame with metadata (including SampleName, Genotype, Dose, etc.)
        # indexes: subset of module_scores index (selected samples or modules)
        # selected_values: dict of user-selected metadata filters, e.g.
        #   {"Genotype": "TDP43", "Dose": "EC10"}

        # --- Step 1: Find M ---
        M = len(module_scores.index)  # total number of samples (or modules)

        # --- Step 2: Find N ---
        tmp = module_scores.loc[sample_order]
        N = len(tmp)  # number of selected samples

        # --- Step 3: Build mask for metadata filters (n) ---
        mask_n = pd.Series(True, index=meta.index)
        for col, vals in selected_values.items():
            if vals:
                if isinstance(vals, str):
                    vals = [vals]
                mask_n &= meta[col].isin(vals)

        # n = number of samples in meta matching user-selected criteria
        n = len(meta[mask_n])

        # --- Step 4: Build mask for k (samples that are both in module and match meta filter) ---
        # first, get the samples used in the module
        samples = module_scores.loc[sample_order].index.tolist()

        # subset metadata to only those samples
        meta_filtered = meta[meta[st.session_state["metadata_index"]].isin(samples)]

        # apply same filter logic again on the subset
        mask_k = pd.Series(True, index=meta_filtered.index)
        for col, vals in selected_values.items():
            if vals:
                if isinstance(vals, str):
                    vals = [vals]
                mask_k &= meta_filtered[col].isin(vals)

        k = len(meta_filtered[mask_k])

        # --- Step 5: Compute p-value using hypergeometric test ---
        p_value = hypergeom.sf(k - 1, M, n, N)
        st.write("Computed p-value is: ", p_value)
    
