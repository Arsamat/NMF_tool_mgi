import streamlit as st
import pandas as pd
import requests
import json
import io
from ui_theme import apply_custom_theme
import seaborn as sns
import matplotlib.pyplot as plt

def run_explore_correlations():
    apply_custom_theme()

    st.subheader("Get correlation statistic for each module")
    st.markdown('''
            Some NMF gene modules are more similar than others, in terms of their ranking of genes by contribution scores. 
            Use this tool to calculate the pairwise Spearman correlations between your gene modules.
                ''')
    if "correlation_values" not in st.session_state:
        st.session_state["correlation_values"] = None
    files = {}

    if "cnmf_gene_loadings" in st.session_state and st.session_state["cnmf_gene_loadings"] is not None:
        buf = io.BytesIO()
        st.session_state["cnmf_gene_loadings"].reset_index(drop=False).to_feather(buf)
        buf.seek(0)

        st.dataframe(st.session_state["cnmf_gene_loadings"])

        files["gene_loadings"] = ("data.feather", buf, "application/octet-stream")    

        if st.button("Get Correlation Statistic"):
            with st.spinner("Getting Information..."):
                r = requests.post(st.session_state["API_URL"] + "explore_correlations", files=files)
                r.raise_for_status()

                data = json.loads(r.content)
                records = data["pairs_by_k"]["result"]

                st.session_state["correlation_values"] = records

    if st.session_state["correlation_values"] is not None:
        df = pd.DataFrame(st.session_state["correlation_values"])
        st.dataframe(df)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"spearman_correlations.csv",
            mime="text/csv"
        )

        value_col = "rho"  # or "abs_rho"

        # Build square correlation matrix
        modules = sorted(set(df["module_i"]) | set(df["module_j"]), key=lambda x: int(x))
        corr_mat = pd.DataFrame(index=modules, columns=modules, dtype=float)

        for _, row in df.iterrows():
            corr_mat.loc[row["module_i"], row["module_j"]] = row[value_col]
            corr_mat.loc[row["module_j"], row["module_i"]] = row[value_col]

        # Fill diagonal with 1.0
        for m in modules:
            corr_mat.loc[m, m] = 1.0

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_mat.astype(float),
            cmap="coolwarm",
            vmin=-1, vmax=1,
            annot=True, fmt=".2f", square=True
        )
        plt.title("Spearman Correlation Heatmap (Modules)")
        plt.xlabel("Module")
        plt.ylabel("Module")
        plt.tight_layout()
        st.pyplot(plt)

        # --- Download Button ---
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="📥 Download Heatmap as PNG",
            data=buf,
            file_name="spearman_correlation_heatmap.png",
            mime="image/png"
        )

    # if st.button("Continue"):
    #         st.session_state["_go_to"] = "Pathview Analysis"
    #         st.rerun()