import streamlit as st
import requests
from streamlit_autorefresh import st_autorefresh

from ui_theme import apply_custom_theme

def home_page_sc():
    apply_custom_theme()

    # --- Session state ---
    st.session_state.setdefault("preprocessed_feather_sc", None)
    st.session_state.setdefault("results_sc", {})
    st.session_state.setdefault("spearman_img_sc", None)
    st.session_state.setdefault("gene_column_sc", None)
    st.session_state.setdefault("hvg_sc", None)
    st.session_state.setdefault("metadata_index_sc", None)
    st.session_state.setdefault("design_factor_sc", None)
    st.session_state.setdefault("gene_symbols_sc", None)
    st.session_state.setdefault("k_range_sc", [])
    st.session_state.setdefault("max_iter_sc", 0)
    st.session_state.setdefault("heatmap_pdf_bytes_sc", [])

    if "counts_sc" not in st.session_state:
        st.session_state["counts_sc"] = None
    if "meta_sc" not in st.session_state:
        st.session_state["meta_sc"] = None

    # For Spearman plots
    st.session_state.setdefault("spearman_boxplot_png_sc", None)
    st.session_state.setdefault("spearman_max_png_sc", None)
    st.session_state.setdefault("pairs_by_k_sc", {})

    st.session_state.setdefault("API_URL_sc", "")
    st.session_state.setdefault("LAMBDA_URL_sc", "")
    st.session_state.setdefault("HEALTH_URL_sc", "")

    st.session_state["API_URL_sc"] = "https://18.218.84.81:8000/"
    #st.session_state["API_URL_sc"] = "http://3.141.231.76:8000/"
    # st.session_state["API_URL_sc"] = "http://52.14.223.10:8000/"

    st.write("# Welcome to NMF exploration tool!")
    st.markdown("""
    Non-negative matrix factorization is a dimensionality reduction and feature extraction approach for non-negative data. 
    An input matrix **V** containing expression counts for **m transcripts x n samples** will be decomposed into **W (m x k)** and **H (k x n)**, such that **V ≈ WH**.

    **H** represents usage scores for the **n samples** for each of **k gene modules**, that is, how much each sample **'uses'** each gene module. This webtool will generate heatmap visualizations of **H**, which can serve as a global view of transcriptomic changes across your sample groups.

    **W** informs how much each of the **m transcripts** contribute to each of the **k gene modules**, which may be useful for your follow-up analyses. Every transcript will have a contribution score to each of the **k gene modules**, but the contribution scores may vary significantly across the modules.

    The choice of **k** must be provided to the algorithm. You have the option to manually iterate through different values of **k (tab 'Run Light NMF')**, and decide which value best captures global transcriptomic differences according to your experimental design.

    Alternatively, you can systematically iterate through different values of **k (tab 'Explore K Values')** and run a silhouette score analysis given your sample groupings, to identify a value of **k** that produces a matrix **H** that best reflects the experimental design.

    This method is not perfect and we encourage the user to manually inspect the heatmaps for matrix **H** at various **k within a range** of the **k** value identified by the **silhouette score** analysis before proceeding.

    Once an optimal value of **k** is determined, proceed to tab **'Run cNMF'**. Here you will be able to run **consensus NMF**, which repeats NMF at your chosen **k** many times with a different starting seed, and then generates the consensus result using the method developed by Kotliar et al. (link to https://elifesciences.org/articles/43803).

    You will be able to download the usage score heatmap for matrix **H**, the usage scores themselves, and the **gene spectra z-scores**, or how much each transcript contributes to each gene module.
    """
    )

    if st.button("Continue"):
        st.session_state["_go_to_sc"] = "Metadata Upload"
        st.rerun()



        


