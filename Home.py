import streamlit as st
from ui_theme import apply_custom_theme
from brb_data_pages.extract_counts_frontend import extract_data
from brb_data_pages.visualize_data import visualize_metadata

apply_custom_theme()

st.set_page_config(page_title="Analysis Suite", layout="wide")
st.title("🔬 Bioinformatics Analysis Suite")

# -------------------------------------------
# TOP-LEVEL NAVIGATION
# -------------------------------------------
st.sidebar.title("Main Menu")

if "_go_to_main" in st.session_state:
        st.session_state["main_section"] = st.session_state["_go_to_main"]
        del st.session_state["_go_to_main"]

main_page = st.sidebar.radio(
    "Select a section:",
    ["NMF for Bulk RNA", "NMF for Single-Cell RNA", "DE Analysis", "Obtain Data", "Feedback Form"],
    key="main_section"
)

# -------------------------------------------
# If user selects NMF → load NMF subpages
# -------------------------------------------
if main_page == "NMF for Bulk RNA":
    from nmf_nav.home import home_page
    from nmf_nav.metadata_upload import run_metadata_upload
    from nmf_nav.preprocess2_data import run_preprocess_data
    from nmf_nav.explore_k_values import run_explore_k_values
    from nmf_nav.run_light_nmf import run_nmf
    from nmf_nav.run_consensus_nmf import run_cnmf
    from nmf_nav.gene_descriptions import run_gene_loadings
    from nmf_nav.explore_module_correlations import run_explore_correlations
    # from pathway_analysis import run_pathway_analysis

    if "_go_to" in st.session_state:
        st.session_state["nmf_page"] = st.session_state["_go_to"]
        del st.session_state["_go_to"]
    
    NMF_PAGES = {
        "Home Page": home_page,
        "Metadata Upload": run_metadata_upload,
        "Preprocess Data for NMF": run_preprocess_data,
        "Explore K Parameter": run_explore_k_values,
        "Run NMF": run_nmf,
        "Run cNMF": run_cnmf,
        "Explore Gene Functions": run_gene_loadings,
        "Spearman Correlation Analysis": run_explore_correlations
        # "Pathview Analysis": run_pathway_analysis
    }

    nmf_subpage = st.sidebar.radio(
        "NMF Tools:",
        list(NMF_PAGES.keys()),
        key="nmf_page"
    )

    # Show selected NMF subpage
    NMF_PAGES[nmf_subpage]()

elif main_page == "NMF for Single-Cell RNA":
    from nmf_nav_sc.home_sc import home_page_sc
    from nmf_nav_sc.metadata_upload_sc import run_metadata_upload_sc
    from nmf_nav_sc.run_cnmf_sc import run_cnmf_sc
    from nmf_nav_sc.gene_descriptions_sc import run_gene_loadings_sc

    if "_go_to_sc" in st.session_state:
        st.session_state["nmf_page_sc"] = st.session_state["_go_to_sc"]
        del st.session_state["_go_to_sc"]
    
    NMF_PAGES_SC = {
        "Home": home_page_sc,
        "Upload Metadata": run_metadata_upload_sc,
        "Run cNMF": run_cnmf_sc,
        "Get Gene Descriptions": run_gene_loadings_sc
    }

    nmf_subpage = st.sidebar.radio(
        "NMF Tools:",
        list(NMF_PAGES_SC.keys()),
        key="nmf_page_sc"
    )

    # Show selected NMF subpage
    NMF_PAGES_SC[nmf_subpage]()
     


# -------------------------------------------
# OTHER TOP-LEVEL SECTIONS
# -------------------------------------------
elif main_page == "Obtain Data":
    from brb_data_pages.extract_counts_frontend import extract_data
    #from brb_data_pages.visualize_data import visualize_metadata
    
    if "_go_to_data_page" in st.session_state:
        st.session_state["data_page"] = st.session_state["_go_to_data_page"]
        del st.session_state["_go_to_data_page"]

    DATA_PAGES = {
        "Get Data": extract_data,
        # "Visualize Data": visualize_metadata
    }

    data_subpage = st.sidebar.radio(
        "NMF Tools:",
        list(DATA_PAGES.keys()),
        key="data_page"
    )

    # Show selected NMF subpage
    DATA_PAGES[data_subpage]()


elif main_page == "DE Analysis":
    st.header("🧪 Differential Expression Analysis")
    st.write("DESeq2, edgeR, volcano plots, etc.")

elif main_page == "Feedback Form":
    st.write("If you experienced an error or would like to provide feedback, you can access the form by clicking the button below")
    st.link_button("Provide Feedback", "https://forms.cloud.microsoft/Pages/ResponsePage.aspx?id=taPMTM1xbU6XS02b65bG1gY_uBy6G5ZIhg2MdbY5agRUODZHVklVVVhHRUZVMjZCVzNZVDY5UUxINS4u")

