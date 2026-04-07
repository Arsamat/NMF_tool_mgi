import streamlit as st
from ui_theme import apply_custom_theme

apply_custom_theme()

st.set_page_config(page_title="Analysis Suite", layout="wide")

# --- Global home styling (only affects this page) ---
st.markdown(
    """
    <style>
    .tool-card {
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.04);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
        min-height: 148px;
    }
    .tool-card h3 {
        margin: 0 0 6px 0;
        font-size: 18px;
        line-height: 1.2;
    }
    .tool-card p {
        margin: 0 0 10px 0;
        opacity: 0.9;
        font-size: 13.5px;
        line-height: 1.35;
    }
    .home-hint {
        font-size: 13.5px;
        opacity: 0.9;
    }
    /* Make primary action buttons feel more prominent */
    div.stButton > button {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
if "ui_mode" not in st.session_state:
    st.session_state["ui_mode"] = "home"
if "active_page" not in st.session_state:
    st.session_state["active_page"] = None   


if st.session_state["ui_mode"] == "home":
    st.title("🔬 Bioinformatics Analysis Suite")
    st.markdown(
        "<div class='home-hint'>Choose a tool below. Some tools include sidebar steps; others open directly. Use the back button to return here.</div>",
        unsafe_allow_html=True,
    )

    # Tool grid (2 columns)
    c1, c2 = st.columns(2)

    def _tool_card(col, title, desc, button_label, active_page_value):
        with col:
            st.markdown(f"<div class='tool-card'><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)
            if st.button(button_label, key=f"go_{active_page_value}"):
                st.session_state["ui_mode"] = "tool"
                st.session_state["active_page"] = active_page_value
                st.rerun()
    _tool_card(
        c1,
        "Browse by Experiment",
        "Download metadata and counts, run your own DE comparison based on experiment, view and access pre-computed DE results",
        "Go to Browse by Experiment",
        "DE Browse by Experiment",
    )
    _tool_card(
        c2,
        "Browse by Metadata",
        "Filter samples from any experiment and download data. Optionally run your own DE comparison or NMF analysis on selected samples.",
        "Go to Browse by Metadata",
        "DE De Novo by Samples",
    )
    _tool_card(
        c1,
        "NMF for Bulk RNA",
        "Upload metadata and counts. Preprocess data, explore K, run NMF or cNMF, visualize and download results",
        "Start Bulk NMF",
        "NMF for Bulk RNA",
    )
    _tool_card(
        c2,
        "NMF for Single-Cell RNA",
        "Upload metadata and counts. Preprocess data, run cNMF, visualize and download results",
        "Start Single-Cell NMF",
        "NMF for Single-Cell RNA",
    )
    _tool_card(
        c1,
        "Feedback Form",
        "Tell us what works / what doesn’t. We’ll use it to improve the workflow.",
        "Open Feedback",
        "Feedback Form",
    )

if st.session_state["ui_mode"] == "tool":
    # -------------------------------------------
    # TOP-LEVEL NAVIGATION
    # -------------------------------------------
    st.sidebar.title("Navigation Menu")

    # if "_go_to_main" in st.session_state:
    #         st.session_state["main_section"] = st.session_state["_go_to_main"]
    #         del st.session_state["_go_to_main"]

    if st.sidebar.button("← Back to Home", key="back_home"):
        st.session_state["ui_mode"] = "home"
        st.session_state["active_main"] = None
        st.rerun()

    main_page = st.session_state["active_page"]

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
    elif main_page == "DE Browse by Experiment":
        from deg.browse_brb_by_experiment.experiment_browser import run_experiment_browser

        st.session_state["deg_api_url"] = "http://18.218.84.81:8000/"           
        run_experiment_browser()

    elif main_page == "DE De Novo by Samples":
        from deg.browse_brb_by_metadata.group_selection import run_group_selection

        st.session_state["deg_api_url"] = "http://18.218.84.81:8000/"
        run_group_selection()

    elif main_page == "Feedback Form":
        st.write("If you experienced an error or would like to provide feedback, you can access the form by clicking the button below")
        st.link_button("Provide Feedback", "https://forms.cloud.microsoft/Pages/ResponsePage.aspx?id=taPMTM1xbU6XS02b65bG1gY_uBy6G5ZIhg2MdbY5agRUODZHVklVVVhHRUZVMjZCVzNZVDY5UUxINS4u")

