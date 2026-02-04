import streamlit as st
import requests
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import zipfile
from preview_heatmap import preview_wide_heatmap_inline
from ui_theme import apply_custom_theme
import json
from nmf_clustering import plot_clusters, plot_module_clusters
from make_expression_heatmap import get_expression_heatmap
from hypergeometric import hypergeom_ui
from module_clustering import m_clustering
from module_heatmap import module_heatmap_ui
import base64
from PIL import Image

def run_nmf():
    apply_custom_theme()

    # -------------------------------------------------------------
    # DEFAULTS
    # -------------------------------------------------------------
    DEFAULTS = {
        "API_URL": "http://52.14.223.10:8000/",
        "preprocessed_feather": None,
        "gene_loadings": None,
        "module_usages": None,
        "previous_heatmaps": {},
        "sample_order": None,
        "display_scores": False,
        "display_loadings": False,
        "display_previous_heatmaps": False,
        "sample_dendogram": None,
        "initial_sample_dendogram": None,
        "initial_module_dendogram": None,
        "module_dendogram": None,
        "sample_order_heatmap": None,
        "module_order_heatmap": None,
        "top_order_heatmap": None,
        "expression_heatmap": None,
        "module_leaf_order": None,
        "module_cluster_labels": None,
        "preview_png": None,
        "annotations_default": None
    }

    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


    # -------------------------------------------------------------
    # CACHING UTILITIES
    # -------------------------------------------------------------
    @st.cache_data
    def cached_feather_bytes(df):
        """DataFrame → feather bytes (cached)."""
        buf = io.BytesIO()
        df.reset_index(drop=False).to_feather(buf)
        buf.seek(0)
        return buf


    def save_pdf(df):
        fig_width = min(200, max(20, 0.15 * df.shape[1]))
        fig_height = min(50, max(6, 0.15 * df.shape[0]))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(df, cmap="viridis", ax=ax, cbar=True)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Modules")
        plt.tight_layout()

        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        buf.seek(0)
        return buf.read()


    # -------------------------------------------------------------
    # UI HEADER
    # -------------------------------------------------------------
    st.subheader("Run NMF algorithm")
    st.markdown("""
    Run NMF at your choice of k and generate a module usage heatmap.  
    Useful for early validation before consensus NMF.
    If the algorithm does not converge try increasing maximum iteration parameter, in some cases by 2x or 3x times
    """)

    k = st.number_input("k", 2, 50, 7)
    max_iter = st.number_input("Maximum number of iterations", 100, 20000, 5000)

    if "meta" not in st.session_state or st.session_state["meta"] is None:
        st.error("Upload metadata first.")
        st.stop()

    meta = st.session_state["meta"]
    st.write("**Metadata available:**")
    st.dataframe(meta.head())

    # -------------------------------------------------------------
    # Run NMF
    # -------------------------------------------------------------
    if st.button("Run NMF"):
        # files = {
        #     "preprocessed": (
        #         "preprocessed.feather",
        #         st.session_state["preprocessed_feather"],
        #         "application/octet-stream",
        #     )
        # }
        data = {"job_id": st.session_state["job_id"], "k": int(k), "max_iter": int(max_iter), "design_factor": "Group"}

        with st.spinner("Running NMF..."):
            try:
                r = requests.post(
                    st.session_state["API_URL"] + "run_regular_nmf",
                    #files=files,
                    data=data,
                    timeout=600,
                )
                r.raise_for_status()
                zip_bytes = io.BytesIO(r.content)

                with zipfile.ZipFile(zip_bytes, "r") as z:
                    with z.open("metadata.json") as f:
                        status = json.load(f)
                    if not status["converged"]:
                        st.error("NMF did not converge. Increase max_iter.")

                    for name in z.namelist():
                        if "_w" in name:
                            st.session_state["module_usages"] = pd.read_feather(z.open(name)).set_index("sample")
                            st.session_state["module_usages_T"] = st.session_state["module_usages"].T
                            
                        elif "_h" in name:
                            st.session_state["gene_loadings"] = pd.read_feather(z.open(name)).set_index("module")
                if r.status_code == 200:
                    st.success("NMF finished running. You can now visualize heatmaps")

            except Exception as e:
                st.error(f"Server error: {e}")

    st.markdown("---")
    

    # -------------------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------------------
    if st.session_state["gene_loadings"] is not None and st.session_state["module_usages"] is not None:
        st.subheader("Review NMF Results")

        def toggle_display(key):
            st.session_state[key] = not st.session_state[key]

        # Gene loadings
        if st.session_state["display_loadings"]:
            st.subheader("NMF Gene Loadings")
            st.dataframe(st.session_state["gene_loadings"])
            st.button("Hide", key="hide_loadings_button", on_click=toggle_display, args=("display_loadings",))
        else:
            st.button("Show Gene Loadings", key="display_loadings_button", on_click=toggle_display, args=("display_loadings",))

        # Module usages
        if st.session_state["display_scores"]:
            st.subheader("NMF Module Usage Scores")
            st.dataframe(st.session_state["module_usages"])
            st.button("Hide", key="hide_usages_button", on_click=toggle_display, args=("display_scores",))
        else:
            st.button("Show Usage Scores", "show_usages_button", on_click=toggle_display, args=("display_scores",))


        df = st.session_state["module_usages_T"]

        # -------------------------------------------------------------
        # PREVIEW HEATMAP (Optimized: form + cached PNG only)
        # -------------------------------------------------------------
        st.subheader("Wide Heatmap Preview (downsampled)")

        with st.form("preview_form"):
            annotation_cols = st.multiselect(
                "Select metadata columns",
                options=meta.columns.tolist(),
                key="annotations_widget"
            )

            average_groups = st.checkbox("Average groups (smooth)")

            submitted = st.form_submit_button("Generate Preview")

        if submitted:
            module_bytes = cached_feather_bytes(st.session_state["module_usages_T"])
            meta_bytes = cached_feather_bytes(meta)
            metadata_index = st.session_state.get("metadata_index", "")
            st.session_state["annotations_default"] = annotation_cols

            files = {
                "df": ("modules.feather", module_bytes, "application/octet-stream"),
                "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
            }

            data = {"metadata_index": metadata_index, "average_groups": average_groups, 
                    "annotation_cols": json.dumps(annotation_cols)}

            resp = requests.post(st.session_state["API_URL"] + "/initial_heatmap_preview/", files=files, data=data)

            st.session_state["preview_png"] = resp.content

        if st.session_state["preview_png"] is not None:
            st.image(st.session_state["preview_png"])
            st.download_button(
                "Download PNG",
                st.session_state["preview_png"],
                "heatmap_preview.png",
                mime="image/png"
            )

            pdf_bytes = save_pdf(df)
            st.download_button(
                "Download PDF",
                pdf_bytes,
                "heatmap.pdf",
                mime="application/pdf"
            )

        # -------------------------------------------------------------
        # ALL BACKEND-RENDERED SECTIONS (unchanged)
        # -------------------------------------------------------------
        # Everything below stays structurally identical but benefits from
        # performance improvements above.

        # SAMPLE CLUSTERING
        if st.checkbox("Hierarchically cluster samples"):
            module_bytes = cached_feather_bytes(st.session_state["module_usages"])
            meta_bytes = cached_feather_bytes(meta)

            st.subheader("Step 1: Run Hierarchical Clustering of Samples to Determine Number of Clusters")
            if st.button("Run"):
                files = {
                    "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                    "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
                }
                data = {"metadata_index": st.session_state["metadata_index"], "k": 0}

                resp = requests.post(st.session_state["API_URL"] + "/cluster_samples/", files=files, data=data)
                if resp.status_code != 200:
                    st.error(resp.text)
                else:
                    payload = resp.json()
                    st.session_state["initial_sample_dendogram"] = bytes.fromhex(payload["dendrogram_png"])
                
            if st.session_state["initial_sample_dendogram"]:
                st.image(st.session_state["initial_sample_dendogram"])

                st.download_button(
                    "Download PNG",
                    st.session_state["initial_sample_dendogram"],
                    "initial_sample_dendogram.png",
                    mime="image/png"
                )


            st.subheader("Step 2: Run Run Hierarchical Clustering of Samples Broken Down Into Selected Number of Clusters")
            k_samples = st.number_input("Number of clusters", 2, 10, 3)

            if st.button("Run Sample Clustering"):
                files = {
                    "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                    "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
                }
                data = {"metadata_index": st.session_state["metadata_index"], "k": str(k_samples)}

                resp = requests.post(st.session_state["API_URL"] + "/cluster_samples/", files=files, data=data)
                if resp.status_code != 200:
                    st.error(resp.text)
                else:
                    payload = resp.json()
                    st.session_state["sample_leaf_order"] = payload["leaf_order"]
                    st.session_state["sample_cluster_labels"] = payload["cluster_labels"]
                    st.session_state["sample_order"] = payload["sample_order"]
                    st.session_state["sample_dendogram"] = bytes.fromhex(payload["dendrogram_png"])
                    st.session_state["sample_order_heatmap"] = bytes.fromhex(payload["heatmap_png"])

            if st.session_state["sample_dendogram"] is not None:
                st.subheader("Sample Dendrogram")
                st.image(st.session_state["sample_dendogram"])

                st.download_button(
                    "Download PNG",
                    st.session_state["sample_dendogram"],
                    "sample_dendogram.png",
                    mime="image/png"
                )

            # ANNOTATED HEATMAP
            st.subheader("Step 3: Create Annotated Heatmap Based on Sample Clustering Order from Previous Step")
            if "sample_leaf_order" in st.session_state:
                with st.form("Annotated_heatmap_form"):
                    annotation_cols_annot = st.multiselect(
                        "Annotation columns",
                        ["Cluster"] + meta.columns.tolist(),
                        default=["Cluster"] + st.session_state["annotations_default"]
                    )
                    submit_annot = st.form_submit_button("Generate Annotated Heatmap")

                if submit_annot:
                    files = {
                        "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                        "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
                    }
                    data = {
                        "metadata_index": st.session_state["metadata_index"],
                        "leaf_order": json.dumps(st.session_state["sample_leaf_order"]),
                        "annotation_cols": json.dumps(annotation_cols_annot),
                        "cluster_labels": json.dumps(st.session_state["sample_cluster_labels"]),
                    }

                    resp = requests.post(st.session_state["API_URL"] + "/annotated_heatmap/", 
                                        files=files, data=data)

                    if resp.status_code == 200:
                        st.session_state["sample_order_heatmap"] = resp.content
                    else:
                        st.error(resp.text)


                if st.session_state["sample_order_heatmap"] is not None:
                    st.image(st.session_state["sample_order_heatmap"])

                    st.download_button(
                        "Download PNG",
                        st.session_state["sample_order_heatmap"],
                        "sample_order_heatmap.png",
                        mime="image/png"
                    )
            
            if st.checkbox("Calculate Hypergeometric Values"):
                hypergeom_ui(meta_bytes, st.session_state["module_usages"], st.session_state["sample_cluster_labels"])

            # MODULE CLUSTERING
            if st.checkbox("Hierarchically Cluster Modules"):
                st.subheader("Step 1: Visualise Hierarchical Clustering of Modules")
                if st.button("Run", key="module_clusters"):
                    st.session_state["initial_module_dendogram"] = m_clustering(
                        st.session_state["module_usages"],
                        st.session_state["sample_order"],
                        0
                    )
                if st.session_state["initial_module_dendogram"]:
                        st.image(st.session_state["initial_module_dendogram"])

                        st.download_button(
                            "Download PNG",
                            st.session_state["initial_module_dendogram"],
                            "initial_module_dendogram.png",
                            mime="image/png"
                        )
                
                st.subheader("Step 2: Select Number of Module Clusters")
                st.write("Clusters will be distinguished based on label colors")

                st.header("Cluster Modules")
                n = st.slider("Module clusters", 2, 12, 4)

                if st.button("Run Module Clustering"):
                    dendro_png = m_clustering(
                        st.session_state["module_usages"],
                        st.session_state["sample_order"],
                        n
                    )
                    if dendro_png:
                        st.session_state["module_dendogram"] = dendro_png

                if st.session_state["module_dendogram"] is not None:
                    st.image(st.session_state["module_dendogram"])

                    st.download_button(
                            "Download PNG",
                            st.session_state["module_dendogram"],
                            "module_dendogram.png",
                            mime="image/png"
                        )

                module_heatmap_ui(
                    meta_bytes,
                    st.session_state["module_usages"],
                    st.session_state["sample_order"],
                    st.session_state["module_leaf_order"],
                    st.session_state["module_cluster_labels"],
                    cnmf=False,
                    default_annotations=st.session_state["annotations_default"]
                )

        # ORDER BY TOP SAMPLES
        if st.checkbox("Visualize Heatmap Ordered by Top Module Samples"):
            st.markdown("""
                You can visualize the heatmap similar to previous steps. 
                But this time it will be reordered by finding top expressed samples for each module.
            """)
            with st.form("annotated_top_samples_form"):
                annotation_cols = st.multiselect("Select metadata columns", 
                                    meta.columns.tolist(),
                                    default = st.session_state["annotations_default"])
                submit_top_samples = st.form_submit_button("Generate Annotated Heatmap")

            if submit_top_samples:
                files = {
                    "module_usages": ("modules.feather", cached_feather_bytes(st.session_state["module_usages"]), "application/octet-stream"),
                    "metadata": ("meta.feather", cached_feather_bytes(meta), "application/octet-stream"),
                }
                data = {"metadata_index": st.session_state["metadata_index"], "annotation_cols": ",".join(annotation_cols)}

                resp = requests.post(st.session_state["API_URL"] + "/heatmap_top_samples/", files=files, data=data)
                if resp.status_code != 200:
                    st.error(resp.text)
                else:
                    st.session_state["top_order_heatmap"] = bytes.fromhex(resp.json()["heatmap_png"])

            if st.session_state["top_order_heatmap"] is not None:
                st.image(st.session_state["top_order_heatmap"])

                st.download_button(
                            "Download PNG",
                            st.session_state["top_order_heatmap"],
                            "top_order_heatmap.png",
                            mime="image/png"
                        )

        # EXPRESSION MATRIX
        if st.checkbox("Visualize Gene Expression Matrix"):
            st.markdown("""
                This step selects top genes from each module. Then visualizes their expression in each sample.
                        """)
            heatmap = get_expression_heatmap(st.session_state["gene_loadings"], st.session_state.get("annotations_default"))
            if heatmap is not None:
                st.session_state["expression_heatmap"] = heatmap

            if st.session_state["expression_heatmap"] is not None:
                st.image(st.session_state["expression_heatmap"])

                st.download_button(
                            "Download PNG",
                            st.session_state["expression_heatmap"],
                            "expression_heatmap.png",
                            mime="image/png"
                        )
                
            

    if st.button("Continue"):
        st.session_state["_go_to"] = "Run cNMF"
        st.rerun()
