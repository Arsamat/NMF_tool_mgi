import streamlit as st
import requests, io, base64, zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from ui_theme import apply_custom_theme
from module_clustering import m_clustering
from module_heatmap import module_heatmap_ui
from make_expression_heatmap import get_expression_heatmap
from hypergeometric import hypergeom_ui
from streamlit_autorefresh import st_autorefresh


def run_cnmf():
    # ================================================================
    # APPLY CUSTOM THEME
    # ================================================================
    apply_custom_theme()


    # ================================================================
    # DEFAULT SESSION STATE (CNMF VERSION)
    # ================================================================
    DEFAULTS = {
        # Data matrices (W and H)
        "cnmf_module_usages": None,
        "cnmf_module_usages_transposed": None,
        "cnmf_gene_loadings": None,
        # PDF heatmap returned from backend
        "cnmf_pdf": None,
        # Clustering variables
        "cnmf_sample_order": None,
        "cnmf_sample_leaf_order": None,
        "cnmf_sample_cluster_labels": None,
        "cnmf_sample_dendogram": None,
        "cnmf_sample_order_heatmap": None,
        "cnmf_initial_sample_dendogram": None,
        "cnmf_module_leaf_order": None,
        "cnmf_module_cluster_labels": None,
        "cnmf_initial_module_dendogram": None,
        "cnmf_module_dendogram": None,
        # Visualization
        "cnmf_top_order_heatmap": None,
        "cnmf_expression_heatmap": None,
        # Preview heatmap
        "cnmf_preview_png": None,
        "cnmf_annotations_default": None,
        # NMF background execution
        "cnmf_running": False,
        "cnmf_zip_bytes": None,
        "cnmf_queue": None,
        "cnmf_executor": None,
        "display_scores": None,
        "display_loadings": None
    }

    # Initialize missing keys
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    # Executor setup
    if st.session_state["cnmf_queue"] is None:
        st.session_state["cnmf_queue"] = queue.Queue()
    if st.session_state["cnmf_executor"] is None:
        st.session_state["cnmf_executor"] = ThreadPoolExecutor(max_workers=1)


    # ================================================================
    # UTILITY FUNCTIONS
    # ================================================================
    @st.cache_data
    def cached_feather_bytes(df):
        """Convert DF → feather bytes (cached)."""
        buf = io.BytesIO()
        df.reset_index(drop=False).to_feather(buf)
        buf.seek(0)
        return buf


    # @st.cache_data
    # def cached_preview_png(df, meta, annotation_cols, average_groups):
    #     """Return PNG bytes for preview, optimized for speed."""
    #     fig = preview_wide_heatmap_inline(
    #         df=df,
    #         meta=meta,
    #         annotation_cols=list(annotation_cols),
    #         average_groups=average_groups,
    #     )
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    #     buf.seek(0)
    #     return buf.getvalue()



    # ================================================================
    # BACKGROUND CNMF JOB HANDLER
    # ================================================================
    def run_cnmf_job(api_url, preprocessed_bytes, meta_bytes, post_data, q):
        """
        Launch CNMF job on backend. 
        Returns ZIP file content into queue when ready.
        """
        try:
            files = {
                "preprocessed": ("preprocessed.feather", preprocessed_bytes, "application/octet-stream"),
                "metadata": ("metadata.tsv", meta_bytes, "text/csv"),
            }

            r = requests.post(
                f"{api_url}run_nmf_files",
                files=files,
                data=post_data,
                timeout=900
            )
            r.raise_for_status()

            # Put ZIP bytes in queue
            q.put(("success", r.content))

        except Exception as e:
            q.put(("error", str(e)))


    # ================================================================
    # PAGE HEADER
    # ================================================================
    st.subheader("Run **Consensus NMF**")
    st.markdown("""
    Run Consensus NMF using the Kotliar et al. method. 
    This version uses background execution, allowing you to navigate elsewhere while CNMF runs.
    """)


    # ================================================================
    # REQUIRED INPUT VALIDATION
    # ================================================================
    # if "preprocessed_feather" not in st.session_state or st.session_state["preprocessed_feather"] is None:
    #     st.error("Run preprocessing first or upload preprocessed data.")
    #     st.stop()

    if "meta" not in st.session_state or st.session_state["meta"] is None:
        st.error("Upload metadata first.")
        st.stop()

    meta = st.session_state["meta"]
    metadata_index = st.session_state.get("metadata_index", "")

    st.write("**Metadata available:**")
    st.dataframe(meta.head())


    # ================================================================
    # CNMF PARAMETERS
    # ================================================================
    k = st.number_input("k", 2, 50, 7)
    if st.session_state["integer_format"]:
        hvg = st.number_input("Number of highly variable genes", 100, 20000, 2000)
    else: 
        hvg = 20000
    max_iter = st.number_input("Maximum number of iterations", 100, 30000, 8000)


    # ================================================================
    # RUN CNMF BUTTON
    # ================================================================
    if st.button("Run CNMF") and not st.session_state["cnmf_running"]:
        api_url = st.session_state["API_URL"]
        preprocessed_bytes = st.session_state["preprocessed_feather"]

        meta_buf = io.StringIO()
        meta.to_csv(meta_buf, sep="\t", index=False)
        meta_bytes = meta_buf.getvalue().encode("utf-8")

        post_data = {
            "k": int(k),
            "hvg": int(hvg),
            "max_iter": int(max_iter),
            "design_factor": "Group",
            "metadata_index": metadata_index,
            "job_id": st.session_state["job_id"],
            "batch_correct": "sample",
            "gene_column": "gene_name"
        }

        st.session_state["cnmf_executor"].submit(
            run_cnmf_job,
            api_url,
            preprocessed_bytes,
            meta_bytes,
            post_data,
            st.session_state["cnmf_queue"]
        )

        st.session_state["cnmf_running"] = True
        st.info("CNMF job started in background… This page will refresh automatically.")


    # ================================================================
    # AUTO-REFRESH WHILE RUNNING
    # ================================================================
    if st.session_state["cnmf_running"]:
        st_autorefresh(interval=5000, key="cnmf_autorefresh")

        try:
            status, payload = st.session_state["cnmf_queue"].get_nowait()

            if status == "success":
                st.session_state["cnmf_zip_bytes"] = payload
                st.success("CNMF completed successfully! You can now visualize heatmaps")
            else:
                st.error(f"CNMF failed: {payload}")

            st.session_state["cnmf_running"] = False

        except queue.Empty:
            st.info("CNMF is still running…")


    # ================================================================
    # PROCESS ZIP RESULTS
    # ================================================================
    zip_bytes = st.session_state.get("cnmf_zip_bytes")
    if not zip_bytes:
        st.info("No CNMF results yet.")
        st.stop()

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

    # Parse ZIP
    for name in zf.namelist():
        
        if name.endswith(".pdf"):
            st.session_state["cnmf_pdf"] = zf.read(name)

        elif ".usages" in name.lower():
            df = pd.read_csv(zf.open(name), sep="\t", index_col=0)
            st.session_state["cnmf_module_usages"] = df
            st.session_state["cnmf_module_usages_transposed"] = st.session_state["cnmf_module_usages"].T

        elif ".gene" in name.lower():
            df = pd.read_csv(zf.open(name), sep="\t", index_col=0)
            st.session_state["cnmf_gene_loadings"] = df

    # Check load
    if st.session_state["cnmf_module_usages"] is None or st.session_state["cnmf_gene_loadings"] is None:
        st.error("ZIP loaded but usages/gene loadings not found.")
        st.stop()


    st.markdown("---")
    st.subheader("Consensus NMF Results")


    # ================================================================
    # DISPLAY PDF FROM BACKEND
    # ================================================================
    if st.session_state["cnmf_pdf"] is not None:
        st.write("**Full heatmap PDF:**")
        st.download_button(
            "Download CNMF Heatmap PDF",
            data=st.session_state["cnmf_pdf"],
            file_name="cnmf_heatmap.pdf",
            mime="application/pdf"
        )


    # ================================================================
    # SHOW MATRICES
    # ================================================================
    def toggle(key):
        st.session_state[key] = not st.session_state[key]

    st.write("**Numeric Matrices Returned:**")

    if st.button("Show/Hide Module Usages"):
        toggle("display_scores")

    if st.session_state.get("display_scores", False):
        st.subheader("Module Usage Matrix (W)")
        st.dataframe(st.session_state["cnmf_module_usages"])


    if st.button("Show/Hide Gene Loadings"):
        toggle("display_loadings")

    if st.session_state.get("display_loadings", False):
        st.subheader("Gene Loadings Matrix (H)")
        st.dataframe(st.session_state["cnmf_gene_loadings"])


    # ================================================================
    # PREVIEW HEATMAP (LOCAL)
    # ================================================================
    st.markdown("---")
    st.subheader("Preview Heatmap")

    module_usages_T = st.session_state["cnmf_module_usages_transposed"]

    with st.form("cnmf_preview_form"):
        annotation_cols = st.multiselect(
            "Metadata columns for annotation",
            options=meta.columns.tolist(),
            key="cnmf_annotation_widget"
        )

        average_groups = st.checkbox("Average groups", value=False)

        submitted = st.form_submit_button("Generate Preview")

    if submitted:
        st.session_state["cnmf_annotations_default"] = annotation_cols

        common_samples = [
            s for s in meta[metadata_index] if s in module_usages_T.columns
        ]

        if not common_samples:
            st.warning("No overlapping samples in module usages and metadata.")
        else:
            #df_sub = module_usages_T[common_samples]
            module_bytes = cached_feather_bytes(st.session_state["cnmf_module_usages_transposed"])
            meta_bytes = cached_feather_bytes(meta)

            files = {
                "df": ("modules.feather", module_bytes, "application/octet-stream"),
                "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
            }

            data = {"metadata_index": metadata_index, "average_groups": average_groups, 
                    "annotation_cols": json.dumps(annotation_cols)}

            resp = requests.post(st.session_state["API_URL"] + "/initial_heatmap_preview/", files=files, data=data)

            st.session_state["cnmf_preview_png"] = resp.content

    if st.session_state["cnmf_preview_png"] is not None:
        st.image(st.session_state["cnmf_preview_png"])
        st.download_button(
            "Download PNG",
            data=st.session_state["cnmf_preview_png"],
            file_name="cnmf_preview.png",
            mime="image/png"
        )


    # ================================================================
    # SAMPLE CLUSTERING (BACKEND)
    # ================================================================
    st.markdown("---")
    if st.checkbox("Cluster Samples"):
        module_bytes = cached_feather_bytes(st.session_state["cnmf_module_usages"])
        meta_bytes = cached_feather_bytes(meta)

        st.subheader("Step 1 — Run Dendrogram Without Clustering")
        if st.button("Run Sample Dendrogram"):
            files = {
                "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
            }
            data = {"metadata_index": metadata_index, "k": 0}

            resp = requests.post(st.session_state["API_URL"] + "/cluster_samples/", files=files, data=data)

            if resp.status_code == 200:
                payload = resp.json()
                st.session_state["cnmf_initial_sample_dendogram"] = bytes.fromhex(payload["dendrogram_png"])
            else:
                st.error(resp.text)

        if st.session_state["cnmf_initial_sample_dendogram"] is not None:
            st.image(st.session_state["cnmf_initial_sample_dendogram"])

            st.download_button(
            "Download PNG",
            data=st.session_state["cnmf_initial_sample_dendogram"],
            file_name="cnmf_initial_sample_dendogram.png",
            mime="image/png"
        )


        st.subheader("Step 2 — Cluster Samples into Groups")
        k_samples = st.number_input("Number of sample clusters", 2, 10, 3)

        if st.button("Run Sample Clustering"):
            files = {
                "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
            }
            data = {"metadata_index": metadata_index, "k": str(k_samples)}

            resp = requests.post(st.session_state["API_URL"] + "/cluster_samples/", files=files, data=data)
            if resp.status_code != 200:
                st.error(resp.text)
            else:
                payload = resp.json()

                st.session_state["cnmf_sample_leaf_order"] = payload["leaf_order"]
                st.session_state["cnmf_sample_cluster_labels"] = payload["cluster_labels"]
                st.session_state["cnmf_sample_order"] = payload["sample_order"]
                st.session_state["cnmf_sample_dendogram"] = bytes.fromhex(payload["dendrogram_png"])
                st.session_state["cnmf_sample_order_heatmap"] = bytes.fromhex(payload["heatmap_png"])

        if st.session_state["cnmf_sample_dendogram"] is not None:
            st.image(st.session_state["cnmf_sample_dendogram"])

            st.download_button(
                "Download PNG",
                data=st.session_state["cnmf_sample_dendogram"],
                file_name="cnmf_sample_dendogram.png",
                mime="image/png"
            )


        # ============================================================
        # ANNOTATED HEATMAP
        # ============================================================
        st.subheader("Step 3 — Annotated Heatmap (Sample Order)")

        if st.session_state.get("cnmf_sample_leaf_order") is not None:
            with st.form("cnmf_annotated_heatmap_form"):
                annotation_cols_annot = st.multiselect(
                    "Annotation columns",
                    ["Cluster"] + meta.columns.tolist(),
                    default=st.session_state.get("cnmf_annotations_default")
                )
                submit_annot = st.form_submit_button("Generate Annotated Heatmap")

            if submit_annot:
                files = {
                    "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                    "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
                }
                data = {
                    "metadata_index": metadata_index,
                    "leaf_order": json.dumps(st.session_state["cnmf_sample_leaf_order"]),
                    "annotation_cols": json.dumps(annotation_cols_annot),
                    "cluster_labels": json.dumps(st.session_state["cnmf_sample_cluster_labels"]),
                }

                resp = requests.post(
                    st.session_state["API_URL"] + "/annotated_heatmap/",
                    files=files,
                    data=data
                )

                if resp.status_code == 200:
                    st.session_state["cnmf_sample_order_heatmap"] = resp.content
                else:
                    st.error(resp.text)

            if st.session_state["cnmf_sample_order_heatmap"] is not None:
                st.image(st.session_state["cnmf_sample_order_heatmap"])

                st.download_button(
                "Download PNG",
                data=st.session_state["cnmf_sample_order_heatmap"],
                file_name="cnmf_sample_order_heatmap.png",
                mime="image/png"
            )


        # ============================================================
        # Hypergeometric test
        # ============================================================
        if st.checkbox("Calculate Hypergeometric Values"):
            hypergeom_ui(
                meta_bytes,
                st.session_state["cnmf_module_usages"],
                st.session_state["cnmf_sample_cluster_labels"]
            )


        # ============================================================
        # MODULE CLUSTERING
        # ============================================================
        if st.checkbox("Cluster Modules"):
            st.subheader("Step 1 — Initial Module Dendrogram")

            if st.button("Run Module Dendrogram"):
                dendro_png = m_clustering(
                    st.session_state["cnmf_module_usages"],
                    st.session_state["cnmf_sample_order"],
                    0,
                    cnmf=True
                )
                st.session_state["cnmf_initial_module_dendogram"] = dendro_png

            if st.session_state["cnmf_initial_module_dendogram"]:
                st.image(st.session_state["cnmf_initial_module_dendogram"])

                st.download_button(
                    "Download PNG",
                    data=st.session_state["cnmf_initial_module_dendogram"],
                    file_name="cnmf_initial_module_dendogram.png",
                    mime="image/png"
                )

            st.subheader("Step 2 — Cluster Modules")
            n_mod = st.slider("Number of module clusters", 2, 12, 4)

            if st.button("Run Final Module Clustering"):
                dendro_png = m_clustering(
                    st.session_state["cnmf_module_usages"],
                    st.session_state["cnmf_sample_order"],
                    n_mod,
                    cnmf=True
                )
                if dendro_png:
                    st.session_state["cnmf_module_dendogram"] = dendro_png

            if st.session_state["cnmf_module_dendogram"]:
                st.image(st.session_state["cnmf_module_dendogram"])

                st.download_button(
                    "Download PNG",
                    data=st.session_state["cnmf_module_dendogram"],
                    file_name="cnmf_module_dendogram.png",
                    mime="image/png"
                )

            # Render heatmap using CNMF values
            module_heatmap_ui(
                meta_bytes,
                st.session_state["cnmf_module_usages"],
                st.session_state["cnmf_sample_order"],
                st.session_state["cnmf_module_leaf_order"],
                st.session_state["cnmf_module_cluster_labels"],
                cnmf=True,
                default_annotations=st.session_state["cnmf_annotations_default"]
            )


    # ================================================================
    # TOP SAMPLE ORDERING (BACKEND)
    # ================================================================
    if st.checkbox("Order by Top Samples"):

        with st.form("cnmf_top_samples_form"):
            annotation_cols = st.multiselect(
                "Metadata columns for annotation",
                meta.columns.tolist(),
                default=st.session_state.get("cnmf_annotations_default")
            )
            submit_top = st.form_submit_button("Generate")

        if submit_top:
            files = {
                "module_usages": ("modules.feather", cached_feather_bytes(st.session_state["cnmf_module_usages"]), "application/octet-stream"),
                "metadata": ("meta.feather", cached_feather_bytes(meta), "application/octet-stream"),
            }
            data = {
                "metadata_index": metadata_index,
                "annotation_cols": ",".join(annotation_cols)
            }

            resp = requests.post(st.session_state["API_URL"] + "/heatmap_top_samples/", files=files, data=data)

            if resp.status_code == 200:
                st.session_state["cnmf_top_order_heatmap"] = bytes.fromhex(resp.json()["heatmap_png"])
            else:
                st.error(resp.text)

        if st.session_state["cnmf_top_order_heatmap"]:
            st.image(st.session_state["cnmf_top_order_heatmap"])

            st.download_button(
                    "Download PNG",
                    data=st.session_state["cnmf_top_order_heatmap"],
                    file_name="cnmf_top_order_heatmap.png",
                    mime="image/png"
                )


    # ================================================================
    # EXPRESSION HEATMAP
    # ================================================================
    if st.checkbox("Show Gene Expression Matrix"):
        expr = get_expression_heatmap(st.session_state["cnmf_gene_loadings"], st.session_state.get("cnmf_annotations_default"))
        if expr is not None:
            st.session_state["cnmf_expression_heatmap"] = expr

        if st.session_state["cnmf_expression_heatmap"] is not None:
            st.image(st.session_state["cnmf_expression_heatmap"])
            st.download_button(
                    "Download PNG",
                    data=st.session_state["cnmf_expression_heatmap"],
                    file_name="cnmf_expression_heatmap.png",
                    mime="image/png"
                )
    # ============================================================================
    # END OF PAGE → NAVIGATION
    # ============================================================================

    if st.button("Continue"):
        st.session_state["_go_to"] = "Explore Gene Functions"
        st.rerun()
