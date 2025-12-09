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


@st.cache_data
def cached_preview_png(df, meta, annotation_cols, average_groups):
    """Return PNG bytes (not fig) for optimal speed."""
    annotation_cols = list(annotation_cols)

    fig = preview_wide_heatmap_inline(
        df=df,
        meta=meta,
        annotation_cols=annotation_cols,
        average_groups=average_groups
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()  # return bytes, not figure


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
If the algorithm doesn not converge try increasing maximum iteration parameter, in some cases by 2x or 3x times
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
    files = {
        "preprocessed": (
            "preprocessed.feather",
            st.session_state["preprocessed_feather"],
            "application/octet-stream",
        )
    }
    data = {"k": int(k), "max_iter": int(max_iter), "design_factor": "Group"}

    with st.spinner("Running NMF..."):
        try:
            r = requests.post(
                st.session_state["API_URL"] + "run_regular_nmf",
                files=files,
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
                        st.session_state["module_usages"] = pd.read_feather(z.open(name))
                    elif "_h" in name:
                        st.session_state["gene_loadings"] = pd.read_feather(z.open(name))
            if r.status_code == 200:
                st.success("NMF finished running. You can now visualize heatmaps")

        except Exception as e:
            st.error(f"Server error: {e}")

st.markdown("---")

# -------------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------------
if st.session_state["gene_loadings"] is not None and st.session_state["module_usages"] is not None:

    def toggle_display(key):
        st.session_state[key] = not st.session_state[key]

    # Gene loadings
    if st.session_state["display_loadings"]:
        st.subheader("NMF Gene Loadings")
        st.dataframe(st.session_state["gene_loadings"])
        st.button("Hide", on_click=toggle_display, args=("display_loadings",))
    else:
        st.button("Show Gene Loadings", on_click=toggle_display, args=("display_loadings",))

    # Module usages
    if st.session_state["display_scores"]:
        st.subheader("NMF Module Usage Scores")
        st.dataframe(st.session_state["module_usages"])
        st.button("Hide", on_click=toggle_display, args=("display_scores",))
    else:
        st.button("Show Usage Scores", on_click=toggle_display, args=("display_scores",))

    # Precompute transpose ONCE
    if "module_usages_T" not in st.session_state:
        st.session_state["module_usages_T"] = st.session_state["module_usages"].T

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
    st.session_state["annotations_default"] = annotation_cols
    common_samples = [s for s in meta[st.session_state["metadata_index"]] if s in df.columns]

    if not common_samples:
        st.warning("No overlapping samples.")
    else:
        df_ordered = df[common_samples]

        st.session_state["preview_png"] = cached_preview_png(
            df_ordered,
            meta,
            tuple(annotation_cols),
            average_groups
        )

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
if st.checkbox("Order by Top Samples"):
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

# EXPRESSION MATRIX
if st.checkbox("Show Gene Expression Matrix"):
    heatmap = get_expression_heatmap(st.session_state["gene_loadings"], st.session_state.get("annotations_default"))
    if heatmap is not None:
        st.session_state["expression_heatmap"] = heatmap

    if st.session_state["expression_heatmap"] is not None:
        st.image(st.session_state["expression_heatmap"])

# Continue to consensus NMF
st.page_link("pages/4_Run_Consensus_NMF.py", label="Continue")

