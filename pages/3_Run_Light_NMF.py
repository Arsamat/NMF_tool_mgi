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


if "API_URL" not in st.session_state:
    st.session_state["API_URL"] = "http://52.14.223.10:8000/"
if "preprocessed_feather" not in st.session_state:
    st.session_state["preprocessed_feather"] = None
if st.session_state["preprocessed_feather"] is None:
        st.error("Need to run preprocessing first or upload preprocessed data. Please, go back to preprocessing page to do so")
if "gene_loadings" not in st.session_state:
    st.session_state["gene_loadings"] = None
if "module_usages" not in st.session_state:
    st.session_state["module_usages"] = None
if "previous_heatmaps" not in st.session_state:
    st.session_state["previous_heatmaps"] = {}
if "sample_order" not in st.session_state:
    st.session_state["sample_order"] = None
if "display_scores" not in st.session_state:
    st.session_state["display_scores"] = False
if "display_loadings" not in st.session_state:
    st.session_state["display_loadings"] = False
if "display_previous_heatmaps" not in st.session_state:
    st.session_state["display_previous_heatmaps"] = False
if "sample_dendogram" not in st.session_state:
    st.session_state["sample_dendogram"] = None
if "module_dendogram" not in st.session_state:
    st.session_state["module_dendogram"] = None
if "sample_dendogram" not in st.session_state:
    st.session_state["sample_dendogram"] = None
if "sample_order_heatmap" not in st.session_state:
    st.session_state["sample_order_heatmap"] = None
if "module_dendogram" not in st.session_state:
    st.session_state["module_dendogram"] = None
if "module_order_heatmap" not in st.session_state:
    st.session_state["module_order_heatmap"] = None
if "top_order_heatmap" not in st.session_state:
    st.session_state["top_order_heatmap"] = None
if "expression_heatmap" not in st.session_state:
    st.session_state["expression_heatmap"] = None
if "module_leaf_order" not in st.session_state:
    st.session_state["module_leaf_order"] = None
if "module_cluster_labels" not in st.session_state:
    st.session_state["module_cluster_labels"] = None


st.subheader("Run NMF algorithm")
st.markdown('''
    Run NMF at your choice of k and generate a module usage heatmap. 
    This is useful as a preliminary validation of k before running the more computationally intensive consensus NMF.
    
    If the algorithm does not converge, try increasing maximum number of iterations. Sometimes by 2x or 3x times.
            ''')

# ---------------- Inputs ----------------
k = st.number_input("k", 2, 50, 7)
meta = None


max_iter = st.number_input("Maximum number of iterations", 100, 20000, 5000)


if "meta" not in st.session_state or st.session_state["meta"] is None:
    st.error("Upload metadata first before proceeding with this step")
else:
    meta = st.session_state["meta"]
    st.write("**Metadata available:**")
    st.dataframe(meta.head())


# ---------------- Helpers ----------------
def plot_heatmap_inline(df):
    """Inline heatmap for small sample counts."""
    plt.figure(figsize=(min(20, 0.5 * df.shape[1]), min(10, 0.5 * df.shape[0])))
    sns.heatmap(df, cmap="viridis", annot=False, cbar=True)
    plt.xlabel("Samples")
    plt.ylabel("Modules")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

def fig_to_png(fig, dpi=300):
    """Convert a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

def save_wide_heatmap_pdf(df, filename="heatmap.pdf"):
    """Save a single wide heatmap to PDF (scroll horizontally in viewer)."""
    # Cap figure size to avoid RendererAgg overflow
    fig_width = min(200, max(20, 0.15 * df.shape[1]))   # scale width, cap at 200 inches
    fig_height = min(50, max(6, 0.15 * df.shape[0]))    # scale height, cap at 50 inches

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(df, cmap="viridis", ax=ax, cbar=True)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Modules")
    plt.tight_layout()

    with PdfPages(filename) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


#function to read matrices from a zip file
def read_files(content):
    zip_bytes = io.BytesIO(content)

    with zipfile.ZipFile(zip_bytes, "r") as z:
        with z.open("metadata.json") as f:
            metadata = json.load(f)
        converged = metadata["converged"]
        if not converged:
            st.error("The algorithm didn't converge at a given number of iterations. Try increasing the parameter")

        for name in z.namelist():
            if "_w" in name:
                with z.open(name) as f:
                    st.session_state["module_usages"] = pd.read_feather(io.BytesIO(f.read()))
            elif "_h" in name:
                with z.open(name) as f:
                    st.session_state["gene_loadings"] = pd.read_feather(io.BytesIO(f.read()))



# ---------------- Run button ----------------
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
            # st.write(r.status_code)
            # st.write(r.headers)
            # st.text(r.text[:500])
            read_files(r.content)
        except requests.exceptions.HTTPError as e:
                # Try to extract the AI-generated summary from FastAPI JSON
                if e.response is not None:
                    try:
                        err_json = e.response.json()
                        user_msg = err_json.get("message", "An unknown server error occurred.")
                        error_id = err_json.get("error_id", "N/A")
                        details = err_json.get("details", "")
                        formatted = f"⚠️ **{user_msg}**\n\nError ID: `{error_id}`\n{details}"
                        st.error(formatted)
                    except ValueError:
                        # Not JSON (plain text or HTML)
                        st.error(f"Server error: {e.response.status_code}\n\n{e.response.text[:500]}")
                else:
                    st.error("error", f"Server error: {e}")

st.markdown("---")

# ---------------- Show results ----------------
if st.session_state["gene_loadings"] is not None and st.session_state["module_usages"] is not None:

    def toggle_display(var_name):
        st.session_state[var_name] = not st.session_state[var_name]

    # --- Gene Loadings Section ---
    if st.session_state.get("display_loadings", False):
        st.subheader("NMF Gene Loadings")
        st.dataframe(st.session_state["gene_loadings"])
        st.button("Hide Gene Loadings", on_click=toggle_display, args=("display_loadings",))
    else:
        st.button("Display Gene Loadings", on_click=toggle_display, args=("display_loadings",))

    
    # --- Module Usage Section ---
    if st.session_state.get("display_scores", False):
        st.subheader("NMF Module Usage Scores")
        st.dataframe(st.session_state["module_usages"])
        st.button("Hide Usage Scores", on_click=toggle_display, args=("display_scores",))
    else:
        st.button("Display Module Usage Scores", on_click=toggle_display, args=("display_scores",))

    # Transpose: samples x modules -> modules x samples
    df = st.session_state["module_usages"].T

# --- Heatmap rendering ---

    # --- Ensure metadata order ---
if meta is not None and st.session_state["module_usages"] is not None:
    st.subheader("Wide Heatmap Preview (downsampled)")
    common_samples = [s for s in meta[st.session_state["metadata_index"]] if s in df.columns]
    annotation_cols = st.multiselect(
        "Select metadata columns for annotation bars",
        options=meta.columns.tolist()
    )

    if not common_samples:
        st.warning("No overlapping sample names between metadata and module usages.")
        annotation_cols = None
    else:
        df = df[common_samples]  # reorder columns to match metadata index
    average_groups = st.checkbox(
        "Average samples across unique combinations of selected metadata variables (smooth heatmap)",
        value=False
    )

    fig = preview_wide_heatmap_inline(df=df, meta=meta, annotation_cols=annotation_cols, average_groups=average_groups)
    st.pyplot(fig)

    png_bytes = fig_to_png(fig)
    st.download_button("Download heatmap (PNG)", data=png_bytes, file_name="heatmap.png", mime="image/png")
    
    st.session_state["previous_heatmaps"][k] = fig
    png_bytes = fig_to_png(fig)

    # Save and offer full PDF
    save_wide_heatmap_pdf(df, "heatmap.pdf")
    with open("heatmap.pdf", "rb") as f:
        st.download_button(
            "Download full heatmap (PDF)",
            f,
            file_name="full_heatmap.pdf",
            mime="application/pdf",
            key="full_heatmap"
        )     
    if st.checkbox("Hierarchically cluster samples"):
        def hex_to_png_bytes(hex_string):
            return bytes.fromhex(hex_string)


        # ------------------------------------------------------
        # Load data from Streamlit session state
        # ------------------------------------------------------
        module_usages = st.session_state["module_usages"]   # DataFrame
        meta = st.session_state["meta"]                     # DataFrame
        metadata_index = st.session_state["metadata_index"] # e.g. "Sample"


        # Convert DataFrame → Feather bytes
        def df_to_feather_bytes(df):
            buf = io.BytesIO()
            df.reset_index(drop=False).to_feather(buf)
            buf.seek(0)
            return buf


        module_bytes = df_to_feather_bytes(module_usages)
        meta_bytes = df_to_feather_bytes(meta)

        # ------------------------------------------------------
        # 1️⃣ CLUSTER SAMPLES
        # ------------------------------------------------------
        st.header("Hierarchical Clustering of Samples")

        k_samples = st.number_input("Number of clusters (samples)", 2, 10, 3)

        if st.button("Cluster Samples"):
            files = {
                "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
            }

            data = {
                "metadata_index": metadata_index,
                "k": str(k_samples)
            }

            res = requests.post(st.session_state["API_URL"] + "/cluster_samples/", files=files, data=data)
            if res.status_code != 200:
                st.error(res.text)
            else:
                payload = res.json()

                st.session_state["sample_leaf_order"] = payload["leaf_order"]
                st.session_state["sample_cluster_labels"] = payload["cluster_labels"]
                
                sample_order = df.T.index[st.session_state["sample_leaf_order"]].tolist()
                st.session_state["sample_order"] = sample_order

                # Display dendrogram
                dendro_png = hex_to_png_bytes(payload["dendrogram_png"])
                st.session_state["sample_dendogram"] = dendro_png

                # Display heatmap
                #heatmap_png = hex_to_png_bytes(payload["heatmap_png"])
                #st.subheader("Heatmap Ordered by Sample Clustering")
                #st.image(heatmap_png)
        
        if st.session_state["sample_dendogram"] is not None:
            st.subheader("Sample Dendrogram")
            st.image(st.session_state["sample_dendogram"])
            st.download_button(
                        "Download PNG",
                        data=st.session_state["sample_dendogram"],
                        file_name="sample_dendogram.png",
                        mime="image/png"
                    )
        # ------------------------------------------------------
        # 2️⃣ ANNOTATED HEATMAP
        # ------------------------------------------------------
        st.header("Annotated Heatmap")

        if "sample_leaf_order" not in st.session_state:
            st.info("Run Step 1 first")
        else:
            tmp = meta.columns.tolist()
            tmp = ["Cluster"] + tmp
            
            annotation_cols = st.multiselect(
                "Select metadata fields for annotation bars",
                options=tmp
            )

            if st.button("Generate Annotated Heatmap"):
                files = {
                    "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                    "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
                }
                
                data = {
                    "metadata_index": metadata_index,
                    "leaf_order": json.dumps(st.session_state["sample_leaf_order"]),
                    "annotation_cols": json.dumps(annotation_cols),
                    "cluster_labels": json.dumps(st.session_state["sample_cluster_labels"])
                }

                res = requests.post(st.session_state["API_URL"] + "/annotated_heatmap/", files=files, data=data)
                if res.status_code != 200:
                    st.error(res.text)
                else:
                    st.subheader("Annotated Heatmap")
                    st.session_state["sample_order_heatmap"] = res.content
            
            if st.session_state["sample_order_heatmap"] is not None:
                st.image(st.session_state["sample_order_heatmap"])
                st.download_button(
                        "Download PNG",
                        data=st.session_state["sample_order_heatmap"],
                        file_name="sample_order_heatmap.png",
                        mime="image/png"
                    )

            if st.checkbox("Calculate Hypergeometric Values"):
                hypergeom_ui(meta_bytes, st.session_state["module_usages"], st.session_state["sample_cluster_labels"])

            if st.checkbox("Hierarchically Cluster Modules"):
                st.header("📊 Cluster Modules (Rows)")
                n_clusters_mod = st.slider("Number of module clusters (k):", 2, 12, 4)

                if st.button("Run Module Clustering"):
                    dendro_png = m_clustering(
                        st.session_state["module_usages"],
                        st.session_state["sample_order"],
                        n_clusters_mod
                    )
                    if dendro_png:
                        st.session_state["module_dendogram"] = dendro_png

                # Display saved image (persistent)
                if  st.session_state["module_dendogram"] is not None:
                    st.image(st.session_state["module_dendogram"])
                    st.download_button(
                        "Download PNG",
                        data=st.session_state["module_dendogram"],
                        file_name="module_dendogram.png",
                        mime="image/png"
                    )
                
                #Get Module Heatmap
                module_heatmap_ui(
                            meta_bytes,
                            st.session_state["module_usages"],
                            st.session_state["sample_order"],
                            st.session_state["module_leaf_order"],
                            st.session_state["module_cluster_labels"],
                            cnmf=False
                        )

            
    if st.checkbox("Order by Top Samples"):
        def df_to_feather_bytes(df):
            buf = io.BytesIO()
            df.reset_index(drop=False).to_feather(buf)
            buf.seek(0)
            return buf

        def hex_to_png_bytes(hex_string):
            return bytes.fromhex(hex_string)

        module_bytes = df_to_feather_bytes(st.session_state["module_usages"])
        meta_bytes   = df_to_feather_bytes(st.session_state["meta"])
        metadata_index = st.session_state["metadata_index"]

        annotation_cols = st.multiselect(
            "Select metadata columns",
            options=st.session_state["meta"].columns.tolist()
        )
        annotation_str = ",".join(annotation_cols)

        if st.button("Generate Top-Ordered Heatmap"):

            files = {
                "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
            }

            data = {
                "metadata_index": metadata_index,
                "annotation_cols": annotation_str,
            }

            res = requests.post(
                st.session_state["API_URL"] + "/heatmap_top_samples/",
                files=files,
                data=data
            )

            if res.status_code != 200:
                st.error(res.text)
            else:
                payload = res.json()

                heatmap_png = hex_to_png_bytes(payload["heatmap_png"])
                st.session_state["top_order_heatmap"] = heatmap_png
        
        if st.session_state["top_order_heatmap"] is not None:
            st.image(st.session_state["top_order_heatmap"])
            st.download_button(
                        "Download PNG",
                        data=st.session_state["top_order_heatmap"],
                        file_name="top_order_heatmap.png",
                        mime="image/png"
                    )

    
    if st.checkbox("Show Gene Expression Matrix"):
        heatmap = get_expression_heatmap(st.session_state["gene_loadings"])
        if heatmap is not None:
            st.session_state["expression_heatmap"] = heatmap
            
        if st.session_state["expression_heatmap"] is not None:
            st.write("Black lines separate modules")
            st.image(st.session_state["expression_heatmap"])
            # Optional: Download button
            st.download_button(
                            "Download PNG",
                            data=st.session_state["expression_heatmap"],
                            file_name="gene_expression_heatmap.png",
                            mime="image/png"
                        )


if st.session_state["previous_heatmaps"]:
    st.subheader("Previous NMF Outputs")
    if st.session_state["display_previous_heatmaps"]:
        if st.button("Hide Heatmaps"):
            st.session_state["display_previous_heatmaps"] = False
            st.rerun()
            
        for k, fig in st.session_state["previous_heatmaps"].items():
            st.pyplot(fig)
            st.caption("K = " + str(k))
    else:
        if st.button("Show previous heatmaps"):
            st.session_state["display_previous_heatmaps"] = True
            st.rerun()
    

st.page_link("pages/4_Run_Consensus_NMF.py", label="Continue")
