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
from clustering_heatmap import plot_heatmap_with_metadata, plot_heatmap_clustered_modules
from nmf_clustering import plot_clusters, plot_module_clusters
from hypergeometric_test import hypergeometric
from expression_heatmap import plot_expression_heatmap
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
if "show_clusters" not in st.session_state:
    st.session_state["show_clusters"] = False


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


# def preview_wide_heatmap_inline(df, step=5):
#     """Show a safe preview by downsampling columns for Streamlit rendering."""
#     df_small = df.iloc[:, ::step]  # take every nth sample
#     fig, ax = plt.subplots(figsize=(20, 8))
#     sns.heatmap(df_small, cmap="viridis", ax=ax, cbar=True)
#     ax.set_xlabel(f"Samples (every {step}th shown)")
#     ax.set_ylabel("Modules")
#     #add to store for results history
#     k = len(df.index)
#     if k not in st.session_state["previous_heatmaps"]:
#         st.session_state["previous_heatmaps"][k] = fig

#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

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
        leaf_order, cluster_labels = plot_clusters(st.session_state["module_usages"])
        
        # Save order persistently
        if leaf_order is not None:
            # Convert from integer indices → actual column names
            st.session_state["sample_order"] = (
                st.session_state["module_usages"].index[leaf_order]
            )
        
        if st.session_state.get("sample_order") is not None:
            st.markdown("### Heatmap ordered by clustering")
            fig = plot_heatmap_with_metadata(
                st.session_state["module_usages"].T,  
                meta,
                st.session_state["sample_order"],
                cluster_labels
            )
            png_bytes = fig_to_png(fig)
            st.download_button("Download heatmap (PNG)", data=png_bytes, file_name="heatmap_clustered_samples.png", mime="image/png")

            if st.checkbox("Perform Hypergeometric test on clusters"):
                hypergeometric(st.session_state["module_usages"], st.session_state["meta"], cluster_labels)
            if st.checkbox("Hierarchically Cluster Rows"):
                leaf_order2, cluster_labels2 = plot_module_clusters(st.session_state["module_usages"].T)
                if st.session_state["sample_order"] is not None:
                    tmp = st.session_state["module_usages"].T
                    tmp = tmp[st.session_state["sample_order"]]

                annotation_cols3 = st.multiselect(
                    "Select metadata columns for annotation bars",
                    options=meta.columns.tolist(),
                    key="annotation_cols3"
                )
                fig = plot_heatmap_clustered_modules(
                    df=tmp,        # modules x samples
                    meta=st.session_state["meta"],              # metadata dataframe
                    annotation_cols=annotation_cols3,     # overlay these
                    module_leaf_order=leaf_order2,                # from clustering
                    cluster_labels = cluster_labels2
                )
                png_bytes = fig_to_png(fig)
                st.download_button("Download heatmap (PNG)", data=png_bytes, file_name="heatmap_clustered_modules.png", mime="image/png")
            
    if st.checkbox("Order by Top Samples"):
        df_usage = st.session_state["module_usages"].copy()

        # Step 1: Assign each sample to its top module
        sample_assignments = df_usage.idxmax(axis=1)
        df_usage["TopModule"] = sample_assignments
        # Step 2: Sort modules numerically (not alphabetically)
        def numeric_module_key(name):
            try:
                return int(name.split("_")[-1])
            except Exception:
                return name  # fallback if non-standard

        modules_sorted = sorted(sample_assignments.unique(), key=numeric_module_key)

        
        # Step 3: Build ordered sample list (within each module, sort by ascending usage)
        ordered_samples = []
        for module in modules_sorted:
            subset = df_usage[df_usage["TopModule"] == module]
            subset = subset.sort_values(by=module, ascending=False)  # right->left decrease
            ordered_samples.extend(subset.index.tolist())

        # Step 4: Reformat for heatmap (modules × samples)
        df_for_heatmap = df_usage.drop(columns=["TopModule"]).T

        annotation_cols2 = st.multiselect(
            "Select metadata columns for annotation bars",
            options=meta.columns.tolist(), key="top_sample_order"
        )

        # Step 5: Plot heatmap using the same function
        fig = preview_wide_heatmap_inline(
            df=df_for_heatmap,
            meta=meta,
            annotation_cols=annotation_cols2,
            average_groups=False,
            sample_order=ordered_samples
        )
        st.pyplot(fig)
        png_bytes = fig_to_png(fig)
        st.download_button("Download heatmap (PNG)", data=png_bytes, file_name="heatmap_ordered_top_samples.png", mime="image/png")
    
    if st.checkbox("Show Gene Expression Matrix"):
        fig = plot_expression_heatmap(st.session_state["gene_loadings"], st.session_state["preprocessed_feather"])
        png_bytes = fig_to_png(fig)
        st.download_button("Download heatmap (PNG)", data=png_bytes, file_name="heatmap_gene_expression.png", mime="image/png")


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