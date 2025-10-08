import streamlit as st
import requests
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import zipfile
from preview_heatmap import preview_wide_heatmap_inline


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


# ---------------- Inputs ----------------
k = st.number_input("k", 2, 50, 7)
meta = None


max_iter = st.number_input("max_iter", 100, 20000, 5000)


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
        for name in z.namelist():
            if "_w" in name:
                with z.open(name) as f:
                    st.session_state["module_usages"] = pd.read_feather(io.BytesIO(f.read()))
            else:
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


# ---------------- Show results ----------------
if st.session_state["gene_loadings"] is not None and st.session_state["module_usages"] is not None:
    st.subheader("NMF Gene Loadings")
    st.dataframe(st.session_state["gene_loadings"])

    st.subheader("NMF Module Usage Scores")
    st.dataframe(st.session_state["module_usages"])

    # Transpose: modules x samples -> samples x modules
    df = st.session_state["module_usages"].T

    # --- Ensure metadata order ---
    if meta is not None:
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

    # --- Heatmap rendering ---
    if len(df.columns) < 50:
        plot_heatmap_inline(df)
    else:
        st.subheader("Wide Heatmap Preview (downsampled)")
        fig = preview_wide_heatmap_inline(df=df, meta=meta, annotation_cols=annotation_cols, step=5)
        st.pyplot(fig)
        
        png_bytes = fig_to_png(fig)

        st.download_button(
            label="Download heatmap (PNG)",
            data=png_bytes,
            file_name="heatmap.png",
            mime="image/png"
        )

        # Save and offer full PDF
        save_wide_heatmap_pdf(df, "heatmap.pdf")
        with open("heatmap.pdf", "rb") as f:
            st.download_button(
                "Download full heatmap (PDF)",
                f,
                file_name="heatmap.pdf",
                mime="application/pdf",
            )

if st.session_state["previous_heatmaps"]:
    st.subheader("Previous NMF Outputs")
    for k, fig in st.session_state["previous_heatmaps"].items():
        st.pyplot(fig)
        st.caption("K = " + str(k))

st.page_link("pages/4_Run_cNMF.py", label="Continue")