import streamlit as st
import requests, io, base64, zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
from ui_theme import apply_custom_theme
from clustering_heatmap import plot_heatmap_with_metadata
from nmf_clustering import plot_clusters
from preview_heatmap import preview_wide_heatmap_inline
from make_expression_heatmap import get_expression_heatmap
from hypergeometric import hypergeom_ui
from module_clustering import m_clustering
from module_heatmap import module_heatmap_ui
import json

apply_custom_theme()

# ============================================================================
# INITIAL SESSION STATE VARIABLES (CONSENSUS NMF ONLY)
# ============================================================================

API_URL_DEFAULT = "http://52.14.223.10:8000/"

st.session_state.setdefault("API_URL", API_URL_DEFAULT)
st.session_state.setdefault("nmf_running", False)
st.session_state.setdefault("nmf_zip_bytes", None)

# Thread executor & queue
if "nmf_queue" not in st.session_state:
    st.session_state["nmf_queue"] = queue.Queue()
if "executor" not in st.session_state:
    st.session_state["executor"] = ThreadPoolExecutor(max_workers=1)

# Canonical CNMF session state variables
defaults = {
    "cnmf_sample_order": None,
    "cnmf_gene_loadings": None,
    "cnmf_module_usages": None,
    "cnmf_display_scores": False,
    "cnmf_display_loadings": False,
    "cnmf_sample_dendogram": None,
    "cnmf_module_dendogram": None,
    "cnmf_sample_leaf_order": None,
    "cnmf_module_leaf_order": None,
    "cnmf_sample_order_heatmap": None,
    "cnmf_module_order_heatmap": None,
    "cnmf_top_order_heatmap": None,
    "cnmf_expression_heatmap": None,
    "cnmf_module_cluster_labels": None
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)


# ============================================================================
# HELPERS
# ============================================================================

def fig_to_png(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf


def read_table(raw: bytes, fname: str) -> pd.DataFrame:
    """Read a table and assign correctly to cnmf_module_usages or cnmf_gene_loadings."""
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", index_col=0)
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

    lower = fname.lower()

    # Module Usages (W matrix)
    if any(k in lower for k in ("usage", "usages", "_w")):
        df = df.reset_index().rename(columns={"index": "Sample"}).set_index("Sample")
        st.session_state["cnmf_module_usages"] = df

    # Gene Loadings (H matrix)
    elif any(k in lower for k in ("gene", "_h")):
        df = df.reset_index().rename(columns={"index": "Module"}).set_index("Module")
        st.session_state["cnmf_gene_loadings"] = df

    return df


def run_nmf_job(api_url, preprocessed_bytes, meta_bytes, post_data, q):
    """Background NMF worker."""
    try:
        files = {
            "preprocessed": ("preprocessed.feather", preprocessed_bytes, "application/octet-stream"),
            "metadata": ("metadata.tsv", meta_bytes, "text/csv"),
        }
        r = requests.post(f"{api_url}run_nmf_files", files=files, data=post_data, timeout=900)
        r.raise_for_status()
        q.put(("success", r.content))

    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            try:
                err_json = e.response.json()
                msg = err_json.get("message", "An unknown server error occurred.")
                eid = err_json.get("error_id", "N/A")
                details = err_json.get("details", "")
                q.put(("error", f"⚠️ {msg}\n\nError ID: `{eid}`\n{details}"))
            except ValueError:
                q.put(("error", e.response.text[:500]))
        else:
            q.put(("error", str(e)))

    except Exception as e:
        q.put(("error", str(e)))


# ============================================================================
# PAGE UI
# ============================================================================

st.markdown("---")
st.subheader("Run Consensus NMF with Selected Parameters")
st.markdown("""
Run consensus NMF following the method of Kotliar et al.  
Gene spectra (H matrix) are small because they are unit normalized across genes.
""")

# Validate required input data
if "preprocessed_feather" not in st.session_state or st.session_state["preprocessed_feather"] is None:
    st.error("Need to run preprocessing first or upload preprocessed data.")
    st.stop()

meta = st.session_state.get("meta")
if meta is None:
    st.error("Upload metadata first before proceeding with this step.")
    st.stop()


# ============================================================================
# NMF PARAM INPUTS
# ============================================================================

k = st.number_input("k", 2, 50, 7)
if st.session_state.get("integer_format", False):
    hvg = st.number_input("Number of highly variable genes", 100, 20000, 2000)
else:
    hvg = 10000

max_iter = st.number_input("Maximum number of iterations", 100, 20000, 5000)

# Preview metadata
st.write("**Metadata available:**")
st.dataframe(meta.head())

# Convert metadata → bytes
meta_buf = io.StringIO()
meta.to_csv(meta_buf, sep="\t", index=False)
meta_bytes = meta_buf.getvalue().encode("utf-8")


# ============================================================================
# RUN NMF BUTTON
# ============================================================================

if st.button("Run NMF") and not st.session_state["nmf_running"]:
    api_url = st.session_state.get("API_URL", API_URL_DEFAULT)
    preprocessed_bytes = st.session_state["preprocessed_feather"]
    metadata_index = st.session_state.get("metadata_index", "")

    post_data = {
        "k": int(k),
        "hvg": int(hvg),
        "max_iter": int(max_iter),
        "design_factor": "Group",
        "metadata_index": metadata_index,
    }

    st.session_state["executor"].submit(
        run_nmf_job, api_url, preprocessed_bytes, meta_bytes, post_data, st.session_state["nmf_queue"]
    )
    st.session_state["nmf_running"] = True
    st.info("NMF job started in background... You can navigate elsewhere.")


# ============================================================================
# POLL QUEUE
# ============================================================================

if st.session_state["nmf_running"]:
    st_autorefresh(interval=5000, key="nmf_autorefresh")
    try:
        status, payload = st.session_state["nmf_queue"].get_nowait()
        if status == "success":
            st.session_state["nmf_zip_bytes"] = payload
            st.success("NMF finished successfully!")
        else:
            st.error(f"NMF failed: {payload}")
        st.session_state["nmf_running"] = False
    except queue.Empty:
        st.info("NMF is still running…")


# ============================================================================
# LOAD ZIP RESULTS
# ============================================================================

zip_bytes = st.session_state.get("nmf_zip_bytes")
if not zip_bytes:
    st.info("No NMF results yet.")
    st.stop()

zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
file_names = [n for n in zf.namelist() if not n.endswith("/")]

st.markdown("---")
st.subheader("NMF Results")

for name in file_names:
    raw = zf.read(name)
    ext = Path(name).suffix.lower()
    short = Path(name).name

    if ext in {".txt", ".tsv", ".csv"}:
        df = read_table(raw, name)

    elif ext == ".pdf":
        continue

    else:
        st.markdown(f"**{short}**")
        try:
            st.code(raw[:2048].decode("utf-8", errors="replace"))
        except:
            st.write("(Preview not supported)")


# ============================================================================
# TOGGLE DISPLAY FOR W AND H MATRICES
# ============================================================================

def toggle_display(var):
    st.session_state[var] = not st.session_state[var]

# Gene Loadings
if st.session_state["cnmf_display_loadings"]:
    st.subheader("NMF Gene Loadings (H Matrix)")
    st.dataframe(st.session_state["cnmf_gene_loadings"])
    st.button("Hide Gene Loadings", on_click=toggle_display, args=("cnmf_display_loadings",))
else:
    st.button("Display Gene Loadings", on_click=toggle_display, args=("cnmf_display_loadings",))

# Module Usages
if st.session_state["cnmf_display_scores"]:
    st.subheader("NMF Module Usage Scores (W Matrix)")
    st.dataframe(st.session_state["cnmf_module_usages"])
    st.button("Hide Usage Scores", on_click=toggle_display, args=("cnmf_display_scores",))
else:
    st.button("Display Usage Scores", on_click=toggle_display, args=("cnmf_display_scores",))


# ============================================================================
# INTERACTIVE HEATMAP
# ============================================================================

st.subheader("Interactive Heatmap")

if st.session_state["cnmf_module_usages"] is not None:
    df = st.session_state["cnmf_module_usages"].T

    common_samples = [
        s for s in meta[st.session_state["metadata_index"]] if s in df.columns
    ]

    annotation_cols = st.multiselect(
        "Select metadata columns for annotation bars",
        options=meta.columns.tolist()
    )

    if not common_samples:
        st.warning("No overlapping sample names between metadata and module usages.")
    else:
        df = df[common_samples]

    average_groups = st.checkbox(
        "Average samples across unique combinations of metadata",
        value=False
    )

    fig = preview_wide_heatmap_inline(df=df, meta=meta,
                                      annotation_cols=annotation_cols,
                                      average_groups=average_groups)
    st.pyplot(fig)

    # PNG Download
    png_bytes = fig_to_png(fig)
    st.download_button("Download heatmap (PNG)",
                       data=png_bytes,
                       file_name="heatmap.png",
                       mime="image/png")

    # (Local PDF)
    # if you want backend PDF remove this section
    try:
        with open("heatmap.pdf", "rb") as f:
            st.download_button("Download full heatmap (PDF)",
                               f,
                               file_name="full_heatmap.pdf",
                               mime="application/pdf")
    except:
        pass


    # ============================================================================
    # HIERARCHICAL CLUSTERING OF SAMPLES
    # ============================================================================

    if st.checkbox("Hierarchically cluster samples"):

        def df_to_feather_bytes(df):
            buf = io.BytesIO()
            df.reset_index(drop=False).to_feather(buf)
            buf.seek(0)
            return buf

        def hex_to_png_bytes(hex_string):
            return bytes.fromhex(hex_string)

        module_bytes = df_to_feather_bytes(st.session_state["cnmf_module_usages"])
        meta_bytes_feather = df_to_feather_bytes(meta)
        metadata_index = st.session_state["metadata_index"]

        st.subheader("Hierarchical Clustering of Samples")
        k_samples = st.number_input("Number of clusters (samples)", 2, 10, 3)

        if st.button("Cluster Samples"):
            files = {
                "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                "metadata": ("meta.feather", meta_bytes_feather, "application/octet-stream"),
            }

            data = {
                "metadata_index": metadata_index,
                "k": str(k_samples),
            }

            res = requests.post(
                st.session_state["API_URL"] + "/cluster_samples/",
                files=files,
                data=data
            )

            if res.status_code != 200:
                st.error(res.text)

            else:
                payload = res.json()
                st.session_state["cnmf_sample_leaf_order"] = payload["leaf_order"]
                st.session_state["cnmf_sample_cluster_labels"] = payload["cluster_labels"]

                sample_order = df.T.index[st.session_state["cnmf_sample_leaf_order"]].tolist()
                st.session_state["cnmf_sample_order"] = sample_order

                st.session_state["cnmf_sample_dendogram"] = hex_to_png_bytes(payload["dendrogram_png"])

        # Show dendrogram
        if st.session_state["cnmf_sample_dendogram"] is not None:
            st.subheader("Sample Dendrogram")
            st.image(st.session_state["cnmf_sample_dendogram"])
            st.download_button(
                "Download PNG",
                data=st.session_state["cnmf_sample_dendogram"],
                file_name="sample_dendrogram.png",
                mime="image/png"
            )

        # ========================================================================
        # ANNOTATED HEATMAP
        # ========================================================================

        st.subheader("Annotated Heatmap")

        if st.session_state["cnmf_sample_leaf_order"] is None:
            st.info("Run clustering first.")
        else:
            tmp_cols = ["Cluster"] + meta.columns.tolist()
            annotation_cols_anno = st.multiselect(
                "Select metadata fields for annotation",
                options=tmp_cols
            )

            if st.button("Generate Annotated Heatmap"):
                files = {
                    "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                    "metadata": ("meta.feather", meta_bytes_feather, "application/octet-stream"),
                }

                data = {
                    "metadata_index": metadata_index,
                    "leaf_order": json.dumps(st.session_state["cnmf_sample_leaf_order"]),
                    "annotation_cols": json.dumps(annotation_cols_anno),
                    "cluster_labels": json.dumps(st.session_state.get("cnmf_sample_cluster_labels", [])),
                }

                res = requests.post(
                    st.session_state["API_URL"] + "/annotated_heatmap/",
                    files=files,
                    data=data
                )

                if res.status_code != 200:
                    st.error(res.text)
                else:
                    st.session_state["cnmf_sample_order_heatmap"] = res.content

            if st.session_state["cnmf_sample_order_heatmap"] is not None:
                st.image(st.session_state["cnmf_sample_order_heatmap"])
                st.download_button(
                    "Download PNG",
                    data=st.session_state["cnmf_sample_order_heatmap"],
                    file_name="sample_order_heatmap.png",
                    mime="image/png"
                )

            # ----------------------------------------------------------------------
            # Hypergeometric test
            # ----------------------------------------------------------------------
            if st.checkbox("Calculate Hypergeometric Values"):
                hypergeom_ui(meta_bytes_feather, st.session_state["cnmf_module_usages"], st.session_state["cnmf_sample_cluster_labels"])

            
            if st.checkbox("Hierarchically Cluster Modules"):
                st.header("📊 Cluster Modules (Rows)")
                n_clusters_mod = st.slider("Number of module clusters (k):", 2, 12, 4)

                if st.button("Run Module Clustering"):
                    dendro_png = m_clustering(
                        st.session_state["cnmf_module_usages"],
                        st.session_state["cnmf_sample_order"],
                        n_clusters_mod,
                        cnmf=True
                    )
                    if dendro_png:
                        st.session_state["cnmf_module_dendogram"] = dendro_png

                # Display saved image (persistent)
                if st.session_state["cnmf_module_dendogram"] is not None:
                    st.image(st.session_state["cnmf_module_dendogram"])
                    st.download_button(
                        "Download PNG",
                        data=st.session_state["cnmf_module_dendogram"],
                        file_name="cnmf_module_dendogram.png",
                        mime="image/png"
                    )

                
                module_heatmap_ui(
                        meta_bytes_feather,
                        st.session_state["cnmf_module_usages"],
                        st.session_state["cnmf_sample_order"],
                        st.session_state["cnmf_module_leaf_order"],
                        st.session_state["cnmf_module_cluster_labels"],
                        cnmf=True
                    )


    # ============================================================================
    # ORDER BY TOP SAMPLES
    # ============================================================================

    if st.checkbox("Order by Top Samples"):

        def df_to_feather_bytes(df):
            buf = io.BytesIO()
            df.reset_index(drop=False).to_feather(buf)
            buf.seek(0)
            return buf

        def hex_to_png_bytes(hex_string):
            return bytes.fromhex(hex_string)

        module_bytes = df_to_feather_bytes(st.session_state["cnmf_module_usages"])
        meta_bytes_feather = df_to_feather_bytes(meta)
        metadata_index = st.session_state["metadata_index"]

        annotation_cols_ts = st.multiselect(
            "Select metadata columns",
            options=meta.columns.tolist()
        )
        annotation_str = ",".join(annotation_cols_ts)

        if st.button("Generate Top-Ordered Heatmap"):

            files = {
                "module_usages": ("modules.feather", module_bytes, "application/octet-stream"),
                "metadata": ("meta.feather", meta_bytes_feather, "application/octet-stream"),
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
                st.session_state["cnmf_top_order_heatmap"] = hex_to_png_bytes(payload["heatmap_png"])

        if st.session_state["cnmf_top_order_heatmap"] is not None:
            st.image(st.session_state["cnmf_top_order_heatmap"])
            st.download_button(
                "Download PNG",
                data=st.session_state["cnmf_top_order_heatmap"],
                file_name="top_order_heatmap.png",
                mime="image/png"
            )

    # ============================================================================
    # EXPRESSION HEATMAP
    # ============================================================================

    if st.checkbox("Show Gene Expression Matrix"):
        heatmap = get_expression_heatmap(st.session_state["cnmf_gene_loadings"])
        if heatmap is not None:
            st.session_state["cnmf_expression_heatmap"] = heatmap

        if st.session_state["cnmf_expression_heatmap"] is not None:
            st.write("Black lines separate modules")
            st.image(st.session_state["cnmf_expression_heatmap"], caption="Generated Expression Heatmap")
            # Optional: Download button
            st.download_button(
                            "Download PNG",
                            data=st.session_state["cnmf_expression_heatmap"],
                            file_name="cnmf_gene_expression_heatmap.png",
                            mime="image/png"
                        )


# ============================================================================
# END OF PAGE → NAVIGATION
# ============================================================================

st.page_link("pages/5_Gene_Descriptions.py", label="Continue")
