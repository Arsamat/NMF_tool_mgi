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
from clustering_heatmap import plot_heatmap_with_metadata, plot_heatmap_clustered_modules
from nmf_clustering import plot_clusters, plot_module_clusters
from hypergeometric_test import hypergeometric
from expression_heatmap import plot_expression_heatmap
apply_custom_theme()

# =========================
# Safe initializations
# =========================
API_URL_DEFAULT = "http://52.14.223.10:8000/"

st.session_state.setdefault("API_URL", API_URL_DEFAULT)
st.session_state.setdefault("nmf_running", False)
st.session_state.setdefault("nmf_zip_bytes", None)

# Create once per session (not per rerun)
if "nmf_queue" not in st.session_state:
    st.session_state["nmf_queue"] = queue.Queue()
if "executor" not in st.session_state:
    st.session_state["executor"] = ThreadPoolExecutor(max_workers=1)
if "cnmf_sample_order" not in st.session_state:
    st.session_state["cnmf_sample_order"] = None
if "cnmf_gene_loadings" not in st.session_state:
    st.session_state["cnmf_gene_loadings"] = None
if "cnmf_module_usages" not in st.session_state:
    st.session_state["cnmf_module_usages"] = None
if "cnmf_display_scores" not in st.session_state:
    st.session_state["cnmf_display_scores"] = False
if "cnmf_display_loadings" not in st.session_state:
    st.session_state["cnmf_display_loadings"] = False

# =========================
# Helpers
# =========================
def fig_to_png(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

def read_table(raw: bytes, fname: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", index_col=0)
    except Exception:
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

    label = df.index.name or "Label"
    lower = fname.lower()
    if any(k in lower for k in ("usage", "usages", "_w")):
        label = "Sample"
        df = df.reset_index().rename(columns={"index": label})
        df = df.set_index("Sample")
        st.session_state["cnmf_module_usages"] = df
    elif any(k in lower for k in ("gene", "_h")):
        label = "Module"
        df = df.reset_index().rename(columns={"index": label})
        df = df.set_index("Module")
        st.session_state["cnmf_gene_loadings"] = df
    return df

def run_nmf_job(api_url, preprocessed_bytes, meta_bytes, post_data, q):
    """Background worker: do NOT read st.session_state here."""
    try:
        files = {
            "preprocessed": ("preprocessed.feather", preprocessed_bytes, "application/octet-stream"),
            "metadata": ("metadata.tsv", meta_bytes, "text/csv"),
        }
        r = requests.post(f"{api_url}run_nmf_files", files=files, data=post_data, timeout=900)
        r.raise_for_status()
        q.put(("success", r.content))
    except requests.exceptions.HTTPError as e:
        # Try to extract the AI-generated summary from FastAPI JSON
        if e.response is not None:
            try:
                err_json = e.response.json()
                user_msg = err_json.get("message", "An unknown server error occurred.")
                error_id = err_json.get("error_id", "N/A")
                details = err_json.get("details", "")
                formatted = f"⚠️ **{user_msg}**\n\nError ID: `{error_id}`\n{details}"
                q.put(("error", formatted))
            except ValueError:
                # Not JSON (plain text or HTML)
                q.put(("error", f"Server error: {e.response.status_code}\n\n{e.response.text[:500]}"))
        else:
            q.put(("error", f"Server error: {e}"))
    except Exception as e:
        q.put(("error", str(e)))

# =========================
# Page UI
# =========================
st.markdown("---")
st.subheader("Run consensus NMF with selected parameters")
st.markdown('''
    Run consensus NMF following the method of Kotliar et al. 
    Gene spectra z-scores (gene contributions to each module) are expected to be very small, 
    as they are unit normalized across scores for all genes included in the analysis.      

''')

# Validate required inputs
if "preprocessed_feather" not in st.session_state or st.session_state["preprocessed_feather"] is None:
    st.error("Need to run preprocessing first or upload preprocessed data. Please go back to the preprocessing page.")
    st.stop()

meta = st.session_state.get("meta")
if meta is None:
    st.error("Upload metadata first before proceeding with this step.")
    st.stop()

# Inputs
k = st.number_input("k", 2, 50, 7)
if st.session_state["integer_format"]:
    hvg = st.number_input("Number of highly variable genes to include", 100, 20000, 2000)
else:
    hvg = 10000
max_iter = st.number_input("Maximum number of iterations", 100, 20000, 5000)

# Show meta preview & build bytes once (safe to pass into thread)
st.write("**Metadata available:**")
st.dataframe(meta.head())
meta_buf = io.StringIO()
meta.to_csv(meta_buf, sep="\t", index=False)
meta_bytes = meta_buf.getvalue().encode("utf-8")

# ---------------- Run button -> Start background thread safely ----------------
if st.button("Run NMF") and not st.session_state["nmf_running"]:
    api_url = st.session_state.get("API_URL", API_URL_DEFAULT)           # copy now
    preprocessed_bytes = st.session_state["preprocessed_feather"]        # copy now
    metadata_index = st.session_state.get("metadata_index", None)        # copy now

    post_data = {
        "k": int(k),
        "hvg": int(hvg),
        "max_iter": int(max_iter),
        "design_factor": "Group",
        "metadata_index": metadata_index if metadata_index is not None else "",
    }

    st.session_state["executor"].submit(
        run_nmf_job,
        api_url,
        preprocessed_bytes,
        meta_bytes,
        post_data,
        st.session_state["nmf_queue"],
    )
    st.session_state["nmf_running"] = True
    st.info("NMF job started in background... you can navigate elsewhere.")


# ---------------- Poll the queue for completion ----------------
if st.session_state["nmf_running"]:
    st_autorefresh(interval=5000, key="nmf_autorefresh")
    try:
        status, payload = st.session_state["nmf_queue"].get_nowait()
        if status == "success":
            st.session_state["nmf_zip_bytes"] = payload
            st.success("NMF finished successfully!")
            st.session_state["nmf_running"] = False
        else:
            st.error(f"NMF failed: {payload}")
            st.session_state["nmf_running"] = False
    except queue.Empty:
        st.info("NMF is still running...")

# ---------------- Display results & heatmap when ready ----------------
zip_bytes = st.session_state.get("nmf_zip_bytes")
if not zip_bytes:
    st.info("No NMF results yet. Click **Run NMF** to generate them.")
    st.stop()

zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
names = [n for n in zf.namelist() if not n.endswith("/")]

st.markdown("---")
st.subheader("NMF Results")
for name in names:
    raw = zf.read(name)
    ext = Path(name).suffix.lower()
    short = Path(name).name

    if ext == ".pdf":
        # st.write("Download full heatmap as a pdf with the link below")
        # st.download_button(
        #     f"Download {short}",
        #     data=raw,
        #     file_name=short,
        #     mime=("application/pdf" if ext == ".pdf" else "application/octet-stream"),
        #     key=f"dl_{name}",
        # )
        continue
    
    if ext in {".txt", ".tsv", ".csv"}:
        df = read_table(raw, name)
        #st.markdown(f"**{short}**")
        #st.dataframe(df, use_container_width=True, height=500)
    else:
        st.markdown(f"**{short}**")
        try:
            st.code(raw[:2048].decode("utf-8", errors="replace"))
        except Exception:
            st.write("(Preview not supported)")
 
    # st.download_button(
    #         f"Download {short}",
    #         data=raw,
    #         file_name=short,
    #         mime=("application/pdf" if ext == ".pdf" else "application/octet-stream"),
    #         key=f"dl_{name}",
    #     )


def toggle_display(var_name):
        st.session_state[var_name] = not st.session_state[var_name]

# --- Gene Loadings Section ---
if st.session_state.get("cnmf_display_loadings", False):
    st.subheader("NMF Gene Loadings")
    st.dataframe(st.session_state["cnmf_gene_loadings"])
    st.button("Hide Gene Loadings", on_click=toggle_display, args=("cnmf_display_loadings",))
else:
    st.button("Display Gene Loadings", on_click=toggle_display, args=("cnmf_display_loadings",))


# --- Module Usage Section ---
if st.session_state.get("cnmf_display_scores", False):
    st.subheader("NMF Module Usage Scores")
    st.dataframe(st.session_state["cnmf_module_usages"])
    st.button("Hide Usage Scores", on_click=toggle_display, args=("cnmf_display_scores",))
else:
    st.button("Display Module Usage Scores", on_click=toggle_display, args=("cnmf_display_scores",))
# ---------------- Interactive Heatmap ----------------
st.subheader("Interactive Heatmap")

usage_files = [n for n in names if "usage" in n.lower() or n.lower().endswith("_w.txt")]

if st.session_state["cnmf_module_usages"] is not None and meta is not None:
    df = st.session_state["cnmf_module_usages"].T
    
    
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
    
    with open("heatmap.pdf", "rb") as f:
        st.download_button(
            "Download full heatmap (PDF)",
            f,
            file_name="full_heatmap.pdf",
            mime="application/pdf",
            key="full_heatmap"
        )
    
    if st.checkbox("Hierarchically cluster samples"):
            leaf_order, cluster_labels = plot_clusters(st.session_state["cnmf_module_usages"])
            
            # Save order persistently
            if leaf_order is not None:
                # Convert from integer indices → actual column names
                st.session_state["cnmf_sample_order"] = (
                    st.session_state["cnmf_module_usages"].index[leaf_order]
                )
            
            if st.session_state.get("cnmf_sample_order") is not None:
                st.markdown("### Heatmap ordered by clustering")
                fig = plot_heatmap_with_metadata(
                    st.session_state["cnmf_module_usages"].T,  
                    meta,
                    st.session_state["cnmf_sample_order"],
                    cluster_labels
                )
                png_bytes = fig_to_png(fig)
                st.download_button("Download heatmap (PNG)", data=png_bytes, file_name="heatmap_clustered_samples.png", mime="image/png")
                if st.checkbox("Perform Hypergeometric test on clusters"):
                    hypergeometric(st.session_state["cnmf_module_usages"], st.session_state["meta"], cluster_labels)
            
            if st.checkbox("Hierarchically Cluster Rows"):
                leaf_order2, cluster_labels2 = plot_module_clusters(st.session_state["cnmf_module_usages"].T)

                annotation_cols3 = st.multiselect(
                    "Select metadata columns for annotation bars",
                    options=meta.columns.tolist(),
                    key="annotation_cols3"
                )
                fig = plot_heatmap_clustered_modules(
                    df=st.session_state["cnmf_module_usages"].T,        # modules x samples
                    meta=st.session_state["meta"],              # metadata dataframe
                    annotation_cols=annotation_cols3,     # overlay these
                    module_leaf_order=leaf_order2,                # from clustering
                    cluster_labels = cluster_labels2
                )
                png_bytes = fig_to_png(fig)
                st.download_button("Download heatmap (PNG)", data=png_bytes, file_name="heatmap_clustered_modules.png", mime="image/png")
            
    if st.checkbox("Order by Top Samples"):
        df_usage = st.session_state["cnmf_module_usages"].copy()

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
            subset = subset.sort_values(by=module, ascending=False)  # right->left decrease expression
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
        fig = plot_expression_heatmap(st.session_state["cnmf_gene_loadings"], st.session_state["preprocessed_feather"])
        png_bytes = fig_to_png(fig)
        st.download_button("Download heatmap (PNG)", data=png_bytes, file_name="heatmap_gene_expression.png", mime="image/png")

st.page_link("pages/5_Gene_Descriptions.py", label="Continue")