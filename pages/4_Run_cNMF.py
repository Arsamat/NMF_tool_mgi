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
    elif any(k in lower for k in ("gene", "_h")):
        label = "Module"
        df = df.reset_index().rename(columns={"index": label})
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
st.subheader("Run NMF with selected parameters")

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
hvg = st.number_input("hvg", 100, 20000, 2000)
max_iter = st.number_input("max_iter", 100, 20000, 5000)

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
        st.write("Download full heatmap as a pdf with the link below")
        st.download_button(
            f"Download {short}",
            data=raw,
            file_name=short,
            mime=("application/pdf" if ext == ".pdf" else "application/octet-stream"),
            key=f"dl_{name}",
        )
        continue
    
    if ext in {".txt", ".tsv", ".csv"}:
        df = read_table(raw, name)
        st.markdown(f"**{short}**")
        st.dataframe(df, use_container_width=True, height=500)
    else:
        st.markdown(f"**{short}**")
        try:
            st.code(raw[:2048].decode("utf-8", errors="replace"))
        except Exception:
            st.write("(Preview not supported)")
    
    st.download_button(
            f"Download {short}",
            data=raw,
            file_name=short,
            mime=("application/pdf" if ext == ".pdf" else "application/octet-stream"),
            key=f"dl_{name}",
        )

# ---------------- Interactive Heatmap ----------------
st.subheader("Interactive Heatmap")

usage_files = [n for n in names if "usage" in n.lower() or n.lower().endswith("_w.txt")]
if usage_files:
    usage_file = st.selectbox("Select usages file", usage_files)
    raw_usage = zf.read(usage_file)
    df_usage = read_table(raw_usage, usage_file)
    if "Sample" not in df_usage.columns:
        st.error("Usages table must have a 'Sample' column.")
        st.stop()

    df_usage = df_usage.set_index("Sample")

    # metadata join & annotation selection
    meta_index_col = st.session_state.get("metadata_index")
    if meta_index_col is None or meta_index_col not in meta.columns:
        st.warning("metadata_index not set or missing in metadata; using intersection only.")
        meta_indexed = meta.set_index(meta.columns[0])
    else:
        meta_indexed = meta.set_index(meta_index_col)

    meta_wo_index = meta.drop(columns=[c for c in [meta_index_col] if c in meta.columns])
    annot_cols = st.multiselect(
        "Select metadata columns to annotate by:",
        meta_wo_index.columns.tolist(),
        default=[]
    )

    # order by metadata index intersection
    common_samples = [s for s in list(meta_indexed.index) if s in df_usage.index]
    if len(common_samples) == 0:
        st.error("No overlapping samples between usage matrix and metadata.")
        st.stop()
    df_usage_ordered = df_usage.loc[common_samples]

    # colors
    col_colors = pd.DataFrame(index=df_usage_ordered.index)
    lut = {}
    palettes = ["Set1", "Set2", "Paired", "Pastel1", "Dark2"]
    for i, col in enumerate(annot_cols):
        unique_vals = meta_wo_index[col].dropna().unique()
        palette = sns.color_palette(palettes[i % len(palettes)], n_colors=len(unique_vals))
        lut[col] = dict(zip(unique_vals, palette))
        col_colors[col] = meta_indexed.loc[df_usage_ordered.index, col].map(lut[col])

    # heatmap matrix
    df_for_heatmap = df_usage_ordered.T.copy()
    try:
        df_for_heatmap = df_for_heatmap.reindex(
            sorted(df_for_heatmap.index, key=lambda x: int(x.split("_")[-1]))
        )
    except Exception:
        pass  # keep original order if module names aren't "Module_#"

    g = sns.clustermap(
        df_for_heatmap,
        col_colors=col_colors if not col_colors.empty else None,
        cmap="viridis",
        figsize=(16, 10),
        col_cluster=False,
        row_cluster=False
    )

    # legend
    for col in annot_cols:
        for label in lut[col]:
            g.ax_col_dendrogram.bar(0, 0, color=lut[col][label], label=f"{col}: {label}", linewidth=0)
    if annot_cols:
        g.ax_col_dendrogram.legend(loc="center", ncol=2, bbox_to_anchor=(0.5, 1.1))

    st.pyplot(g)
    png_bytes = fig_to_png(g.figure)
    st.download_button("Download heatmap (PNG)", data=png_bytes, file_name="heatmap.png", mime="image/png")
else:
    st.info("No usages file found in results.")

st.page_link("pages/5_Gene_Descriptions.py", label="Continue")