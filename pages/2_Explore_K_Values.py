import streamlit as st
import requests
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt   # <-- make sure pyplot is imported here too
from anova_plots import make_permanova_boxplot
import io
import json


#if "API_URL" not in st.session_state:
  #  st.session_state["API_URL"] = "http://3.141.231.76:8000/"
if "preprocessed_feather" not in st.session_state:
    st.session_state["preprocessed_feather"] = None

if "silhouette_data" not in st.session_state:
    st.session_state["silhouette_data"] = None

if st.session_state["preprocessed_feather"] is None:
        st.error("Need to run preprocessing first or upload preprocessed data. Please, go back to preprocessing page to do so")

st.title("Run K Metrics")

#meta_up = st.file_uploader("Upload metadata", type=["csv", "tsv", "txt"])
ks = st.slider("Select range of K values", 2, 50, (2, 10))
max_iter = st.number_input("max_iter", 100, 20000, 5000)
design_factor = st.text_input("Design factor", "Group")
#sample_column = st.text_input("Column name that stores sample names")

#HELPER Function
def fig_to_png(fig, dpi=300):
    """Convert a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf


if st.button("Run analysis"):
    meta_up = st.session_state["meta"]
    sample_column = st.session_state["metadata_index"]
    if st.session_state["preprocessed_feather"] is None:
        st.error("Need preprocessed feather in session_state['preprocessed_feather']")
    elif meta_up is None:
        st.error("Please upload metadata first.")
    else:
        with st.spinner("Crunching numbers..."):
            meta_buf = io.StringIO()
            st.session_state["meta"].to_csv(meta_buf, sep="\t", index=False)
            meta_bytes = meta_buf.getvalue().encode("utf-8")
            files = {
                "preprocessed_feather": (
                    "preprocessed.feather",
                    st.session_state["preprocessed_feather"],
                    "application/octet-stream",
                ),
                "meta": ("metadata.tsv", meta_bytes, "text/csv"),
            }
            data = {
                "k_min": ks[0],
                "k_max": ks[1],
                "max_iter": max_iter,
                "design_factor": design_factor,
                "sample_column": sample_column
            }

            try: 
                r = requests.post(
                    st.session_state["API_URL"] + "k_metrics",
                    files=files,
                    data=data,
                    timeout=600
                )
                r.raise_for_status()
                st.session_state["silhouette_data"] = r.json()
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
                

if st.session_state.get("silhouette_data"):
    # Load results into DataFrame
    df = pd.DataFrame(st.session_state["silhouette_data"]["results"])
    if not df.empty:
        df
        # Create figure
        fig, ax = plt.subplots(figsize=(8,6))

        # Scatter: one dot per replicate
        ax.scatter(df["k"], df["silhouette"], alpha=0.7, label="Replicates")

        # Mean silhouette per K
        means = df.groupby("k")["silhouette"].mean()
        ax.plot(means.index, means.values, color="red", marker="o", label="Mean silhouette")

        # Labels and style
        ax.set_xlabel("K")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette scores across K (10 replicates each)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Display in Streamlit
        st.pyplot(fig)

        png_bytes = fig_to_png(fig)

        st.download_button(
                label="Download Silhouette score (PNG)",
                data=png_bytes,
                file_name="Silhouette.png",
                mime="image/png"
        )
            

st.page_link("pages/3_Run_Light_NMF.py", label="Continue")

            # df_perm = pd.read_csv(StringIO(r.text), sep="\t")

            # # Normalize col names
            # df_perm.columns = df_perm.columns.str.strip().str.lower()

            # st.subheader("PERMANOVA Results (Raw)")
            # st.dataframe(df_perm)

            # # ---- Plotting ----
            # if "k" in df_perm.columns and "r2_adj" in df_perm.columns:
            #     k_sorted = sorted(df_perm["k"].unique())
            #     fig_perm = make_permanova_boxplot(df_perm, k_sorted)
            #     st.subheader("PERMANOVA r2_adj Distribution")
            #     st.pyplot(fig_perm)
            # else:
            #     st.warning(f"Columns available: {list(df_perm.columns)}")

            # # ---- Download button ----
            # st.download_button(
            #     label="Download Results (TSV)",
            #     data=r.content,
            #     file_name="permanova_summary_all.tsv",
            #     mime="text/tab-separated-values"
            # )
