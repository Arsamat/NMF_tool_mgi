import streamlit as st
import requests
import json
from io import BytesIO

def module_heatmap_ui(
        meta_bytes,
        module_usages,
        sample_order,
        module_leaf_order,
        module_cluster_labels,
        cnmf=False
):
    """
    One function with UI + backend call.
    Persistent display. Supports NMF and CNMF based on `cnmf` flag.
    """

    st.header("📊 Module Heatmap with Column Annotations")

    # --- Safety checks ---
    if module_usages is None:
        st.warning("Run NMF first.")
        return
    if sample_order is None:
        st.warning("Cluster samples first.")
        return
    if module_leaf_order is None:
        st.warning("Cluster modules first.")
        return
    if module_cluster_labels is None:
        st.warning("Cluster modules first.")
        return

    meta = st.session_state["meta"]

    annotation_cols = st.multiselect(
        "Select sample metadata to annotate:",
        options=meta.columns.tolist(),
        key=("cnmf_ann_cols" if cnmf else "nmf_ann_cols")
    )

    # ---------- RUN ONLY WHEN BUTTON CLICKED ----------
    if st.button("Generate Module Heatmap", key=("cnmf_heatmap_btn" if cnmf else "nmf_heatmap_btn")):

        # prepare module usages
        buf_mod = BytesIO()
        module_usages.reset_index(drop=False).to_feather(buf_mod)
        buf_mod.seek(0)

        files = {
            "module_usages": ("module.feather", buf_mod, "application/octet-stream"),
            "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
        }

        data = {
            "metadata_index": st.session_state["metadata_index"],
            "sample_order": json.dumps(sample_order),
            "module_leaf_order": json.dumps(module_leaf_order),
            "module_cluster_labels": json.dumps(module_cluster_labels),
            "annotation_cols": json.dumps(annotation_cols),
        }

        url = st.session_state["API_URL"] + "/module_heatmap/"
        resp = requests.post(url, data=data, files=files)

        if resp.status_code != 200:
            st.error(resp.text)
            return

        result = resp.json()
        png_bytes = BytesIO(bytes.fromhex(result["heatmap_png"]))

        # ---------- SAVE TO THE CORRECT SESSION STATE ----------
        if cnmf:
            st.session_state["cnmf_module_order_heatmap"] = png_bytes
        else:
            st.session_state["module_order_heatmap"] = png_bytes

    # ---------- DISPLAY PERSISTENT HEATMAP ----------
    key_name = "cnmf_module_order_heatmap" if cnmf else "module_order_heatmap"

    if key_name in st.session_state and st.session_state[key_name] is not None:
        st.image(st.session_state[key_name])
        st.download_button(
            "Download PNG",
            data=st.session_state[key_name],
            file_name=key_name + ".png",
            mime="image/png",
        )

