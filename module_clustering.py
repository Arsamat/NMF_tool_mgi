import streamlit as st
import requests
import json
from io import BytesIO


def m_clustering(module_usages, sample_order, n_clusters_mod, cnmf=False):


    buf = BytesIO()
    module_usages.reset_index(drop=False).to_feather(buf)
    buf.seek(0)

    files = {
        "module_usages": ("module.feather", buf, "application/octet-stream")
    }

    data = {
        "sample_order": json.dumps(sample_order),
        "n_clusters": n_clusters_mod
    }

    resp = requests.post(
        st.session_state["API_URL"] + "/cluster_modules/",
        files=files,
        data=data
    )

    if resp.status_code != 200:
        st.error(resp.text)
        return None

    result = resp.json()

    if n_clusters_mod == 0:
        return BytesIO(bytes.fromhex(result["dendrogram_png"]))

    # Dynamic prefix selection
    prefix = "cnmf_" if cnmf else ""

    st.session_state[f"{prefix}module_leaf_order"]    = result["module_leaf_order"]
    st.session_state[f"{prefix}module_cluster_labels"] = result["cluster_labels"]

    dendro_png = BytesIO(bytes.fromhex(result["dendrogram_png"]))
    return dendro_png





