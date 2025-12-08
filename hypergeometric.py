import streamlit as st
import requests
import json
from io import BytesIO


def hypergeom_ui(meta_bytes, module_usages, cluster_labels):
    st.header("🔍 Hypergeometric Enrichment Test")

    if "meta" not in st.session_state:
        st.warning("Upload metadata first.")
        return

    meta = st.session_state["meta"]

    # ---------------------------------------------------------
    # 1. Select cluster
    # ---------------------------------------------------------
    clusters = sorted(set(cluster_labels))
    chosen_cluster = st.selectbox("Select cluster:", clusters)

    # ---------------------------------------------------------
    # 2. Select metadata variables
    # ---------------------------------------------------------
    selected_cols = st.multiselect(
        "Select metadata variables:",
        options=meta.columns.tolist(),
        default=st.session_state["annotations_default"]
    )

    # ---------------------------------------------------------
    # 3. For each column, select values
    # ---------------------------------------------------------
    selected_values = {}

    for col in selected_cols:
        vals = sorted(meta[col].dropna().unique().tolist())
        val = st.selectbox(f"Select value for `{col}`:", vals, key=f"mv_{col}")
        selected_values[col] = val

    # Button to run test
    if st.button("Run Hypergeometric Test"):
        # send module usage dataframe
        buf = BytesIO()
        module_usages.reset_index(drop=False).to_feather(buf)
        buf.seek(0)

        files = {
            "module_usages": ("module.feather", buf, "application/octet-stream"),
            "metadata": ("meta.feather", meta_bytes, "application/octet-stream"),
        }

        data = {
            "metadata_index": st.session_state["metadata_index"],
            "cluster_labels": json.dumps(cluster_labels),
            "cluster_id": int(chosen_cluster),
            "selected_values": json.dumps(selected_values)
        }

        resp = requests.post(st.session_state["API_URL"] + "/hypergeom/", files=files, data=data)

        if resp.status_code == 200:
            result = resp.json()
            st.success(f"p-value: **{result['p_value']:.3e}**")
            #st.json(result)
        else:
            st.error(resp.text)
