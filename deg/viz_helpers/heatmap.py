"""
DEG expression heatmap: rendered on the backend with MongoDB metadata (seaborn + annotation bars).
"""

from __future__ import annotations

import io
import json

import requests
import streamlit as st

from deg.viz_helpers.common import DEG_API_URL

METADATA_SORT_KEYS = ("CellType", "Treatment", "Genotype", "Timepoint")

def _sort_key_options() -> list[str]:
    return ["Group", *METADATA_SORT_KEYS]


def render_heatmap_expander(
    state_prefix: str,
    widget_prefix: str,
    group_a: list | None = None,
    group_b: list | None = None,
):
    with st.expander("Heatmap", expanded=True):
        hm_df = st.session_state.get(f"{state_prefix}heatmap_df")
        if hm_df is None or hm_df.empty:
            return

        ga = list(group_a or [])
        gb = list(group_b or [])

        st.subheader("Gene expression heatmap")
        st.caption(
            "Gene expression (log₂ CPM) for the top variable genes. "
            "The figure is built on the server using **current MongoDB metadata** for these samples "
            "(not the table loaded in this session). Sample names are not drawn on the axis; "
            "the legend lists annotation values. Leave sort levels as — to use the default bar set "
            "(Group plus any of CellType, Treatment, Genotype, Timepoint found in the database)."
        )

        sort_keys = []

        with st.form("preview_form"):
            sort_keys = st.multiselect(
                "Select annotation columns",
                options=["Group", "CellType", "Treatment", "Genotype", "Timepoint"],
                key=f"{widget_prefix}annotations_widget"
            )

            submitted = st.form_submit_button("Generate Preview")

        api_url = st.session_state.get("deg_api_url", DEG_API_URL).rstrip("/") + "/"

        if not ga or not gb:
            st.warning("Group sample lists are missing; reload heatmap after running DEG with both groups.")
            return

        if submitted:
            with st.spinner("Rendering heatmap on server…"):
                try:
                    csv_buf = io.BytesIO()
                    hm_df.to_csv(csv_buf)
                    csv_buf.seek(0)
                    resp = requests.post(
                        f"{api_url}deg_heatmap_render/",
                        files={"heatmap_csv": ("heatmap_matrix.csv", csv_buf, "text/csv")},
                        data={
                            "group_a_json": json.dumps(ga),
                            "group_b_json": json.dumps(gb),
                            "annotation_cols_json": json.dumps(sort_keys),
                        },
                        timeout=180,
                    )
                    if resp.status_code != 200:
                        try:
                            detail = resp.json().get("detail", resp.text)
                        except Exception:
                            detail = resp.text
                        st.error(f"Heatmap render failed ({resp.status_code}): {detail}")
                        return
                    st.session_state[f"{state_prefix}heatmap_image"] = resp.content
                except requests.exceptions.RequestException as e:
                    st.error(f"Heatmap request failed: {e}")
                    return
        
        
        png_bytes = st.session_state.get(f"{state_prefix}heatmap_image")
        if png_bytes is not None:
            st.image(png_bytes, use_container_width=True)
            st.download_button(
                "Download heatmap (PNG)",
                png_bytes,
                "deg_expression_heatmap.png",
                "image/png",
                key=f"{widget_prefix}heatmap_png_download",
            )
