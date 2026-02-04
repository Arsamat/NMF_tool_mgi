import streamlit as st
import requests
import pandas as pd
import plotly.express as px

import streamlit as st
import pandas as pd
import plotly.express as px

def visualize_metadata():
    if "metadata_tmp" in st.session_state and st.session_state["metadata_tmp"] is not None:
        meta = st.session_state["metadata_tmp"].copy()

        # --- Basic column choices (only categorical-ish cols by default) ---
        # If you want ALL columns, replace cat_cols with list(meta.columns)
        columns_to_display = ["Genotype", "Treatment", "Dose", "Timepoint", "CellType", "Maturity", "Background"]
        cat_cols = [c for c in columns_to_display]
        if not cat_cols:
            st.warning("No categorical-like metadata columns found to plot.")
            return

        st.subheader("Metadata faceted bar chart")

        # --- Controls ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x_col = st.selectbox("X axis", cat_cols, index=cat_cols.index("Genotype") if "Genotype" in cat_cols else 0)
        with c2:
            row_col = st.selectbox("Facet rows", ["(none)"] + cat_cols,
                                index=(["(none)"] + cat_cols).index("Treatment") if "Treatment" in cat_cols else 0)
        with c3:
            col_col = st.selectbox("Facet cols", ["(none)"] + cat_cols,
                                index=(["(none)"] + cat_cols).index("CellType") if "CellType" in cat_cols else 0)
        with c4:
            color_col = st.selectbox("Color (stack)", ["(none)"] + cat_cols,
                                    index=(["(none)"] + cat_cols).index("Maturity") if "Maturity" in cat_cols else 0)

        # Optional: cap grid size
        #max_rows = st.slider("Max facet rows (most frequent)", 1, 20, 6)
    #max_cols = st.slider("Max facet cols (most frequent)", 1, 20, 6)

        # --- Build list of grouping columns ---
        group_cols = [x_col]
        if row_col != "(none)":
            group_cols.insert(0, row_col)
        if col_col != "(none)":
            group_cols.insert(1 if row_col != "(none)" else 0, col_col)
        if color_col != "(none)":
            group_cols.append(color_col)

        # --- Filter to top N facet levels so the grid stays readable ---
        df = meta
        facet_rows = None
        facet_cols = None

        if row_col != "(none)":
            top_row_vals = df[row_col].value_counts(dropna=True).index
            df = df[df[row_col].isin(top_row_vals)]
            facet_rows = row_col

        if col_col != "(none)":
            top_col_vals = df[col_col].value_counts(dropna=True).index
            df = df[df[col_col].isin(top_col_vals)]
            facet_cols = col_col

        # --- Count rows for each combination ---
        counts = (
            df.groupby(group_cols, dropna=False)
            .size()
            .reset_index(name="count")
        )

        # --- Consistent category ordering (helps across facets) ---
        category_orders = {}
        category_orders[x_col] = (
            counts.groupby(x_col)["count"].sum().sort_values(ascending=False).index.tolist()
        )
        if color_col != "(none)":
            category_orders[color_col] = (
                counts.groupby(color_col)["count"].sum().sort_values(ascending=False).index.tolist()
            )

        # --- Plot ---
        fig = px.bar(
            counts,
            x=x_col,
            y="count",
            color=None if color_col == "(none)" else color_col,
            facet_row=facet_rows,
            facet_col=facet_cols,
            category_orders=category_orders,
            text="count",
            height=180 * (len(df[row_col].value_counts()) if row_col != "(none)" else 1),
        )
        fig.update_xaxes(type="category")

        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            bargap=0.25,
            barmode="stack",  # only matters when color is set
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        st.plotly_chart(fig, use_container_width=True)

        # Optional: show the underlying counts table
        with st.expander("Show aggregated counts table"):
            st.dataframe(counts)
    
    else:
        st.subheader("No data to visualize")


    

