import streamlit as st
import plotly.express as px
import pandas as pd

def add_facet_gutters(fig, gap_x=0.02, gap_y=0.05):
    """
    Add spacing between facet subplots by shrinking each subplot's axis domain.
    gap_x / gap_y are fractions of the total figure width/height.
    Works across Plotly versions (no facet_*_spacing needed).
    """
    layout = fig.layout
    x_axes = [k for k in layout if k.startswith("xaxis")]
    y_axes = [k for k in layout if k.startswith("yaxis")]

    for ax_name in x_axes:
        ax = layout[ax_name]
        if hasattr(ax, "domain") and ax.domain:
            d0, d1 = ax.domain
            ax.domain = [d0 + gap_x / 2, d1 - gap_x / 2]

    for ax_name in y_axes:
        ax = layout[ax_name]
        if hasattr(ax, "domain") and ax.domain:
            d0, d1 = ax.domain
            ax.domain = [d0 + gap_y / 2, d1 - gap_y / 2]


def visualize_metadata():
    if "metadata_tmp" not in st.session_state or st.session_state["metadata_tmp"] is None:
        st.subheader("No data to visualize")
        return

    columns_to_display = ["Genotype", "Treatment", "Dose", "Timepoint", "CellType", "Maturity", "Background"]
    cat_cols = [c for c in columns_to_display]
    if not cat_cols:
        st.warning("No categorical-like metadata columns found to plot.")
        return

    st.subheader("Metadata faceted bar chart")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_col = st.selectbox("X axis", cat_cols, index=cat_cols.index("Genotype") if "Genotype" in cat_cols else 0)
    with c2:
        row_col = st.selectbox(
            "Facet rows", ["(none)"] + cat_cols,
            index=(["(none)"] + cat_cols).index("Treatment") if "Treatment" in cat_cols else 0
        )
    with c3:
        col_col = st.selectbox(
            "Facet cols", ["(none)"] + cat_cols,
            index=(["(none)"] + cat_cols).index("CellType") if "CellType" in cat_cols else 0
        )
    with c4:
        color_col = st.selectbox(
            "Color (stack)", ["(none)"] + cat_cols,
            index=(["(none)"] + cat_cols).index("Maturity") if "Maturity" in cat_cols else 0
        )

    # (Optional) small UI controls for spacing / labels (remove if you want fixed values)
    with st.expander("Plot options", expanded=False):
        gap_x = st.slider("Horizontal gap between subplots", 0.0, 0.08, 0.02, step=0.005)
        gap_y = st.slider("Vertical gap between subplots", 0.0, 0.12, 0.05, step=0.005)
        tickangle = st.slider("X tick label angle", -90, 90, 45, step=5)

    # --- Build list of grouping columns ---
    group_cols = [x_col]
    if row_col != "(none)":
        group_cols.insert(0, row_col)
    if col_col != "(none)":
        group_cols.insert(1 if row_col != "(none)" else 0, col_col)
    if color_col != "(none)":
        group_cols.append(color_col)

    df = st.session_state["metadata_tmp"].copy()

    # --- Facet settings + keep facet levels (so we can force empty facet cells to exist) ---
    facet_rows = row_col if row_col != "(none)" else None
    facet_cols = col_col if col_col != "(none)" else None

    row_levels = df[row_col].value_counts(dropna=True).index.tolist() if facet_rows else []
    col_levels = df[col_col].value_counts(dropna=True).index.tolist() if facet_cols else []

    # --- Aggregate counts ---
    counts = (
        df.groupby(group_cols, dropna=False)
          .size()
          .reset_index(name="count")
    )

    # --- Consistent ordering (x + color + facet orders) ---
    category_orders = {}
    category_orders[x_col] = (
        counts.groupby(x_col)["count"].sum().sort_values(ascending=False).index.tolist()
    )
    if color_col != "(none)":
        category_orders[color_col] = (
            counts.groupby(color_col)["count"].sum().sort_values(ascending=False).index.tolist()
        )
    if facet_rows:
        category_orders[row_col] = row_levels
    if facet_cols:
        category_orders[col_col] = col_levels

    # --- Force "empty facet cells" to render (Plotly only creates facets that exist in data) ---
    # Add a dummy 0-count row for each missing (row, col) combination.
    if facet_rows or facet_cols:
        def facet_key(d: pd.DataFrame):
            if facet_rows and facet_cols:
                return set(zip(d[facet_rows], d[facet_cols]))
            if facet_rows:
                return set((v, None) for v in d[facet_rows].unique().tolist())
            return set((None, v) for v in d[facet_cols].unique().tolist())

        existing = facet_key(counts)

        all_row_vals = row_levels if facet_rows else [None]
        all_col_vals = col_levels if facet_cols else [None]

        # Choose a valid x value and (optional) color value for the dummy row
        x_dummy = category_orders[x_col][0] if category_orders.get(x_col) else df[x_col].dropna().iloc[0]
        color_dummy = None
        if color_col != "(none)":
            color_dummy = category_orders[color_col][0] if category_orders.get(color_col) else df[color_col].dropna().iloc[0]

        dummy_rows = []
        for rv in all_row_vals:
            for cv in all_col_vals:
                k = (rv if facet_rows else None, cv if facet_cols else None)
                if k in existing:
                    continue

                row = {x_col: x_dummy, "count": 0}
                if facet_rows:
                    row[facet_rows] = rv
                if facet_cols:
                    row[facet_cols] = cv
                if color_col != "(none)":
                    row[color_col] = color_dummy

                dummy_rows.append(row)

        if dummy_rows:
            counts = pd.concat([counts, pd.DataFrame(dummy_rows)], ignore_index=True)

    # --- Plot ---
    n_rows = len(row_levels) if facet_rows else 1
    fig_height = max(420, 220 * n_rows)

    fig = px.bar(
        counts,
        x=x_col,
        y="count",
        color=None if color_col == "(none)" else color_col,
        facet_row=facet_rows,
        facet_col=facet_cols,
        category_orders=category_orders,
        text="count",
        height=fig_height,
    )

    # Ensure categorical x axis and show tick labels + title on *every* facet
    fig.update_xaxes(type="category")
    fig.for_each_xaxis(lambda ax: ax.update(title_text=x_col, showticklabels=True, tickangle=tickangle))

    # Some Plotly versions also hide inner y tick labels; force them on for consistency
    fig.for_each_yaxis(lambda ay: ay.update(showticklabels=True))

    # Add more separation between facet subplots (no facet_*_spacing needed)
    add_facet_gutters(fig, gap_x=gap_x, gap_y=gap_y)

    # Layout cosmetics
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        bargap=0.25,
        barmode="stack",
    )

    # Clean facet annotation labels: "Treatment=foo" -> "foo"
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Hide text labels for dummy zero-only traces to avoid stray "0"
    for trace in fig.data:
        try:
            ys = list(trace.y) if trace.y is not None else []
            if ys and all((v == 0 or v is None) for v in ys):
                trace.text = None
        except Exception:
            pass

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show aggregated counts table"):
        st.dataframe(counts)








    

