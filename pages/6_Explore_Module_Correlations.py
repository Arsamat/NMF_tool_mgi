import streamlit as st
import pandas as pd
import requests
import json
import io
from ui_theme import apply_custom_theme
apply_custom_theme()

st.subheader("Get correlation statistic for each module")
st.markdown('''
        Some NMF gene modules are more similar than others, in terms of their ranking of genes by contribution scores. 
        Use this tool to calculate the pairwise Spearman correlations between your gene modules.
            ''')
if "correlation_values" not in st.session_state:
    st.session_state["correlation_values"] = None

if "preprocessed_feather" in st.session_state:
    k = st.number_input("k", 2, 30, 7)
    #design_factor = st.text_input("Design Factor")

    files = {
        "preprocessed": (
            "preprocessed.feather",
            st.session_state["preprocessed_feather"],
            "application/octet-stream",
        )
    }

    data = {"k": k, "design_factor": "Group"}

    if st.button("Get Correlation Statistic"):
        with st.spinner("Getting Information..."):
            r = requests.post(st.session_state["API_URL"] + "explore_correlations", files=files, data=data)
            r.raise_for_status()

            data = json.loads(r.content)
            k = list(data["pairs_by_k"].keys())[0]  
            records = data["pairs_by_k"][k]

            st.session_state["correlation_values"] = records

if st.session_state["correlation_values"] is not None:
    df = pd.DataFrame(st.session_state["correlation_values"])
    st.dataframe(df)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"correlation_k{k}.csv",
        mime="text/csv"
    )


