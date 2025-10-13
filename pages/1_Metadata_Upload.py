import streamlit as st
import pandas as pd
from ui_theme import apply_custom_theme
apply_custom_theme()

if "meta" not in st.session_state:
    st.session_state["meta"] = None
if "metadata_index" not in st.session_state:
    st.session_state["metadata_index"] = None

st.subheader("Start by uploading your metadata file. You won't be able to use further features without it.")
st.write("You can come back later and upload a different file if you want to use a different data set later")

tmp_file = st.file_uploader("Upload metadata", type=["csv", "tsv", "txt"])
if st.button("Click to Finish Upload") and tmp_file is not None:
    sep = "," if tmp_file.name.endswith(".csv") else "\t"
    meta = pd.read_csv(tmp_file, sep=sep)
    st.session_state["meta"] = meta

if st.session_state["meta"] is not None:
    st.dataframe(st.session_state["meta"])

if st.session_state["metadata_index"] is not None and st.session_state["metadata_index"] != "":
    st.write("Column name that stores samples:")
    st.write(st.session_state["metadata_index"])

if st.session_state["meta"] is not None:
    st.write("Please, provide the name of the column that stores sample names before proceeding. Use exactly the same name as in the file")
    sample_column = st.text_input("Enter sample name column")

    if st.button("Save"):
        st.session_state["metadata_index"] = sample_column

if st.session_state["metadata_index"] is not None and st.session_state["meta"] is not None and st.session_state["metadata_index"] != "":
        st.page_link("pages/1_Preprocess_Data.py", label="Continue")

