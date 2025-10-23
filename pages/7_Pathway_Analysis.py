import streamlit as st
import requests
import pandas as pd
import io
import zipfile
from ui_theme import apply_custom_theme
apply_custom_theme()

st.set_page_config(page_title="Pathway Visualization", layout="wide")
if "zip_file" not in st.session_state:
    st.session_state["zip_file"] = None

st.title("🧬 Pathway Visualization")
st.write("Upload your **gene_spectra** file obtained from running NMF or a gene list from DEG analysis.")

# --- Upload file ---
uploaded_file = st.file_uploader("Upload your file here", type=["txt", "tsv", "csv"])
to_send = None
format_type=None
choice=None
payload = {}

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    content = uploaded_file.getvalue()

    # --- Auto-detect separator ---
    detected_sep = None
    for sep in [",", "\t", ";", " "]:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep)
            if len(df.columns) > 1:
                detected_sep = sep
                break
        except Exception:
            continue
    if detected_sep is None:
        st.error("❌ Could not detect file delimiter automatically.")
        st.stop()

    st.dataframe(df.head())

    # --- Choose input type ---
    choice = st.radio(
        "How did you obtain this file?",
        ("From NMF", "From DEG analysis"),
        horizontal=True, 
        index=None
    )

    # --- Case 1: DEG Analysis ---
    if choice == "From DEG analysis":
        format_type = st.radio(
            "What format are your gene names?",
            ("SYMBOL", "ENSEMBL", "ENTREZ"),
            horizontal=True
        )
        payload["gene_format"] = format_type

        gene_column = st.selectbox(
            "Select the column that contains gene names:",
            options=df.columns.tolist(),
            index=0
        )

        fold_change_column = st.selectbox(
            "Select the column that contains fold change information:",
            options=df.columns.tolist(),
            index=1
        )

        st.write("✅ Preview of selected columns:")
        st.dataframe(df[[gene_column, fold_change_column]].head())

        to_send = df[[gene_column, fold_change_column]]
        to_send = to_send.rename(columns={gene_column: "Gene", fold_change_column: "Value"})

    # --- Case 2: NMF Analysis ---
    elif choice == "From NMF":
        if "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "Module"}, inplace=True)

        module = st.selectbox(
            "Select which module you want to analyze:",
            options=df["Module"].tolist()
        )

        tmp = df[df["Module"] == module].drop(columns=["Module"])
        tmp = tmp.T.reset_index(drop=False)
        tmp.columns = ["Gene", "Value"]
        to_send = tmp
        payload["gene_format"] = "SYMBOL"
        # replace with subset for submission

    # --- Run pathway analysis ---
    if st.button("🚀 Run Pathway Analysis") and uploaded_file is not None:
        if choice == "From DEG analysis" and format_type == None:
            st.error("Plase select gene format type")
            st.stop()
        with st.spinner("Running pathway analysis... please wait..."):
            try:
                # Convert the processed dataframe into a temporary buffer
                buffer = io.StringIO()
                to_send.to_csv(buffer, index=False)
                buffer.seek(0)

                # Build request
                files = {"file": (uploaded_file.name, buffer.getvalue(), "text/csv")}
                response = requests.post(
                    st.session_state["API_URL"] + "run_pathview/",
                    files=files,
                    data=payload,  # <--- send metadata here
                    timeout=600
                )

                if response.status_code == 200:
                    z = zipfile.ZipFile(io.BytesIO(response.content))
                    st.session_state["zip_file"] = z

                    st.success("✅ Pathway analysis complete!")
                else:
                    st.error(f"❌ Request failed with status {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error during request: {e}")

if st.session_state["zip_file"] is not None:
    #Display and download pathway data frame
    z = st.session_state["zip_file"]
    with z.open("kegg_dataframe.csv") as f:
        df = pd.read_csv(f)
    st.dataframe(df.head())

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)

    st.download_button(
        "Press to Download",
        csv,
        "pathways.csv",
        "text/csv",
        key='download-csv'
    )
    #Download zip file with pathway images
    zip_filename = "pathway_results.zip"
    inner_zip_bytes = io.BytesIO(z.read(zip_filename))
    st.download_button(
        label="⬇️ Download Processed Results (ZIP)",
        data=inner_zip_bytes.getvalue(),
        file_name="results.zip",
        mime="application/zip"
    )
