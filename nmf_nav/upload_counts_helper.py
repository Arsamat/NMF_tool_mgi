import streamlit as st
import requests
import streamlit.components.v1 as components


def upload_counts():

    # --------------------------------------------
    # STEP 1 — REQUEST PRESIGNED UPLOAD URL
    # --------------------------------------------
    st.subheader("Step 1: Generate Upload Link")
    if st.button("Prepare upload", key="preprocessing_url"):
        API_URL = st.session_state["API_URL"]
        resp = requests.post(f"{API_URL}/create_upload_url")
        if resp.status_code != 200:
            st.error(f"Failed to create upload URL: {resp.status_code}")
            st.stop()

        data = resp.json()
        st.session_state["job_id"] = data["job_id"]
        st.session_state["s3_key"] = data["s3_key"]
        st.session_state["upload_url"] = data["upload_url"]

        st.success("Upload URL created.")
    
    # --------------------------------------------
    # STEP 2 — DIRECT BROWSER → S3 UPLOAD (REAL)
    # --------------------------------------------
    if "upload_url" in st.session_state:
        upload_url = st.session_state["upload_url"]

        st.subheader("Step 2: Upload counts matrix")
        html = f"""
        <div style="font-family: sans-serif;">
        <input type="file" id="fileInput" />

        <div style="margin-top: 12px;">
            <progress id="progress" value="0" max="100" style="width: 100%; height: 18px;"></progress>
        </div>

        <div id="status" style="margin-top: 10px;"></div>

        <script>
            const uploadUrl = "{upload_url}";
            const progressBar = document.getElementById("progress");
            const statusDiv = document.getElementById("status");

            function notifyStreamlit(value) {{
                if (window.Streamlit) {{
                    window.Streamlit.setComponentValue(value);
                }}
            }}

            document.getElementById("fileInput").onchange = () => {{
                const file = document.getElementById("fileInput").files[0];
                if (!file) return;

                statusDiv.innerHTML =
                    "Uploading: <b>" + file.name + "</b> (" + Math.round(file.size / 1e6) + " MB)";

                const xhr = new XMLHttpRequest();

                xhr.upload.onprogress = (event) => {{
                    if (event.lengthComputable) {{
                        progressBar.value = (event.loaded / event.total) * 100;
                    }}
                }};

                xhr.onload = () => {{
                    if (xhr.status === 200) {{
                        statusDiv.innerHTML = "✅ Upload complete!";
                        notifyStreamlit("uploaded");
                    }} else {{
                        statusDiv.innerHTML =
                            "❌ Upload failed. Status: " + xhr.status + "<br/>" + xhr.responseText;
                    }}
                }};

                xhr.onerror = () => {{
                    statusDiv.innerHTML =
                        "❌ Upload error (network/CORS). Check browser console.";
                }};

                xhr.open("PUT", uploadUrl, true);
                xhr.setRequestHeader("Content-Type", "application/octet-stream");
                xhr.send(file);
            }};
        </script>

        <hr style="margin: 8px 0;" />
        </div>
        """

        components.html(
            html,
            height=150
        )
        

def upload_preprocessed_counts():
    # --------------------------------------------
    # STEP 1 — REQUEST PRESIGNED UPLOAD URL
    # --------------------------------------------
    st.subheader("Step 1: Generate Upload Link")
    if st.button("Prepare upload", key="already_preprocessed_url"):
        API_URL = st.session_state["API_URL"]
        resp = requests.post(f"{API_URL}/create_preprocessed_upload_url")
        if resp.status_code != 200:
            st.error(f"Failed to create upload URL: {resp.status_code}")
            st.stop()

        data = resp.json()
        st.session_state["job_id"] = data["job_id"]
        st.session_state["s3_key"] = data["s3_key"]
        st.session_state["upload_url"] = data["upload_url"]

        st.success("Upload URL created.")
    
    # --------------------------------------------
    # STEP 2 — DIRECT BROWSER → S3 UPLOAD (REAL)
    # --------------------------------------------
    if "upload_url" in st.session_state:
        upload_url = st.session_state["upload_url"]

        st.subheader("Step 2: Upload counts matrix")

        html = f"""
        <div style="font-family: sans-serif;">
        <input type="file" id="fileInput" />

        <div style="margin-top: 12px;">
            <progress id="progress" value="0" max="100" style="width: 100%; height: 18px;"></progress>
        </div>

        <div id="status" style="margin-top: 10px;"></div>

        <script>
            const uploadUrl = "{upload_url}";
            const progressBar = document.getElementById("progress");
            const statusDiv = document.getElementById("status");

            function notifyStreamlit(value) {{
                if (window.Streamlit) {{
                    window.Streamlit.setComponentValue(value);
                }}
            }}

            document.getElementById("fileInput").onchange = () => {{
                const file = document.getElementById("fileInput").files[0];
                if (!file) return;

                const isCsv =
                    file.name.toLowerCase().endsWith(".csv") ||
                    file.type === "text/csv";

                if (!isCsv) {{
                    statusDiv.innerHTML = "❌ Only CSV files are allowed.";
                    return;
                }}

                statusDiv.innerHTML =
                    "Uploading: <b>" + file.name + "</b> (" + Math.round(file.size / 1e6) + " MB)";

                const xhr = new XMLHttpRequest();

                xhr.upload.onprogress = (event) => {{
                    if (event.lengthComputable) {{
                        progressBar.value = (event.loaded / event.total) * 100;
                    }}
                }};

                xhr.onload = () => {{
                    if (xhr.status === 200) {{
                        statusDiv.innerHTML = "✅ Upload complete!";
                        notifyStreamlit("uploaded");
                    }} else {{
                        statusDiv.innerHTML =
                            "❌ Upload failed. Status: " + xhr.status + "<br/>" + xhr.responseText;
                    }}
                }};

                xhr.onerror = () => {{
                    statusDiv.innerHTML =
                        "❌ Upload error (network/CORS). Check browser console.";
                }};

                xhr.open("PUT", uploadUrl, true);
                xhr.setRequestHeader("Content-Type", "application/octet-stream");
                xhr.send(file);
            }};
        </script>

        <hr style="margin: 8px 0;" />
        </div>
        """


        components.html(
            html,
            height=220
        )
