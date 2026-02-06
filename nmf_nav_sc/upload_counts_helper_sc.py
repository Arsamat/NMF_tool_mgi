import streamlit as st
import requests
import streamlit.components.v1 as components
import json


def upload_counts():
    st.session_state["MULTIPART_API_BASE"] = "https://dckrsgoq45.execute-api.us-east-2.amazonaws.com"


    # --------------------------------------------
    # STEP 1 — REQUEST PRESIGNED UPLOAD URL
    # --------------------------------------------
    st.subheader("Step 1: Generate Upload Link")
    if st.button("Prepare upload", key="preprocessing_url"):
        API_URL = st.session_state["API_URL_sc"]
        resp = requests.get(f"{API_URL}/create_multipart_upload")
        if resp.status_code != 200:
            st.error(f"Failed to create upload URL: {resp.status_code}")
            st.stop()

        data = resp.json()
        st.session_state["job_id_sc"] = data["job_id"]
        st.session_state["key_upload_sc"] = data["key"]
        st.session_state["upload_id_sc"] = data["uploadId"]

        st.success("Upload URL created.")
    
    # --------------------------------------------
    # STEP 2 — DIRECT BROWSER → S3 UPLOAD (REAL)
    # --------------------------------------------
    if "upload_id_sc" in st.session_state:

        st.subheader("Step 2: Upload counts matrix")
        html = f"""
                <div style="font-family: sans-serif;">
                <input type="file" id="fileInput" />

                <div style="margin-top: 12px;">
                    <progress id="progress" value="0" max="100" style="width: 100%; height: 18px;"></progress>
                </div>

                <div id="status" style="margin-top: 10px;"></div>

                <script>
                    // Normalize API base to ALWAYS end with exactly one "/"
                    const apiBase = "{st.session_state["MULTIPART_API_BASE"].rstrip("/")}/";

                    const progressBar = document.getElementById("progress");
                    const statusDiv = document.getElementById("status");

                    async function postJSON(url, body) {{
                    const r = await fetch(url, {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: JSON.stringify(body)
                    }});
                    if (!r.ok) throw new Error(await r.text());
                    return r.json();
                    }}

                    function sleep(ms) {{
                    return new Promise((resolve) => setTimeout(resolve, ms));
                    }}

                    async function putWithTimeout(url, body, timeoutMs) {{
                    const controller = new AbortController();
                    const timer = setTimeout(() => controller.abort(), timeoutMs);
                    try {{
                        return await fetch(url, {{
                        method: "PUT",
                        body: body,
                        signal: controller.signal
                        }});
                    }} finally {{
                        clearTimeout(timer);
                    }}
                    }}

                    document.getElementById("fileInput").onchange = async () => {{
                    const file = document.getElementById("fileInput").files[0];
                    if (!file) return;

                    statusDiv.innerHTML = "Initializing multipart upload for <b>counts.csv</b>…";

                    try {{
                        const key = "{st.session_state["key_upload_sc"]}";
                        const uploadId = "{st.session_state["upload_id_sc"]}";
                        const job_id = "{st.session_state["job_id_sc"]}";

                        // --------------------------------------------------
                        // Multipart upload parameters
                        // --------------------------------------------------
                        const PART_SIZE = 128 * 1024 * 1024; // 128 MB
                        const CONCURRENCY = 6;

                        const totalParts = Math.ceil(file.size / PART_SIZE);
                        let uploadedBytes = 0;
                        const etags = new Array(totalParts);
                        const queue = Array.from(Array(totalParts).keys());

                        let fatalError = null;

                        async function uploadOne(partIndex) {{
                        const partNumber = partIndex + 1;
                        const start = partIndex * PART_SIZE;
                        const end = Math.min(start + PART_SIZE, file.size);
                        const blob = file.slice(start, end);

                        const MAX_RETRIES = 4;

                        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {{
                            try {{
                            const signed = await postJSON(
                                apiBase + "sign_part",
                                {{
                                key: key,
                                uploadId: uploadId,
                                partNumber: partNumber
                                }}
                            );

                            // 10 minutes per part timeout (tune if needed)
                            const resp = await putWithTimeout(signed.url, blob, 10 * 60 * 1000);

                            if (!resp.ok) {{
                                throw new Error("Part " + partNumber + " failed (HTTP " + resp.status + ")");
                            }}

                            // NOTE: This requires S3 CORS to ExposeHeaders: ["ETag"]
                            const etag = resp.headers.get("ETag");
                            if (!etag) {{
                                throw new Error(
                                "Missing ETag for part " + partNumber +
                                ". Check S3 CORS: ExposeHeaders must include ETag."
                                );
                            }}

                            etags[partIndex] = {{
                                ETag: etag,
                                PartNumber: partNumber
                            }};

                            uploadedBytes += (end - start);
                            progressBar.value = (uploadedBytes / file.size) * 100;
                            statusDiv.innerHTML =
                                "Uploading counts.csv: " +
                                (uploadedBytes / 1e9).toFixed(2) +
                                " / " +
                                (file.size / 1e9).toFixed(2) +
                                " GB";

                            return; // success
                            }} catch (err) {{
                            // Retry with backoff unless it's the final attempt
                            if (attempt === MAX_RETRIES) throw err;
                            await sleep(500 * attempt * attempt);
                            }}
                        }}
                        }}

                        async function worker() {{
                        while (queue.length && !fatalError) {{
                            const idx = queue.shift();
                            try {{
                            await uploadOne(idx);
                            }} catch (err) {{
                            fatalError = err;
                            }}
                        }}
                        }}

                        await Promise.all(Array.from({{ length: CONCURRENCY }}, () => worker()));
                        if (fatalError) throw fatalError;

                        // --------------------------------------------------
                        // Complete multipart upload
                        // --------------------------------------------------
                        statusDiv.innerHTML = "Finalizing upload…";

                        await postJSON(
                        apiBase + "complete_multipart_upload",
                        {{
                            key: key,
                            uploadId: uploadId,
                            parts: etags
                        }}
                        );

                        progressBar.value = 100;
                        statusDiv.innerHTML = "✅ counts.csv uploaded successfully!";

                        if (window.Streamlit) {{
                        window.Streamlit.setComponentValue({{
                            status: "uploaded",
                            job_id: job_id,
                            s3_key: key
                        }});
                        }}
                    }} catch (err) {{
                        console.error(err);
                        statusDiv.innerHTML = "❌ Upload failed: " + err.message;
                    }}
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
        API_URL = st.session_state["API_URL_sc"]
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
