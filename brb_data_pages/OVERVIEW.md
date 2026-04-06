# brb_data_pages Overview

## What Is This?

The `brb_data_pages` folder contains Streamlit UI pages for **browsing, filtering, and downloading BRB-seq data**. BRB-seq (Barcode-based RNA sequencing) is a bulk RNA-seq method, and this module lets researchers explore the dataset without writing any code.

Think of it as a "data portal" — you log in, filter the samples you care about, and download their gene expression counts.

---

## Files at a Glance

```
brb_data_pages/
├── extract_counts_frontend.py   # Main page: filter samples + download counts
├── visualize_data.py            # Interactive chart: explore sample distribution
└── backend_download.py          # Helper: get S3 download links from the backend
```

---

## How It Works: The Main Workflow

### Step 1 — Authenticate
The page is password-protected. Users enter a password (stored in Streamlit secrets) to access the data.

### Step 2 — Wake Up the Server
The data lives on an **EC2 compute instance** that's turned off when not in use (to save costs). When a user first loads the page, it triggers an **AWS Lambda function** to start the EC2 instance. This takes 1–4 minutes.

The page auto-refreshes every 8 seconds, checking if the server is ready.

### Step 3 — Load the Metadata Schema
Once the server is up, the app fetches the list of available metadata columns (e.g., Genotype, Treatment, CellType, Timepoint) and their unique values.

### Step 4 — Filter Samples
The user picks which metadata values they want:
- Which genotypes? (e.g., WT, TDP43)
- Which treatments? (e.g., Rotenone)
- Which cell types?

### Step 5 — Find Matching Samples
The app sends the filters to the backend (`POST /get_samples`), which queries MongoDB and returns a table of matching samples with their metadata.

### Step 6 — Load the Counts Table
The user clicks "Load Counts Table". The backend fetches the raw gene expression counts for the selected samples and returns a downloadable file.

### Step 7 — Download or Continue
- **Download** the counts as a CSV/via S3 presigned link
- **Move to NMF Tool** — passes the data directly to the NMF analysis module without re-uploading

---

## The Visualization Page (`visualize_data.py`)

This generates an **interactive bar chart** using Plotly to explore the metadata distribution.

You can configure:
- **X axis** — e.g., Genotype
- **Facet rows** — e.g., Treatment (creates one row of charts per treatment)
- **Facet columns** — e.g., CellType
- **Color/stack** — e.g., Maturity (stacks bars by maturity level)

This helps answer questions like: "How many WT vs TDP43 iMNs do I have per treatment condition?"

---

## The Download Helper (`backend_download.py`)

A small utility used by other modules too (not just this folder). It asks the backend for a **presigned S3 URL** — a temporary link that lets you download a large file directly from AWS S3.

```python
presigned_download_url(api_base, job_id, data_type)
# → Returns a direct download URL (valid for ~1 hour)
```

---

## How Data Moves Through This Module

```
User (Browser)
    │
    │  1. Password → authenticated
    │  2. Lambda trigger → EC2 wakes up
    │  3. GET /get_metadata/ → load schema
    │  4. Select filters in UI
    │  5. POST /get_samples → metadata table
    │  6. POST /get_counts → counts matrix
    │  7. GET /download_preprocessed_data → S3 link
    ▼
AWS S3 (direct download)
```

---

## Where This Module Is Used

This module isn't just standalone — `backend_download.py` and `visualize_data.py` are **imported and reused** in:
- `deg/` — for downloading data within the DEG analysis workflow
- `nmf_nav/` — for downloading preprocessed data in the NMF workflow

---

## Key Things to Know

- The backend URL is hardcoded to an EC2 IP (`http://18.218.84.81:8000/`)
- The password is stored in Streamlit's secrets management (`st.secrets["PASSWORD"]`)
- Data is transferred as **Feather format** (a fast binary format for DataFrames) and then as **ZIP archives**
- This module does not store any data itself — it's purely a UI that talks to the backend
