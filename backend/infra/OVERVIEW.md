# backend/infra/ Overview

## What Is This?

The `infra/` folder is the **infrastructure layer** — it handles all external system connections: MongoDB (metadata storage), AWS S3 (file storage), and LLM APIs (OpenAI for gene descriptions + error summaries).

Every other module in the backend imports from here rather than connecting to these services directly.

---

## Files

```
infra/
├── db_utils.py               # MongoDB queries + S3 parquet reads for count data
├── s3_utils.py               # S3 presigned URLs and file upload/download helpers
└── llm_request_functions.py  # OpenAI API calls for gene descriptions and error summarization
```

---

## File 1: `db_utils.py`

### Purpose
All MongoDB interactions and retrieval of gene count data from S3 parquet shards.

### Setup
- Connects to MongoDB on startup using `MONGODB_URI` or `MONGO_URI` env var
- Database: `brb_seq`, Collection: `metadata`

### Functions

---

#### `get_all_metadata_values() → dict`
- Fetches all documents from the `metadata` MongoDB collection
- Returns `{"columns": [...], "unique_values": {col: [sorted unique values]}}"`
- Excludes `SampleName` and `_id` from unique value lists
- **Used by:** `GET /get_metadata/`

---

#### `query_metadata(filters: dict) → StreamingResponse`
- Builds a MongoDB `$in` query from the filters dict
- Example: `{"Genotype": ["WT", "TDP43"], "Treatment": ["Rotenone"]}`
- Returns 404 if no samples match
- Returns a Feather-format binary file (fast binary DataFrame serialization)
- **Used by:** `POST /get_samples/`

---

#### `query_counts(metadata: UploadFile) → StreamingResponse` *(async)*
- Reads the Feather metadata file uploaded by the frontend
- Extracts `SampleName` column to get the list of requested samples
- Calls `get_counts_subset()` to fetch the actual expression counts
- Calls `upload_brb()` to push the full counts to S3 (for later download)
- Returns a ZIP containing:
  - `counts.feather` — first 5 rows only (preview)
  - `job.json` — contains the `job_id` for downloading the full file from S3
- **Used by:** `POST /get_counts/`

---

#### `get_counts_subset(sample_names: list) → pd.DataFrame`
- The most complex function: retrieves gene expression counts from **S3 parquet shards**
- Uses `s3fs` to glob all `.parquet` files in `brb-seq-data-storage/counts/`
- For each shard, reads only the schema first, then loads only the columns matching requested samples (avoids loading the full matrix)
- Normalizes the gene index column (handles various naming conventions: `Geneid`, `gene`, `genes`, `index`, etc.)
- Concatenates across all shards and collapses duplicate sample columns
- **Returns:** DataFrame where rows = genes (`Geneid` column), columns = sample names
- **Used by:** `query_counts()` and `deg.deg_utils.run_deg_analysis()`

---

#### `get_run_metadata(sample_names: list) → pd.DataFrame`
- Queries MongoDB for the `Run` (sequencing batch) field of each sample
- **Returns:** DataFrame with columns `["SampleName", "Run"]`
- **Used by:** `deg.deg_utils.run_deg_analysis()` to include batch as a covariate in the DEG model

---

#### `update_database(counts_data, metadata) → dict` *(async)*
- Reads uploaded Feather files for counts and metadata
- Converts counts to Apache Arrow Parquet format (snappy compression)
- Uploads the parquet file to `brb-seq-data-storage/counts/`
- **Returns:** `{"status": "success", "saved_parquet": s3_path}`
- **Used by:** `POST /update_db/`

---

#### `return_metadata(metadata_df) → StreamingResponse`
- Serializes a DataFrame to Feather format in memory
- Returns as a streaming file attachment
- Helper used by `query_metadata()`

---

## File 2: `s3_utils.py`

### Purpose
Generates presigned S3 URLs so the frontend can upload/download large files directly to/from S3 without routing through the server. Also handles programmatic file uploads.

### Setup
- S3 bucket: `nmf-tool-bucket`
- Region: `us-east-2`
- Uses boto3 S3 client

### Functions

---

#### `create_url() → dict`
- Generates a new `job_id` (UUID)
- Creates a presigned **PUT** URL valid for 1 hour for: `jobs/{job_id}/counts.csv`
- **Returns:** `{job_id, s3_key, upload_url}`
- **Used by:** `POST /create_upload_url`

---

#### `create_preprocessed_url() → dict`
- Same as `create_url()` but for `jobs/{job_id}/preprocessed_counts.csv`
- **Used by:** `POST /create_preprocessed_upload_url`

---

#### `download_data_util(job_id, data_type) → RedirectResponse`
- Checks if the file exists in S3 (`head_object`); raises 404 if not
- Generates a presigned **GET** URL valid for 5 minutes
- Returns HTTP 302 redirect pointing the browser directly to S3
- `data_type` controls which file is linked: `"counts"` → `counts.csv`, `"preprocessed"` → `preprocessed_counts.csv`
- **Used by:** `GET /download_preprocessed_data`

---

#### `upload_brb(df: pd.DataFrame) → str`
- Converts a DataFrame to CSV in memory
- Generates a new `job_id`
- Uploads CSV directly to S3: `jobs/{job_id}/counts.csv`
- **Returns:** `job_id`
- **Used by:** `db_utils.query_counts()` after fetching the count subset from MongoDB

---

## File 3: `llm_request_functions.py`

### Purpose
Wraps OpenAI API calls for two use cases: describing genes and summarizing Python errors.

### Setup
- Reads `OPENAI_API_KEY` from environment
- Initializes `openai.OpenAI` client

### Functions

---

#### `clean_json_output(raw_output: str) → str`
- Strips markdown code fences (` ```json ... ``` `) that GPT sometimes wraps responses in
- Returns `"{}"` if input is empty
- **Used by:** `model_request()` to ensure parseable JSON

---

#### `model_request(genes_batch) → dict`
- Sends a batch of gene names to GPT with the prompt: *"Give me a brief description (< 20 words) of each gene including disease associations, organelles, and pathways"*
- **Returns:** `{gene_name: description, ...}`
- **Used by:** `pathway.gene_loadings_gpt.gpt_utils()` (run in parallel via ThreadPoolExecutor)

> **Note:** This function currently uses an invalid model name (`"gpt-5-chat-latest"`). It may need updating to a valid OpenAI model ID.

---

#### `summarize_traceback(tb: str) → str`
- Sends a Python traceback string to GPT
- Asks it to generate a 1–2 sentence user-friendly explanation distinguishing user errors from server errors
- Falls back to a generic message if the API call fails
- **Used by:** `fast_api.py` global exception handler — every unhandled backend error gets summarized before being shown to the user

---

## How These Connect to the Rest of the Backend

```
                ┌──────────────────────────────────┐
                │         API Routers              │
                │  (metadata, uploads, deg, nmf)   │
                └──────────┬───────────────────────┘
                           │ imports
                           ▼
         ┌─────────────────────────────────────────────┐
         │                  infra/                     │
         │                                             │
         │  db_utils.py      s3_utils.py    llm_*.py   │
         └─────┬──────────────────┬──────────┬─────────┘
               │                  │          │
               ▼                  ▼          ▼
           MongoDB             AWS S3     OpenAI API
       (metadata, counts)  (files, jobs)  (descriptions)
```

**Which modules import from infra:**
| Importer | Uses |
|---------|------|
| `api/routers/metadata.py` | `db_utils` |
| `api/routers/uploads.py` | `s3_utils` |
| `deg/deg_utils.py` | `db_utils` (get_counts_subset, get_run_metadata) |
| `nmf/nmf_utils.py` | `s3_utils` (download from S3 for NMF runs) |
| `pathway/gene_loadings_gpt.py` | `llm_request_functions` (model_request) |
| `fast_api.py` | `llm_request_functions` (summarize_traceback) |

---

## Environment Variables Required

| Variable | Used by | Purpose |
|----------|---------|---------|
| `MONGODB_URI` or `MONGO_URI` | `db_utils.py` | MongoDB connection string |
| `OPENAI_API_KEY` | `llm_request_functions.py` | OpenAI API access |
| `DEG_RESULTS_BUCKET` | `precomputed_deg_utils.py` | S3 bucket for DEG results (default: `brb-seq-data-storage`) |
| `DEG_RESULTS_REGION` | same | AWS region (default: `us-east-2`) |
