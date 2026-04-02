import pandas as pd
import os
from pymongo import MongoClient 
import pyarrow as pa
import boto3, io
import pyarrow.parquet as pq
import s3fs
import pyarrow.dataset as ds
import zipfile
import json
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from infra.s3_utils import upload_brb

# Read from environment; never commit the actual URL
MONGO_URI = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("Set MONGODB_URI (or MONGO_URI) in the environment")

client = MongoClient(MONGO_URI)
db = client["brb_seq"]
collection = db["metadata"]
#collection = db["test"]

def get_all_metadata_values():
    result = list(collection.find({}))
    df = pd.DataFrame(result)

    columns = []
    for col in df.columns.tolist():
        if col != "SampleName" and col != "_id":
            columns.append(col)
    
    for col in columns:
        vals = df[col].dropna().unique().tolist()
        types = {type(v) for v in vals}
        if len(types) > 1:
            print(col, types)

    unique_vals = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in columns
        if col != "SampleName" and col != "_id"
    }

    return {
        "columns": columns,
        "unique_values": unique_vals
    }

def query_response(metadata_df, counts_data):
    buffer = io.BytesIO()
    #job_id = upload_brb(counts_data)
    #MODIFY!!!!!
    job_id = "123"
    counts_shape = counts_data.shape
    #n = min(counts_shape[0], counts_shape[1], 100)

    counts_return = counts_data
    #counts_return = counts_data.iloc[:n, :n]

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        # buf = io.BytesIO()
        # counts_return.to_feather(buf)
        # buf.seek(0)
        # z.writestr("counts", buf.read())

        buf2 = io.BytesIO()
        metadata_df.to_feather(buf2)
        buf2.seek(0)
        z.writestr("metadata", buf2.read())

        #CHANGE LATER!!!
        z.writestr("job.json", json.dumps({"job_id": job_id}))
    
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type= "applications/zip",
        headers={"Content-Disposition": 'attachment; filename="db_data'}
    )

def return_metadata(metadata_df):
    buf = io.BytesIO()
    metadata_df.to_feather(buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="db_data.feather"'
        },
    )


def query_metadata(filters):
    print(filters)
    #first obtain all relevant sample names
    hmap = filters["filters"]
    # Build MongoDB query
    mongo_query = {
        key: {"$in": values}
        for key, values in hmap.items()
    }

    # Query metadata collection
    #results_cursor = collection.find(mongo_query, {"_id": 0, "SampleName": 1})
    results_cursor = collection.find(mongo_query)
    results = list(results_cursor)

    if not results:
        raise HTTPException(
            status_code=204,
            detail="No matching samples found"
        )
    
    metadata_df = pd.DataFrame(results)
    metadata_df = metadata_df.drop("_id", axis=1)
      
    #result = [row["SampleName"] for row in results]
    #counts_data =  get_counts_subset(result)
    return return_metadata(metadata_df)    
    #After obtaining all relevant sample names now we can filter the parquet file to get counts table we need

async def query_counts(metadata):
    meta_df = pd.read_feather(io.BytesIO(await metadata.read()))
    result = [row["SampleName"] for index, row in meta_df.iterrows()]
    counts_data =  get_counts_subset(result)

    job_id = upload_brb(counts_data)

    counts_return = counts_data.head()

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        buf = io.BytesIO()
        counts_return.to_feather(buf)
        buf.seek(0)
        z.writestr("counts", buf.read())


        z.writestr("job.json", json.dumps({"job_id": job_id}))
    
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type= "applications/zip",
        headers={"Content-Disposition": 'attachment; filename="db_data'}
    )


def get_counts_subset(sample_names):
    fs = s3fs.S3FileSystem()
    files = fs.glob("brb-seq-data-storage/counts/*.parquet")

    dfs = []

    for file in files:
        schema = pq.read_schema(file, filesystem=fs)
        names = list(schema.names)

        # sample columns that exist in THIS shard
        cols_in_file = [c for c in sample_names if c in names]
        if not cols_in_file:
            continue

        # Try common gene/index column names (if present as a real column)
        candidates = ["__index_level_0__", "Geneid", "gene", "genes", "index", "Unnamed: 0"]
        gene_col = next((c for c in candidates if c in names), None)

        # If gene_col exists as a real column, read it + samples; otherwise read only samples
        cols_to_read = ([gene_col] if gene_col else []) + cols_in_file

        t = pq.read_table(file, columns=cols_to_read, filesystem=fs)
        d = t.to_pandas()

        # ---- Normalize gene index ----
        if gene_col and gene_col in d.columns:
            # gene id came in as a normal column
            d = d.set_index(gene_col)
            print(gene_col)
        else:
            # gene id likely restored as pandas index (or not stored at all)
            # If it's already a meaningful index, keep it.
            # If it's a RangeIndex, then this file truly has no gene id information available.
            if isinstance(d.index, pd.RangeIndex):
                raise ValueError(
                    f"No gene/index column found and pandas index is RangeIndex for file: {file}. "
                    f"Schema columns include: {names[:20]} ..."
                )

        # keep only sample cols (in case gene_col snuck in)
        d = d[cols_in_file]
        dfs.append(d)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, axis=1).fillna(0)
    out = out.groupby(level=0, axis=1).sum()  # collapse duplicate sample columns across shards
    idx_name = out.index.name or "index"   # pandas uses "index" if None when resetting
    out = out.reset_index().rename(columns={idx_name: "Geneid"})

    return out

def get_run_metadata(sample_names):
    docs = list(collection.find({"SampleName": {"$in": sample_names}}))
    df = pd.DataFrame(docs)
    return df[["SampleName", "Run"]]



async def update_database(counts_data, metadata):
    # Step 1 — Read uploaded Feather files into DataFrames
    counts_df = pd.read_feather(io.BytesIO(await counts_data.read()))
    metadata_df = pd.read_feather(io.BytesIO(await metadata.read()))

    #Step 2 — Save metadata rows to MongoDB
    # meta_dic = metadata_df.to_dict(orient="records")
    # collection.insert_many(meta_dic)

    # Step 3 — Convert counts_df into an Arrow table
    table = pa.Table.from_pandas(counts_df, preserve_index=True)

    # Step 4 — Prepare an in-memory buffer for Parquet
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)

    # Step 5 — Upload the parquet file to your S3 bucket
    fs = s3fs.S3FileSystem()

    # Build filename — use sample batch name, timestamp, or similar
    # Example: "counts_batch_20250102_1530.parquet"
    #filename = f"counts_batch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    filename = counts_data.filename
    if ".txt" in filename:
        filename = filename.replace(".txt", ".parquet")
    if ".csv" in filename:
        filename = filename.replace(".csv", ".parquet")

    s3_path = f"brb-seq-data-storage/counts/{filename}"

    with fs.open(s3_path, "wb") as f:
        f.write(buf.read())

    return {
        "status": "success",
        #"inserted_metadata_rows": len(meta_dic),
        "saved_parquet": s3_path
    }



    