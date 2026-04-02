import boto3
import uuid 
import tempfile
import os
from fastapi import HTTPException
import pandas as pd
import io
from fastapi.responses import RedirectResponse
import shutil


#upload for pre processing
BUCKET = "nmf-tool-bucket"
REGION = "us-east-2"

s3 = boto3.client("s3", region_name=REGION)

def create_url():
    job_id = str(uuid.uuid4())
    s3_key = f"jobs/{job_id}/counts.csv"

    presigned_url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": BUCKET,
            "Key": s3_key,
            "ContentType": "application/octet-stream"
        },
        ExpiresIn=3600
    )

    return {
        "job_id": job_id,
        "s3_key": s3_key,
        "upload_url": presigned_url
    }

def create_preprocessed_url():
    job_id = str(uuid.uuid4())
    s3_key = f"jobs/{job_id}/preprocessed_counts.csv"

    presigned_url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": BUCKET,
            "Key": s3_key,
            "ContentType": "application/octet-stream"
        },
        ExpiresIn=3600
    )

    return {
        "job_id": job_id,
        "s3_key": s3_key,
        "upload_url": presigned_url
    }    

def download_data_util(job_id, data_type):
    if data_type == "counts":
        s3_key = f"jobs/{job_id}/counts.csv"
    elif data_type == "preprocessed":
       s3_key = f"jobs/{job_id}/preprocessed_counts.csv" 

    try:
        s3.head_object(Bucket=BUCKET, Key=s3_key)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found")

    url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": BUCKET,
            "Key": s3_key,
            "ResponseContentDisposition": 'attachment; filename="preprocessed.csv"',
            "ResponseContentType": "application/octet-stream",
        },
        ExpiresIn=300,
    )

    return RedirectResponse(url=url, status_code=302)

    # try:
    #     try:
    #         s3.download_file(BUCKET, s3_key, preprocessed_path)
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=404,
    #             detail=f"Could not download counts from S3 (Bucket={BUCKET}, Key={s3_key}). Error: {e}"
    #         )
        
    #     df = pd.read_csv(preprocessed_path)

    #     df_buf = io.BytesIO()
    #     df.to_feather(df_buf)
    #     df_buf.seek(0)

    #     return StreamingResponse(
    #         df_buf,
    #         media_type="application/octet-stream",
    #         headers={"Content-Disposition": 'attachment; filename="preprocessed.feather"'}
    #     )
    
    # finally:
    #     # Cleanup temp dir with inputs
    #     if os.path.exists(tmp_dir):
    #         shutil.rmtree(tmp_dir)
    
def upload_brb(df):
    print("Uploading data to s3")
    buf = io.StringIO()
    df.to_csv(buf, index=True)  # index=True if you want rownames preserved
    data = buf.getvalue().encode("utf-8")

    job_id = str(uuid.uuid4())
    s3_key = f"jobs/{job_id}/counts.csv"

    s3.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=data,
        ContentType="text/csv; charset=utf-8",
    )

    print("Done Uplaoding data")
    
    return job_id

