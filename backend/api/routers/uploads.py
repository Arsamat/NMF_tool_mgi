import uuid

import boto3
from fastapi import APIRouter, Query

from api.schemas.uploads import CompleteMultipartRequest, SignPartRequest
from infra.s3_utils import create_preprocessed_url, create_url, download_data_util

router = APIRouter(tags=["uploads"])

BUCKET = "nmf-tool-bucket"
REGION = "us-east-2"

s3 = boto3.client("s3", region_name=REGION)


@router.post("/create_upload_url")
def create_upload_url():
    return create_url()


@router.post("/create_preprocessed_upload_url")
def create_preprocessed_upload():
    return create_preprocessed_url()


@router.get("/download_preprocessed_data")
def download_data(job_id: str = Query(...), data_type: str = Query(...)):
    return download_data_util(job_id, data_type)


@router.get("/create_multipart_upload")
def create_multipart_upload():
    job_id = str(uuid.uuid4())
    key = f"jobs/{job_id}/counts.csv"

    resp = s3.create_multipart_upload(
        Bucket=BUCKET,
        Key=key,
    )
    return {"job_id": job_id, "key": key, "uploadId": resp["UploadId"]}


@router.post("/sign_part")
def sign_part(req: SignPartRequest):
    url = s3.generate_presigned_url(
        ClientMethod="upload_part",
        Params={
            "Bucket": BUCKET,
            "Key": req.key,
            "UploadId": req.uploadId,
            "PartNumber": req.partNumber,
        },
        ExpiresIn=3600,
    )
    return {"url": url}


@router.post("/complete_multipart_upload")
def complete_multipart_upload(req: CompleteMultipartRequest):
    resp = s3.complete_multipart_upload(
        Bucket=BUCKET,
        Key=req.key,
        UploadId=req.uploadId,
        MultipartUpload={"Parts": req.parts},
    )
    return {"location": resp.get("Location"), "etag": resp.get("ETag")}
