from pydantic import BaseModel


class CreateMultipartRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"


class SignPartRequest(BaseModel):
    key: str
    uploadId: str
    partNumber: int


class CompleteMultipartRequest(BaseModel):
    key: str
    uploadId: str
    parts: list  # [{ "ETag": "...", "PartNumber": 1 }, ...]
