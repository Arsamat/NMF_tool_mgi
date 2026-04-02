import os

# limit BLAS threads per worker to avoid deadlocks
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import datetime
import logging
import traceback
import uuid

import boto3
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from infra.llm_request_functions import summarize_traceback

from api.routers import deg, health, heatmaps, metadata, nmf, pathway, precomputed_deg, uploads

app = FastAPI()

ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "https://nmftoolmgi.streamlit.app",
    # or your custom domain if you have one
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = "error.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions, log them, and return a friendly AI-generated summary."""
    error_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().isoformat()
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    log_message = (
        f"ID: {error_id}\n"
        f"Time: {timestamp}\n"
        f"Path: {request.url.path}\n"
        f"Error: {exc}\n"
        f"Traceback:\n{tb}\n"
        f"{'-' * 60}\n"
    )
    logging.error(log_message)

    summary = summarize_traceback(tb)

    return JSONResponse(
        status_code=500,
        content={
            "error_id": error_id,
            "message": summary,
            "details": "This issue has been logged for review.",
        },
    )


cloudwatch = boto3.client("cloudwatch", region_name="us-east-2")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path in ("/",):
        return await call_next(request)

    try:
        cloudwatch.put_metric_data(
            Namespace="FastAPIApp",
            MetricData=[
                {
                    "MetricName": "RequestCount",
                    "Dimensions": [
                        {"Name": "Service", "Value": "FastAPI"},
                    ],
                    "Value": 1,
                    "Unit": "Count",
                }
            ],
        )
    except Exception as e:
        print(f"Failed to push metric: {e}")

    return await call_next(request)


app.include_router(health.router)
app.include_router(nmf.router)
app.include_router(pathway.router)
app.include_router(heatmaps.router)
app.include_router(metadata.router)
app.include_router(deg.router)
app.include_router(uploads.router)
app.include_router(precomputed_deg.router)
