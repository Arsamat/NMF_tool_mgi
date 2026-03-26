"""Resolve FastAPI download endpoints that return {\"download_url\": ...} (presigned S3)."""

from __future__ import annotations

import urllib.parse

import requests
import streamlit as st


@st.cache_data(ttl=240, show_spinner=False)
def presigned_download_url(api_base: str, job_id: str, data_type: str) -> str:
    base = api_base.rstrip("/")
    q = urllib.parse.urlencode({"job_id": job_id, "data_type": data_type})
    url = f"{base}/download_preprocessed_data?{q}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    payload = r.json()
    if "download_url" not in payload:
        raise KeyError("Response missing download_url")
    return payload["download_url"]
