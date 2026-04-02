from fastapi import APIRouter, File, Request, UploadFile

from infra.db_utils import get_all_metadata_values, query_counts, query_metadata, update_database

router = APIRouter(tags=["metadata"])


@router.get("/get_metadata/")
async def get_metadata_values():
    return get_all_metadata_values()


@router.post("/get_samples/")
async def get_samples_json(request: Request):
    filters = await request.json()

    result = query_metadata(filters)

    return result


@router.post("/get_counts/")
async def get_counts(metadata: UploadFile = File(...)):
    return await query_counts(metadata)


@router.post("/update_db/")
async def update_db(
    counts_table: UploadFile = File(...),
    metadata: UploadFile = File(...),
):
    result = await update_database(counts_table, metadata)
    return result
