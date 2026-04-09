import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from infra.llm_request_functions import model_request


def break_chunks(genes, size):
    for i in range(0, len(genes), size):
        yield genes[i : i + size]


async def gpt_utils(file, top_n):
    content = await file.read()
    df = pd.read_feather(io.BytesIO(content))
    if df is not None:
        if df.columns[0] == "Module":
            tmp = list(df["Module"])
            modules = ["Module_" + str(x) for x in tmp]
            df = df.drop("Module", axis=1)
            df = df.T
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "Gene"})
            df.columns = ["Gene"] + modules

        else:
            modules = list(df.index.astype(str))
            df = df.T
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "Gene"})
            df.columns = ["Gene"] + modules

    print(df.head)

    hmap = {}
    all_top_genes = set()
    for module in df.columns:
        if module == "Gene":
            continue
        tmp = df[["Gene", module]].copy()
        tmp = tmp.sort_values(by=module, ascending=False).head(top_n)
        hmap[module] = tmp
        all_top_genes.update(tmp["Gene"])

    output = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(model_request, batch) for batch in break_chunks(list(all_top_genes), 20)]
        for future in as_completed(futures):
            try:
                result = future.result()
                output.update(result)
            except Exception as e:
                print("API error:", e)

    results = {}
    for module, tmp in hmap.items():
        tmp = tmp.copy()
        tmp["Description"] = tmp["Gene"].map(output).fillna("No description available")
        results[module] = tmp.to_dict(orient="records")

    return JSONResponse(content=jsonable_encoder(results))
