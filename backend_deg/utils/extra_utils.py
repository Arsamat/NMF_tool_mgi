import pandas as pd
import io
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_request_functions import model_request
from fastapi.encoders import jsonable_encoder

#break data into chunks to send to chatGPT
def break_chunks(genes, size):
    for i in range(0, len(genes), size):
        yield genes[i: i + size]

#Make request to ChatGPT to summarise gene functions
async def gpt_utils(
    file,
    top_n
):
    # Read feather file into DataFrame
    content = await file.read()
    df = pd.read_feather(io.BytesIO(content))
    if df is not None:
        if df.columns[0] == "Module":
            tmp = list(df["Module"])
            modules = ["Module_" + str(x) for x in tmp]
            df = df.drop("Module", axis= 1)
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


    #if "Gene" not in df.columns:
     #   return JSONResponse(status_code=400, content={"error": "DataFrame must have a 'Gene' column"})

    hmap = {}
    all_top_genes = set()
    for module in df.columns:
        if module == "Gene":
            continue
        tmp = df[["Gene", module]].copy()
        tmp = tmp.sort_values(by=module, ascending=False).head(top_n)
        hmap[module] = tmp
        all_top_genes.update(tmp["Gene"])

    # Parallel API calls
    output = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(model_request, batch) for batch in break_chunks(list(all_top_genes), 20)]
        for future in as_completed(futures):
            try:
                result = future.result()
                output.update(result)
            except Exception as e:
                print("API error:", e)

    # Attach descriptions back to each module
    results = {}
    for module, tmp in hmap.items():
        tmp = tmp.copy()
        tmp["Description"] = tmp["Gene"].map(output)
        results[module] = tmp.to_dict(orient="records")

    return JSONResponse(content=jsonable_encoder(results))