from pydantic import BaseModel


class DEGResultsGroupsRequest(BaseModel):
    experiment: str


class DEGResultsTermsRequest(BaseModel):
    experiment: str
    group: str
    context: str = ""


class DEGResultsFetchRequest(BaseModel):
    experiment: str
    output_dir: str
    de_csv_path: str
    gsea_barplot_path: str = ""
