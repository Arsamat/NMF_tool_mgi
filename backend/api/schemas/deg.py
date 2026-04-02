from pydantic import BaseModel


class DEGGroupsRequest(BaseModel):
    group_a: list[str]
    group_b: list[str]


class DEGResearchRequest(BaseModel):
    disease_context: str = "Unknown"
    tissue: str = "Unknown"
    num_genes: int = 10
