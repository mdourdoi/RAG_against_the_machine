from pydantic import BaseModel
from typing import List
from .MinimalSearchResults import MinimalSearchResults


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int
