from pydantic import BaseModel
from typing import Sequence
from .MinimalSearchResults import MinimalSearchResults


class StudentSearchResults(BaseModel):
    search_results: Sequence[MinimalSearchResults]
    k: int
