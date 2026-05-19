from pydantic import BaseModel
from typing import List
from .MinimalSource import MinimalSource


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]
