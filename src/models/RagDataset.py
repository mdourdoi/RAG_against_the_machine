from pydantic import BaseModel
from typing import List
from .AnsweredQuestion import AnsweredQuestion
from .UnansweredQuestion import UnansweredQuestion


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
