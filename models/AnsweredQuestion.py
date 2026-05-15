from typing import List
from .UnansweredQuestion import UnansweredQuestion
from .MinimalSource import MinimalSource


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
