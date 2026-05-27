from .StudentSearchResults import StudentSearchResults
from typing import Sequence
from .MinimalAnswer import MinimalAnswer


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: Sequence[MinimalAnswer]
