from .StudentSearchResults import StudentSearchResults
from typing import List
from .MinimalAnswer import MinimalAnswer


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]
