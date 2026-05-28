from ..models import RagDataset, MinimalSearchResults
from ..models import MinimalSource, StudentSearchResults
from .SearchEngine import SearchEngine
import re


class Searcher:
    """Transform a dataset of questions into retrieval results."""

    def __init__(self, engine: SearchEngine):
        """Store the search engine used for dataset retrieval.

        Args:
            engine: Search engine used to query the indexed corpus.
        """
        self.engine = engine

    def process_dataset(
            self,
            dataset_path: str,
            k: int) -> StudentSearchResults:
        """Load a dataset and retrieve sources for each question.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of sources to retrieve per question.

        Returns:
            A Pydantic model containing the retrieval results.
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            err = f"Unable to open {dataset_path}, "
            err += "the file may not exist or permissions may be missing."
            raise Exception(err)
        try:
            inputs_dataset = RagDataset.model_validate_json(content)
        except Exception:
            raise Exception('Invalid dataset (invalid pydantic validation)')
        list_questions = []
        for q in inputs_dataset.rag_questions:
            q_spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', q.question)
            q_clean = re.sub(r'[_./-]', ' ', q_spaced)
            list_questions.append(f"{q.question} {q_clean}")
        answers = self.engine.search_question(list_questions, k=k)
        results = []
        for i, question_obj in enumerate(inputs_dataset.rag_questions):
            sources_pydantic = [MinimalSource.model_validate(
                source) for source in answers[i]]
            minimal_result = MinimalSearchResults(
                question_id=question_obj.question_id,
                question=question_obj.question,
                retrieved_sources=sources_pydantic
            )
            results.append(minimal_result)
        return StudentSearchResults(search_results=results, k=k)
