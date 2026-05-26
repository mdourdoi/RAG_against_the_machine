from ..models import RagDataset, MinimalSearchResults
from ..models import MinimalSource, StudentSearchResults
from .SearchEngine import SearchEngine
import re


class Searcher:
    def __init__(self, engine: SearchEngine):
        self.engine = engine

    def process_dataset(
            self,
            dataset_path: str,
            k: int) -> StudentSearchResults:
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            err = f"Impossible to open {dataset_path}, "
            err += "file may not exit or not enough privilege."
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
            sources_pydantic = [
                MinimalSource(
                    **source) for source in answers[i]]
            minimal_result = MinimalSearchResults(
                question_id=question_obj.question_id,
                question=question_obj.question,
                retrieved_sources=sources_pydantic
            )
            results.append(minimal_result)
        return StudentSearchResults(search_results=results, k=k)
