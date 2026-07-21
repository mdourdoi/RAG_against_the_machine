import json
import os
import fire

from .data_ingestion import ChunksCreator, FilesGetter, KnowledgeBaseBuilder
from .search import SearchEngine, Searcher
import uuid
from .models import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalSearchResults,
    MinimalAnswer,
    MinimalSource
)


class RAG:
    """Command-line entry points for indexing, searching, answering
      and evaluating."""

    def _check_5_percent_overlap(
            self,
            retrieved_source: MinimalSource,
            expected_source: dict) -> bool:
        """Return whether two sources overlap enough to count as a match.

        Args:
            retrieved_source: Source returned by the retriever.
            expected_source: Ground-truth source description.

        Returns:
            True when the source paths match and the character overlap is at
            least 5 percent of the expected span.
        """
        found_path = retrieved_source.file_path.replace('\\', '/')
        expected_path = expected_source["file_path"].replace('\\', '/')

        if found_path not in expected_path and expected_path not in found_path:
            return False

        start_inter = max(
            retrieved_source.first_character_index,
            expected_source["first_character_index"])
        end_inter = min(
            retrieved_source.last_character_index,
            expected_source["last_character_index"])

        overlap_len = max(0, end_inter - start_inter)

        expected_len = expected_source["last_character_index"] - \
            expected_source["first_character_index"]

        if expected_len == 0:
            return False

        overlap_ratio = overlap_len / expected_len
        return bool(overlap_ratio >= 0.05)

    def _calculate_recall_at_k(
            self,
            predictions: StudentSearchResults,
            ground_truth_path: str) -> float:
        """Compute recall@k for a set of predictions.

        Args:
            predictions: Predicted search results.
            ground_truth_path: Path to the JSON file containing expected
                sources.

        Returns:
            The average recall score across all questions.
        """
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)

            gt_dict = {q["question_id"]: q.get("sources", [])
                       for q in gt_data["rag_questions"]}
        except Exception:
            raise Exception("Invalid verification file.")

        total_score: float = 0
        num_questions = len(predictions.search_results)

        for result in predictions.search_results:
            expected_sources = gt_dict.get(result.question_id, [])
            if not expected_sources:
                continue

            found_count = 0

            for expected in expected_sources:
                is_found = any(
                    self._check_5_percent_overlap(
                        ret_source,
                        expected) for ret_source in result.retrieved_sources)
                if is_found:
                    found_count += 1

            question_score = found_count / len(expected_sources)
            total_score += question_score

        return (total_score / num_questions) if num_questions > 0 else 0.0

    def index(self, max_chunk_size: int = 2000,
              source: str = "data/raw/vllm-0.10.1") -> None:
        """Build the BM25 index from the source.

        Args:
            max_chunk_size: Maximum chunk size used during ingestion.
        """

        if not isinstance(max_chunk_size, int):
            raise ValueError(
                'Invalid max_chunk_size_value: must be an integer')

        print(f"Starting indexation with max chunk size {max_chunk_size}...")
        file_getter = FilesGetter()
        extracted_data = file_getter.extract_files_data(str(source))

        if not extracted_data:
            raise Exception("Extraction error: nothing could be extracted")

        chunker = ChunksCreator(max_chunk_size)
        builder = KnowledgeBaseBuilder(chunker=chunker)
        builder.build_from_files(extracted_data)
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 5) -> None:
        """Search the indexed corpus for a single query and print results.

        Args:
            query: User query to search.
            k: Number of sources to retrieve.
        """

        if not isinstance(k, int):
            raise ValueError("Invalid value for k : must be an integer")
        if k <= 0:
            raise ValueError("Invalid value for k : must be > 0")

        engine = SearchEngine(save_dir="data/processed/")
        raw_sources = engine.search_question([str(query)], k=k)[0]
        sources = [MinimalSource.model_validate(src) for src in raw_sources]

        result = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=str(uuid.uuid4()),
                    question=query,
                    retrieved_sources=sources
                )
            ],
            k=k
        )

        print(result.model_dump_json(indent=4))

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 5,
            save_directory: str = "data/output/search_results") -> None:
        """Run retrieval for every question in a dataset and save the output.

        Args:
            dataset_path: Path to the dataset file.
            k: Number of sources to retrieve per question.
            save_directory: Directory where the JSON output is written.
        """

        if not isinstance(k, int):
            raise ValueError("Invalid value for k : must be an integer")
        if k <= 0:
            raise ValueError("Invalid value for k : must be > 0")

        print(f"Searching dataset {dataset_path}...")
        engine = SearchEngine(save_dir="data/processed/")
        searcher = Searcher(engine)

        results = searcher.process_dataset(str(dataset_path), k)

        os.makedirs(str(save_directory), exist_ok=True)
        filename = os.path.basename(str(dataset_path))
        save_path = os.path.join(str(save_directory),
                                 filename).replace('\\', '/')

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(results.model_dump_json(indent=4))

        print(f"Saved student_search_results to {save_path}")

    def answer(self, query: str, k: int = 5) -> None:
        """Retrieve sources for one question and generate an answer.

        Args:
            query: User question to answer.
            k: Number of sources to retrieve before generation.
        """
        if not isinstance(k, int):
            raise ValueError("Invalid value for k : must be an integer")
        if k <= 0:
            raise ValueError("Invalid value for k : must be > 0")

        engine = SearchEngine(save_dir="data/processed/")
        raw_sources = engine.search_question([str(query)], k=k)[0]

        sources = [MinimalSource.model_validate(src) for src in raw_sources]

        from .llm import Generator

        generator = Generator()
        response_text = generator.generate_single_answer(str(query), sources)

        result = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=str(uuid.uuid4()),
                    question=str(query),
                    retrieved_sources=sources,
                    answer=response_text
                )
            ],
            k=k
        )

        print(result)

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        """Generate answers for a saved search-results dataset.

        Args:
            student_search_results_path: Path to the search-results JSON file.
            save_directory: Directory where the answered dataset is written.
        """

        print(f"Generating answers for {student_search_results_path}...")

        try:
            with open(
                str(student_search_results_path),
                    'r', encoding='utf-8') as f:
                search_data = StudentSearchResults.model_validate_json(
                    f.read())
        except FileNotFoundError:
            err = f"Can't open {str(student_search_results_path)}."
            raise FileNotFoundError(err)
        except Exception:
            err = "'Invalid student_search_results_path "
            err += "(invalid pydantic validation)'"
            raise Exception(err)

        from .llm import Generator

        generator = Generator()
        final_results = generator.process_dataset(search_data)

        os.makedirs(str(save_directory), exist_ok=True)
        filename = os.path.basename(str(student_search_results_path))
        save_path = os.path.join(
            str(save_directory),
            filename).replace(
            '\\',
            '/')

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(final_results.model_dump_json(indent=4))

        print(f"Saved student_search_results_and_answer to {save_path}")

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str) -> None:
        """Evaluate recall@k for a saved answer file with the k-value
        in the file.

        Args:
            student_answer_path: Path to the predicted answers JSON file.
            dataset_path: Path to the ground-truth dataset JSON file.
        """
        print(f"Evaluating results for {student_search_results_path}...")

        with open(student_search_results_path, 'r', encoding='utf-8') as f:
            try:
                predictions = StudentSearchResults.model_validate_json(
                    f.read())
            except Exception:
                err = "Invalid student_answer_path file "
                err += "(pydantic validation error)"
                raise Exception(err)

        recall_score = self._calculate_recall_at_k(predictions, dataset_path)

        print("=" * 40)
        print(f"FINAL RECALL@{predictions.k} : {recall_score * 100:.2f}%")
        print("=" * 40)


if __name__ == "__main__":
    try:
        fire.Fire(RAG)
    except Exception as err:
        print(f'Error: {err}')
