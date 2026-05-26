import json
import os
import fire

from .data_ingestion import ChunksCreator, FilesGetter, KnowledgeBaseBuilder
from .search import SearchEngine, Searcher
from .llm import Generator
import uuid
from .models import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalSearchResults,
    MinimalAnswer,
    MinimalSource
)


class MoulinetteCLI:

    def _check_5_percent_overlap(
            self,
            retrieved_source,
            expected_source: dict) -> bool:
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
        return overlap_ratio >= 0.05

    def _calculate_recall_at_k(
            self,
            predictions: StudentSearchResults,
            ground_truth_path: str) -> float:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        gt_dict = {q["question_id"]: q.get("sources", [])
                   for q in gt_data["rag_questions"]}

        total_score = 0
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

    def index(self, max_chunk_size: int = 2000) -> None:
        if not isinstance(max_chunk_size, int):
            raise ValueError(
                'Invalid max_chunk_size_value: must be an integer')
        print(f"Starting indexation with max chunk size {max_chunk_size}...")
        source = "data/raw/vllm-0.10.1"
        file_getter = FilesGetter()
        extracted_data = file_getter.extract_files_data(source)

        if not extracted_data:
            raise Exception("Extraction error: nothing could be extracted")

        chunker = ChunksCreator(max_chunk_size)
        builder = KnowledgeBaseBuilder(chunker=chunker)
        builder.build_from_files(extracted_data)
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 5) -> None:
        if not isinstance(query, str):
            raise ValueError("Invalid query, must be a string")
        if not isinstance(k, int):
            raise ValueError("Invalid value for k : must be an integer")
        if k <= 0:
            raise ValueError("Invalid value for k : must be > 0")
        engine = SearchEngine(save_dir="data/processed/")
        raw_sources = engine.search_question([query], k=k)[0]
        sources = [MinimalSource(**src) if isinstance(src, dict)
                   else src for src in raw_sources]

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
            k: int = 10,
            save_directory: str = "data/output/search_results") -> None:
        if not isinstance(dataset_path, str):
            raise ValueError("Invalid value for dataset_path, must be a str.")
        if not isinstance(k, int):
            raise ValueError("Invalid value for k : must be an integer")
        if k <= 0:
            raise ValueError("Invalid value for k : must be > 0")
        if not isinstance(save_directory, str):
            raise ValueError("Invalid save_directory, must be a str.")
        print(f"Searching dataset {dataset_path}...")
        engine = SearchEngine(save_dir="data/processed/")
        searcher = Searcher(engine)

        results = searcher.process_dataset(dataset_path, k)

        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(dataset_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(results.model_dump_json(indent=4))

        print(f"Saved student_search_results to {save_path}")

    def answer(self, query: str, k: int = 10) -> None:
        if not isinstance(k, int):
            raise ValueError("Invalid value for k : must be an integer")
        if k <= 0:
            raise ValueError("Invalid value for k : must be > 0")
        if not isinstance(query, str):
            raise ValueError("Invalid query, must be a string")
        engine = SearchEngine(save_dir="data/processed/")
        raw_sources = engine.search_question([query], k=k)[0]

        sources = [MinimalSource(**src) if isinstance(src, dict)
                   else src for src in raw_sources]

        generator = Generator()
        response_text = generator.generate_single_answer(query, sources)

        result = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=str(uuid.uuid4()),
                    question=query,
                    retrieved_sources=sources,
                    answer=response_text
                )
            ],
            k=k
        )

        print(result.model_dump_json(indent=4))

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        if not isinstance(student_search_results_path, str):
            err = "Invalid student_search_results_path, must be a string"
            raise ValueError(err)
        if not isinstance(save_directory, str):
            err = "Invalid save_directory, must be a string"
            raise ValueError(err)

        print(f"Generating answers for {student_search_results_path}...")

        try:
            with open(student_search_results_path, 'r', encoding='utf-8') as f:
                search_data = StudentSearchResults.model_validate_json(
                    f.read())
        except FileNotFoundError:
            err = f"Can't open {student_search_results_path}."
            raise FileNotFoundError(err)
        except Exception:
            err = "'Invalid student_search_results_path "
            err += "(invalid pydantic validation)'"
            raise Exception(err)
        generator = Generator()
        final_results = generator.process_dataset(search_data)

        os.makedirs(save_directory, exist_ok=True)
        filename = os.path.basename(student_search_results_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(final_results.model_dump_json(indent=4))

        print(f"Saved student_search_results_and_answer to {save_path}")

    def evaluate(
            self,
            student_answer_path: str,
            dataset_path: str,
            k: int = 10) -> None:
        print(f"Evaluating search results for {student_answer_path}...")

        with open(student_answer_path, 'r', encoding='utf-8') as f:
            predictions = StudentSearchResults.model_validate_json(f.read())

        recall_score = self._calculate_recall_at_k(predictions, dataset_path)

        print("=" * 40)
        print(f"FINAL RECALL@{k} : {recall_score * 100:.2f}%")
        print("=" * 40)


if __name__ == "__main__":
    try:
        fire.Fire(MoulinetteCLI)
    except Exception as err:
        print(f'Error: {err}')
