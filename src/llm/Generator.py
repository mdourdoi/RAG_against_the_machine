import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..models import StudentSearchResults, StudentSearchResultsAndAnswer
from ..models import MinimalAnswer


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Chargement du modèle {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Modèle chargé avec succès !")

    def process_dataset(
            self,
            search_result: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        final_answers = []
        for result in search_result.search_results:
            answer = self._generate_single_answer(
                result.question, result.retrieved_sources)
            minimal_answer = MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_source=result.retrieved_sources,
                answer=answer)
            final_answers.append(minimal_answer)
        return StudentSearchResultsAndAnswer(
            search_results=final_answers,
            k=search_result.k
        )
