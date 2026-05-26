import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..models import StudentSearchResults, StudentSearchResultsAndAnswer
from ..models import MinimalAnswer, MinimalSource
import re
from tqdm import tqdm


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B",
                 domain: str = "the vLLM codebase"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.domain = domain
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model successfully loaded !")

    def process_dataset(
            self,
            search_result: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        final_answers = []
        for result in tqdm(
            search_result.search_results,
            'Answering batch',
            unit='prompt',
                leave=True):
            answer = self.generate_single_answer(
                result.question, result.retrieved_sources)
            minimal_answer = MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=answer)
            final_answers.append(minimal_answer)
        return StudentSearchResultsAndAnswer(
            search_results=final_answers,
            k=search_result.k
        )

    def generate_single_answer(
            self,
            question: str,
            retrieved_sources: list[MinimalSource]) -> str:
        context_blocks = []
        for i, source in enumerate(retrieved_sources):
            context_blocks.append(
                f"--- Document {i + 1} ---\n{source.content}")
        full_context = "\n\n".join(context_blocks)

        sys_content = (
            f"You are an expert technical assistant specializing in {self.domain}.\n"
            "Your ONLY goal is to extract facts from the provided context.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Pay extreme attention to boolean logic (True/False) and variable names.\n"
            "- Do not reverse cause and effect.\n"
            "- Answer the user's question ONLY by using the provided context.\n"
            "- If the context does not contain the answer, explicitly say 'I don't know'.\n"
            "- Always extract and quote specific values, tuples, or code snippets.\n"
            "- Be brief, precise, and highly technical."
        )

        user_content = f"Context:\n{full_context}\n\nQuestion: {question}"

        messages = [{"role": "system",
                     "content": sys_content},
                    {"role": "user",
                     "content": user_content}]

        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            [text_prompt],
            return_tensors="pt").to(
            self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            temperature=0.1
        )

        generated_ids = [output_ids[len(input_ids):]
                         for input_ids, output_ids in zip(
            model_inputs.input_ids, generated_ids)]

        raw_answer = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

        clean_answer = re.sub(
            r'<think>.*?</think>',
            '',
            raw_answer,
            flags=re.DOTALL)

        return clean_answer.strip()
