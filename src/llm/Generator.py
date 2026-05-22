import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..models import StudentSearchResults, StudentSearchResultsAndAnswer
from ..models import MinimalAnswer, MinimalSource
import re


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading {model_name}...")
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
        for result in search_result.search_results:
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

        sys_content = "You are a technical expert on vLLM. Answer the prompt"
        sys_content += "ONLY by using the given context."
        sys_content += " Tell that you don't know if you can't."
        sys_content += "Go straight to the point but be precise."

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
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(
            model_inputs.input_ids, generated_ids)]

        raw_answer = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

        clean_answer = re.sub(
            r'<think>.*?</think>',
            '',
            raw_answer,
            flags=re.DOTALL)

        return clean_answer.strip()
