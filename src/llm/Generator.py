import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        print(f"Chargement du tokenizer pour {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Chargement du modèle {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Modèle chargé avec succès !")
