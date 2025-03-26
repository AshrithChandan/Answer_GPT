from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

def load_phi2():
    base_model_id = "microsoft/phi-2"
    adapter_path = os.path.abspath("D:/ashrith projects/python/answergpt/src/modules/answergpt-qlora")

    print("📁 Loading LoRA adapter from:", adapter_path)

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token  # Required for generation

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Manual text generation function
    def generate(prompt: str) -> str:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=300,
                min_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50
            )


        return tokenizer.decode(output[0], skip_special_tokens=True)

    return generate
