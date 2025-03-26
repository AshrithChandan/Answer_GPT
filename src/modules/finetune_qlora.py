import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import bitsandbytes as bnb

BASE_MODEL = "microsoft/phi-2"
LOG_FILE = "data/log.jsonl"
OUTPUT_DIR = "output/answergpt-qlora"

def load_data():
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    examples = []
    for item in data:
        prompt = f"You are AnswerGPT. Use the following context to answer the question.\n\nContext:\n{item['context']}\n\nQuestion:\n{item['instruction']}\n\nAnswer:"
        examples.append({"prompt": prompt, "response": item["response"]})

    return Dataset.from_list(examples)

def tokenize(example, tokenizer):
    full_text = example["prompt"] + " " + example["response"]
    tokenized = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def train():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    dataset = load_data()
    dataset = dataset.map(lambda x: tokenize(x, tokenizer))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()
