from src.model_loader import load_phi2
from src.prompts import build_prompt
from src.modules.web_search import get_web_context
from src.modules.vector_store import query_text, add_text
from colorama import Fore, Style
import json
import os

# Load your generate function from fine-tuned model
generate = load_phi2()

# Path for logging Q&A
log_path = os.path.join("data", "log.jsonl")

def get_combined_context(query):
    web_context = get_web_context(query)
    local_contexts = query_text(query)
    combined = "Local Memory:\n" + "\n".join(local_contexts) + "\n\nWeb Results:\n" + web_context
    return combined

def log_interaction(query, context, response):
    os.makedirs("data", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        json.dump({
            "instruction": query,
            "context": context,
            "response": response
        }, f)
        f.write("\n")

def chat():
    print(Fore.GREEN + "🧠 AnswerGPT (Fine-tuned Phi-2 + QLoRA) is ready to chat!" + Style.RESET_ALL)
    while True:
        user_input = input(Fore.CYAN + "\nYou: " + Style.RESET_ALL)
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Store new facts into Qdrant memory
        if user_input.strip().lower().startswith("remember"):
            fact = user_input[len("remember"):].strip()
            if fact:
                add_text(fact)
                print(Fore.MAGENTA + "🧠 Got it! I'll remember that." + Style.RESET_ALL)
            else:
                print(Fore.RED + "⚠️ You didn't give me anything to remember!" + Style.RESET_ALL)
            continue

        # Get context + generate answer
        context = get_combined_context(user_input)
        prompt = build_prompt(user_input, context)
        response = generate(prompt).split("Answer:")[-1].strip()

        # Log and display
        log_interaction(user_input, context, response)
        print(Fore.YELLOW + "\nAnswerGPT:" + Style.RESET_ALL, response)

if __name__ == "__main__":
    chat()
