import gradio as gr
from src.model_loader import load_phi2
from src.prompts import build_prompt
from src.modules.web_search import get_web_context
from src.modules.vector_store import query_text

# Load your generate() function
generate = load_phi2()

def answergpt_chat(message, history):
    # Step 1: Build context from local + web
    context = "Local Memory:\n" + "\n".join(query_text(message))
    context += "\n\nWeb Results:\n" + get_web_context(message)

    # Step 2: Construct prompt
    prompt = build_prompt(message, context)

    # Step 3: Generate response from model
    raw = generate(prompt)
    output = raw.split("Answer:")[-1].strip() if "Answer:" in raw else raw.strip()


    return output

# Launch Gradio Chat UI
gr.ChatInterface(
    fn=answergpt_chat,
    title="AnswerGPT (Fine-tuned 🧠)",
    description="Ask me anything! I'm powered by Phi-2 + your own memory + real-time web search.",
    chatbot=gr.Chatbot(height=400),
    theme="default",
).launch()
