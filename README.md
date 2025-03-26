# 🤖 AnswerGPT

AnswerGPT is a real-time, context-aware, hybrid chatbot powered by:

- 🧠 **Phi-2** — A powerful, lightweight language model by Microsoft
- 📦 **QLoRA Fine-Tuning** — Trained on your own interactions
- 💾 **Qdrant Vector DB** — For memory storage and semantic search
- 🌐 **DuckDuckGo Web Scraping** — To pull live, real-world info
- 🎛️ **Gradio Web UI** — For a slick ChatGPT-style interface

---

## 🚀 Features

✅ Fine-tuned local model (Phi-2 + QLoRA)
✅ Real-time web + memory hybrid answering
✅ Custom prompt formatting
✅ Add long-term memory via `remember` command
✅ Local Gradio UI
✅ Modular and hackable Python code

---

## 🧱 Project Structure

```
answergpt/
├── src/
│   ├── main.py                  # CLI chatbot
│   ├── model_loader.py          # Load base + LoRA model
│   ├── prompts.py               # Prompt formatting
│   ├── modules/
│   │   ├── vector_store.py      # Qdrant memory interface
│   │   ├── web_search.py        # DuckDuckGo scraping
│   │   └── answergpt-qlora/     # Fine-tuned LoRA model folder
├── gradio_ui.py                # Web UI with Gradio
├── data/
│   └── log.jsonl                # Chat logs for future training
├── models/                     # [Optional] model artifacts
├── .gitignore
└── README.md
```

---

## 🛠️ Setup (Local)

### 1. Clone the Repo
```bash
git clone https://github.com/YOUR_USERNAME/answergpt.git
cd answergpt
```

### 2. (Optional) Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> ✅ Or install manually:
```bash
pip install transformers peft qdrant-client sentence-transformers colorama gradio duckduckgo-search
```

### 4. Put Your Fine-Tuned Model in:
```
models/answergpt-qlora/
```
This should contain:
- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer_config.json`, etc.

### 5. Run CLI Chatbot
```bash
python -m src.main
```

### 6. Run Gradio UI
```bash
python gradio_ui.py
```
It will launch in your browser at [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## ✨ Commands

- **remember ...** — Adds a fact to long-term vector memory (Qdrant)
```bash
You: remember Pluto is a dwarf planet
```
- **ask anything** — Hybrid context-aware answer
```bash
You: What is Pluto?
```

---

## 🧠 Training Your Own Model

You can fine-tune Phi-2 with QLoRA using the built-in training pipeline on Colab. Just run:
```python
src/modules/finetune_qlora.py
```
Make sure `data/log.jsonl` has high-quality Q&A logs.

---

## 🌍 Credits
- [Phi-2](https://huggingface.co/microsoft/phi-2) by Microsoft
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Gradio](https://gradio.app/)
- [Qdrant](https://qdrant.tech)

---

## 📄 License
This project is for educational & research purposes. You are responsible for compliance with licenses of pretrained models.

---

## 💬 Contact
Feel free to reach out via GitHub Issues or contribute with a PR! Let's build the future of local AI together 🚀

