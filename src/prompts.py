def build_prompt(question, context):
    return f"""You are AnswerGPT, a helpful AI assistant. Use the following information to answer the question clearly.

{context}

Question: {question}
Answer:"""
